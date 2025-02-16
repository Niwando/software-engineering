#!/usr/bin/env python3
import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from datetime import timedelta, datetime, time as dtime
from dotenv import load_dotenv
from supabase import create_client, Client

def generate_and_insert_predictions():
    load_dotenv()
    # Set up Supabase connection.
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set as environment variables.")
    supabase = create_client(url, key)

    # Model definition.
    class MultiStepLSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, forecast_horizon, dropout=0.2):
            super(MultiStepLSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.forecast_horizon = forecast_horizon
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, forecast_horizon)
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out

    features_to_normalize = ['open', 'high', 'low', 'close', 'volume',
                               'sin_hour', 'cos_hour', 'sin_minute', 'cos_minute',
                               'day_of_week', 'day_of_month', 'month', 'day_of_year', 'year']
    INPUT_SIZE = len(features_to_normalize)
    HIDDEN_SIZE = 128
    NUM_LAYERS = 1
    MODEL_FORECAST_HORIZON = 60  
    TARGET_HORIZON = 390          # chain predictions to cover 390 minutes (full trading day)
    WINDOW_SIZE = 60              # use 60 minutes of data to predict the next 60 minutes              
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  Query the DB for recent data.
    stocks_query = supabase.table("stock_data").select("symbol").execute()
    stocks_data = stocks_query.dict().get("data", [])
    # Get unique symbols.
    symbols = list({row["symbol"] for row in stocks_data})

    for symbol in symbols:
        print(f"\n=== Predicting for symbol: {symbol} ===")
        scaler_path = f"src/lstm/scalers/scaler_minute_{symbol}.pkl"
        if not os.path.exists(scaler_path):
            print(f"Scaler file {scaler_path} not found for {symbol}. Skipping.")
            continue
        model_file = f"src/lstm/models/trained_model_{symbol}.pth"
        if not os.path.exists(model_file):
            print(f"Model file {model_file} not found for {symbol}. Skipping.")
            continue

        # Load the pre-trained model.
        model = MultiStepLSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, MODEL_FORECAST_HORIZON, dropout=0.2)
        try:
            model.load_state_dict(torch.load(model_file, map_location=device))
        except Exception as e:
            print(f"Error loading model for {symbol}: {e}")
            continue
        model.to(device)
        model.eval()
        print(f"Model loaded for {symbol}.")

        scaler = joblib.load(scaler_path)

        # Query the DB for recent data for this symbol.
        try:
            res = supabase.table("stock_data")\
                .select("*")\
                .eq("symbol", symbol)\
                .order("timestamp", desc=True)\
                .limit(WINDOW_SIZE)\
                .execute()
            records = res.dict().get("data", [])
            if len(records) < WINDOW_SIZE:
                print(f"Not enough data in DB for {symbol}. Require at least {WINDOW_SIZE} records. Skipping.")
                continue
            df = pd.DataFrame(records)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.sort_values("timestamp", inplace=True)
        except Exception as e:
            print(f"Error retrieving data for {symbol}: {e}")
            continue

        # Use the last WINDOW_SIZE rows as the starting chain.
        initial_chain = df.iloc[-WINDOW_SIZE:][features_to_normalize].values
        current_chain = torch.tensor(initial_chain, dtype=torch.float32).to(device)

        # Determine the next market open based on the last available timestamp.
        last_timestamp = df["timestamp"].iloc[-1]
        # Compute next day.
        next_day = last_timestamp.date() + timedelta(days=1)
        # Skip weekends: if next_day is Saturday (5) or Sunday (6), increment until it's a weekday.
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        market_open = pd.Timestamp.combine(next_day, dtime(9, 30))
        print(f"Last timestamp for {symbol}: {last_timestamp}. Next market open: {market_open}")

        predictions = []  
        total_predicted = 0

        while total_predicted < TARGET_HORIZON:
            with torch.no_grad():
                pred_norm = model(current_chain.unsqueeze(0)).cpu().numpy().flatten()
            # Inverse-transform only the "close" value
            placeholder = np.zeros((len(pred_norm), INPUT_SIZE))
            placeholder[:, 3] = pred_norm
            pred_original = scaler.inverse_transform(placeholder)[:, 3]
            remaining = TARGET_HORIZON - total_predicted
            use_length = min(len(pred_norm), remaining)
            for i in range(use_length):
                forecast_time = market_open + timedelta(minutes=total_predicted + i)
                predictions.append({
                    "time": forecast_time.isoformat(),
                    "predicted_close": float(pred_original[i]),
                    "symbol": symbol
                })
            total_predicted += use_length

            # Build new rows for chaining.
            new_rows = []
            for i in range(len(pred_norm)):
                ft = market_open + timedelta(minutes=total_predicted - len(pred_norm) + i)
                # Use predicted close for price fields.
                last_chain_row = current_chain[-1].cpu().numpy().reshape(1, -1)
                last_volume = scaler.inverse_transform(last_chain_row)[0, 4]
                row = {
                    "open": pred_original[i],
                    "high": pred_original[i],
                    "low": pred_original[i],
                    "close": pred_original[i],
                    "volume": last_volume,
                    "hour": ft.hour,
                    "minute": ft.minute,
                    "sin_hour": np.sin(2 * np.pi * ft.hour / 24),
                    "cos_hour": np.cos(2 * np.pi * ft.hour / 24),
                    "sin_minute": np.sin(2 * np.pi * ft.minute / 60),
                    "cos_minute": np.cos(2 * np.pi * ft.minute / 60),
                    "day_of_week": ft.weekday(),
                    "day_of_month": ft.day,
                    "month": ft.month,
                    "day_of_year": ft.timetuple().tm_yday,
                    "year": ft.year
                }
                new_rows.append(row)
            new_rows_arr = scaler.transform(
                np.array([[row[col] for col in features_to_normalize] for row in new_rows])
            )
            updated_chain = np.concatenate([current_chain.cpu().numpy(), new_rows_arr], axis=0)
            current_chain = torch.tensor(updated_chain[-WINDOW_SIZE:], dtype=torch.float32).to(device)

        # Insert predictions into the DB.
        BATCH_SIZE = 50
        MAX_RETRIES = 3
        num_preds = len(predictions)
        print(f"Inserting {num_preds} prediction rows for {symbol} into DB.")
        for start in range(0, num_preds, BATCH_SIZE):
            batch = predictions[start:start+BATCH_SIZE]
            retries = 0
            while retries < MAX_RETRIES:
                try:
                    res = supabase.table("predicted_values").insert(batch, upsert=True).execute()
                    resp = res.dict()
                    if resp.get("error"):
                        print(f"Error inserting prediction batch for {symbol} starting at row {start}: {resp['error']}")
                    else:
                        print(f"Inserted prediction rows {start} to {start+len(batch)-1} for {symbol}")
                    break
                except Exception as ex:
                    retries += 1
                    print(f"Exception inserting prediction batch for {symbol} at row {start}, retry {retries}: {ex}")
                    time.sleep(2)
            else:
                print(f"Failed to insert prediction batch for {symbol} starting at row {start} after {MAX_RETRIES} retries.")

    print("Prediction process complete for all symbols.")

