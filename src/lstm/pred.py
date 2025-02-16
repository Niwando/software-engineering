#!/usr/bin/env python3
import os
import glob
import time
from datetime import timedelta, datetime, time as dtime
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib

# ------------------------------
# Define the model architecture
# ------------------------------
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

# ------------------------------
# Global settings and parameters
# ------------------------------
features_to_normalize = ['open', 'high', 'low', 'close', 'volume',
                           'sin_hour', 'cos_hour', 'sin_minute', 'cos_minute',
                           'day_of_week', 'day_of_month', 'month', 'day_of_year', 'year']
INPUT_SIZE = len(features_to_normalize)
HIDDEN_SIZE = 128
NUM_LAYERS = 1

# Prediction settings
MODEL_FORECAST_HORIZON = 60  
TARGET_HORIZON = 390  
WINDOW_SIZE = 60       # last 60 minutes of historical data as input
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Folders and file patterns
# ------------------------------
models_folder = "./models/"
processed_folder = "./processed_data/"
scalers_folder = "./scalers/"
predictions_folder = "./predictions/"
os.makedirs(predictions_folder, exist_ok=True)

# Find all model files matching pattern "trained_model_*.pth"
model_files = glob.glob(os.path.join(models_folder, "trained_model_*.pth"))

for model_path in model_files:
    # Extract the symbol from the filename
    base_model = os.path.basename(model_path)
    symbol = base_model.replace("trained_model_", "").replace(".pth", "")
    
    print(f"Processing symbol: {symbol}")

    # Construct paths for scaler and processed CSV
    scaler_path = os.path.join(scalers_folder, f"scaler_minute_{symbol}.pkl")
    processed_csv = os.path.join(processed_folder, f"processed_stock_{symbol}.csv")
    
    if not os.path.exists(scaler_path):
        print(f"Scaler file {scaler_path} not found for symbol {symbol}. Skipping.")
        continue
    if not os.path.exists(processed_csv):
        print(f"Processed CSV {processed_csv} not found for symbol {symbol}. Skipping.")
        continue

    # Load model
    model = MultiStepLSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, MODEL_FORECAST_HORIZON, dropout=0.2)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model for symbol {symbol}: {e}")
        continue
    model.to(device)
    model.eval()

    # Load scaler
    scaler = joblib.load(scaler_path)

    # Load processed CSV data
    try:
        df = pd.read_csv(processed_csv, index_col="timestamp", parse_dates=True)
    except Exception as e:
        print(f"Error loading processed data for symbol {symbol}: {e}")
        continue

    df = df[df['symbol'] == symbol]
    if df.empty:
        print(f"No data found in {processed_csv} for symbol {symbol}. Skipping.")
        continue
    df = df.sort_index()
    if len(df) < WINDOW_SIZE:
        print(f"Not enough data in {processed_csv} for symbol {symbol}. Require at least {WINDOW_SIZE} rows. Skipping.")
        continue

    # Use last WINDOW_SIZE rows as the starting chain
    initial_chain = df.iloc[-WINDOW_SIZE:][features_to_normalize].values
    current_chain = torch.tensor(initial_chain, dtype=torch.float32).to(device)

    # Set forecast start time: next trading day market open at 9:30
    last_timestamp = df.index[-1]
    next_day = (last_timestamp + pd.Timedelta(days=1)).date()
    market_open = pd.Timestamp.combine(next_day, dtime(9, 30))

    predictions = []  # List to store (timestamp, predicted_close)
    total_predicted = 0

    # Iterative (chain) prediction until TARGET_HORIZON minutes predicted
    while total_predicted < TARGET_HORIZON:
        with torch.no_grad():
            batch_pred_norm = model(current_chain.unsqueeze(0)).cpu().numpy().flatten()
        
        # Inverse transform the predicted normalized "close" values
        placeholder = np.zeros((len(batch_pred_norm), INPUT_SIZE))
        placeholder[:, 3] = batch_pred_norm  # "close" is at index 3
        batch_pred_original = scaler.inverse_transform(placeholder)[:, 3] 

        # Determine how many predictions to use from this batch
        remaining = TARGET_HORIZON - total_predicted
        use_length = min(len(batch_pred_norm), remaining)

        # Append predictions with forecast times
        for i in range(use_length):
            forecast_time = market_open + pd.Timedelta(minutes=total_predicted + i)
            predictions.append((forecast_time, batch_pred_original[i]))
        total_predicted += use_length

        # Build new rows for chaining: for each minute, construct a full feature row
        new_rows = []
        for i in range(len(batch_pred_norm)):
            ft = market_open + pd.Timedelta(minutes=total_predicted - len(batch_pred_norm) + i)
            # Use the predicted close for price fields and the last known volume
            last_chain_row = current_chain[-1].cpu().numpy().reshape(1, -1)
            last_volume = scaler.inverse_transform(last_chain_row)[0, 4]
            row_dict = {
                "open": batch_pred_original[i],
                "high": batch_pred_original[i],
                "low": batch_pred_original[i],
                "close": batch_pred_original[i],
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
            new_rows.append(row_dict)
        new_rows_df = pd.DataFrame(new_rows)
        new_rows_array = scaler.transform(new_rows_df[features_to_normalize].values)
        # Update chain: take last WINDOW_SIZE rows
        updated_chain = np.concatenate([current_chain.cpu().numpy(), new_rows_array], axis=0)
        current_chain = torch.tensor(updated_chain[-WINDOW_SIZE:], dtype=torch.float32).to(device)

    # Create predictions DataFrame including the symbol column
    pred_df = pd.DataFrame(predictions, columns=["time", "predicted_close"])
    pred_df["symbol"] = symbol  # add symbol column

    # Save to CSV in predictions folder
    output_file = os.path.join(predictions_folder, f"predictions_{symbol}.csv")
    pred_df.to_csv(output_file, index=False)
    print(f"Predictions for {symbol} saved to {output_file}")

print("Prediction loop complete.")
