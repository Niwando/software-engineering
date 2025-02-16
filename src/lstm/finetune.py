#!/usr/bin/env python3
import os
import time
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from datetime import timedelta
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv
from supabase import create_client, Client

# ------------------------------
# Data Processing Helpers
# ------------------------------
def process_stock_data(df: pd.DataFrame, fill_method: str = 'ffill',
                       filter_market_hours: bool = True, timezone: str = 'US/Eastern') -> pd.DataFrame:
    
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone)
    else:
        df.index = df.index.tz_convert(timezone)
    
    processed_dfs = []
    for symbol, group in df.groupby('symbol'):
        group = group[~group.index.duplicated(keep='last')]
        full_range = pd.date_range(start=group.index.min(), end=group.index.max(), freq='1T', tz=timezone)
        group = group.reindex(full_range)
        group['symbol'] = symbol
        if fill_method == 'ffill':
            group = group.ffill()
        elif fill_method == 'bfill':
            group = group.bfill()
        elif fill_method == 'interpolate':
            group = group.interpolate()
        else:
            raise ValueError(f"Invalid fill_method: {fill_method}.")
        if filter_market_hours:
            group = group.between_time('09:30', '16:00')
        processed_dfs.append(group)
    processed_df = pd.concat(processed_dfs)
    return processed_df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize('US/Eastern')
    else:
        df.index = df.index.tz_convert('US/Eastern')
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_minute'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['cos_minute'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    df['year'] = df.index.year
    return df

def create_sequences_multi(data: np.ndarray, window_size: int, forecast_horizon: int):
    X, y = [], []
    for i in range(len(data) - window_size - forecast_horizon + 1):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size:i+window_size+forecast_horizon, 3])
    return np.array(X), np.array(y)

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ------------------------------
# Model Architecture
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
# Fine-tuning from DB Data
# ------------------------------
def fine_tune_all_stocks():
    load_dotenv()
    # Set up Supabase connection.
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set as environment variables.")
    supabase: Client = create_client(url, key)

    stocks = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "NFLX", "AVGO", "PYPL"]
    WINDOW_SIZE = 60         # past 60 minutes input
    forecast_horizon = 60    # next 60 minutes to predict
    EPOCHS = 10
    learning_rate = 1e-4
    features_list = ['open', 'high', 'low', 'close', 'volume',
                     'sin_hour', 'cos_hour', 'sin_minute', 'cos_minute',
                     'day_of_week', 'day_of_month', 'month', 'day_of_year', 'year']
    input_size = len(features_list)
    hidden_size = 128
    num_layers = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for stock in stocks:
        print(f"\n=== Fine-tuning stock: {stock} ===")
        # Query the DB for all raw stock_data rows for this symbol.
        try:
            res = supabase.table("stock_data")\
                .select("*")\
                .eq("symbol", stock)\
                .execute()
            records = res.dict().get("data", [])
            if not records:
                print(f"No data found in DB for {stock}. Skipping.")
                continue
            df = pd.DataFrame(records)
            # Convert timestamp to datetime and set as index.
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
        except Exception as e:
            print(f"Error retrieving data for {stock}: {e}")
            continue

        # Process data.
        try:
            processed_df = process_stock_data(df, fill_method='ffill', filter_market_hours=True)
            processed_df = add_time_features(processed_df)
        except Exception as e:
            print(f"Error processing data for {stock}: {e}")
            continue

        print(f"Data processed for {stock}. Time range: {processed_df.index.min()} to {processed_df.index.max()}")

        # For fine-tuning, select the last 60 days of data.
        end_time = processed_df.index.max()
        start_time = end_time - pd.Timedelta(days=60)
        finetune_df = processed_df.loc[start_time:end_time]
        print(f"Fine-tuning data range for {stock}: {finetune_df.index.min()} to {finetune_df.index.max()}")

        # Normalize using an existing scaler.
        scaler_file = f"src/lstm/scalers/scaler_minute_{stock}.pkl"
        if not os.path.exists(scaler_file):
            print(f"Scaler file not found for {stock}. Skipping fine-tuning.")
            continue
        scaler = joblib.load(scaler_file)
        finetune_norm_values = scaler.transform(finetune_df[features_list])
        finetune_norm_df = pd.DataFrame(finetune_norm_values, columns=features_list, index=finetune_df.index)
        finetune_norm_df['symbol'] = finetune_df['symbol']

        # Create training sequences.
        data_array = finetune_norm_df[features_list].values
        X_multi, y_multi = create_sequences_multi(data_array, WINDOW_SIZE, forecast_horizon)
        print(f"Sequences for {stock}: X: {X_multi.shape}, y: {y_multi.shape}")

        dataset = StockDataset(X_multi, y_multi)
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

        # Load pre-trained model.
        model_file = f"src/lstm/models/trained_model_{stock}.pth"
        if not os.path.exists(model_file):
            print(f"Pre-trained model file not found for {stock}. Skipping fine-tuning.")
            continue
        model = MultiStepLSTMModel(input_size, hidden_size, num_layers, forecast_horizon, dropout=0.2)
        model.to(device)
        model.load_state_dict(torch.load(model_file, map_location=device))
        print(f"Pre-trained model loaded for {stock}.")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_x.size(0)
            avg_loss = total_loss / len(dataset)
            print(f"{stock} - Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")

        # Save the fine-tuned model.
        torch.save(model.state_dict(), model_file)
        print(f"Fine-tuned model saved for {stock}.")

    print("Fine-tuning complete for all stocks.")

