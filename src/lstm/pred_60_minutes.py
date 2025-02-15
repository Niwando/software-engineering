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

# Adjust forecast horizon to 60 to match the saved models
FORECAST_HORIZON = 60  
WINDOW_SIZE = 60       # use the last 60 minutes as input
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Folders and file patterns (executed from the "lstm" folder)
# ------------------------------
models_folder = "./models/"
processed_folder = "./processed_data/"
scalers_folder = "./scalers/"
predictions_folder = "./predictions/"
os.makedirs(predictions_folder, exist_ok=True)

# Find all model files in models_folder matching pattern "trained_model_*.pth"
model_files = glob.glob(os.path.join(models_folder, "trained_model_*.pth"))

for model_path in model_files:
    # Extract the symbol from the filename.
    # Expected filename format: trained_model_{symbol}.pth
    base_model = os.path.basename(model_path)
    symbol = base_model.replace("trained_model_", "").replace(".pth", "")
    
    print(f"Processing symbol: {symbol}")

    # Construct paths for scaler and processed CSV
    scaler_path = os.path.join(scalers_folder, f"scaler_minute_{symbol}.pkl")
    processed_csv = os.path.join(processed_folder, f"processed_stock_{symbol}.csv")
    
    # Check existence of scaler and CSV files
    if not os.path.exists(scaler_path):
        print(f"Scaler file {scaler_path} not found for symbol {symbol}. Skipping.")
        continue
    if not os.path.exists(processed_csv):
        print(f"Processed CSV {processed_csv} not found for symbol {symbol}. Skipping.")
        continue

    # Load model
    model = MultiStepLSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, FORECAST_HORIZON, dropout=0.2)
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

    # Filter data to the current symbol (if needed) and ensure it's sorted
    df = df[df['symbol'] == symbol]
    if df.empty:
        print(f"No data found in {processed_csv} for symbol {symbol}. Skipping.")
        continue
    df = df.sort_index()
    if len(df) < WINDOW_SIZE:
        print(f"Not enough data in {processed_csv} for symbol {symbol}. Require at least {WINDOW_SIZE} rows. Skipping.")
        continue

    # Prepare input tensor from the last WINDOW_SIZE rows
    input_data = df.iloc[-WINDOW_SIZE:][features_to_normalize].values
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)

    # Predict normalized close values
    with torch.no_grad():
        pred_norm = model(input_tensor).cpu().numpy().flatten()

    # Inverse transform for "close" price (assumed to be at index 3)
    pred_placeholder = np.zeros((len(pred_norm), len(features_to_normalize)))
    pred_placeholder[:, 3] = pred_norm
    pred_prices = scaler.inverse_transform(pred_placeholder)[:, 3]

    # Generate forecast times for the next day using a 60-minute horizon.
    # Here we assume market open time of 9:30 on the next day.
    last_timestamp = df.index[-1]
    next_day = (last_timestamp + timedelta(days=1)).date()
    market_open = pd.Timestamp.combine(next_day, dtime(9, 30))
    forecast_times = pd.date_range(start=market_open, periods=FORECAST_HORIZON, freq='T')

    # Create a DataFrame for the predictions
    pred_df = pd.DataFrame({
        "time": forecast_times,
        "predicted_close": pred_prices
    })

    # Save predictions to CSV in the predictions folder
    output_file = os.path.join(predictions_folder, f"predictions_{symbol}.csv")
    pred_df.to_csv(output_file, index=False)
    print(f"Predictions for {symbol} saved to {output_file}")

print("Prediction loop complete.")
