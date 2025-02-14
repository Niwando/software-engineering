#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify

# Define the Flask app
app = Flask(__name__)

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
        # The final fully connected layer outputs a vector of length forecast_horizon
        self.fc = nn.Linear(hidden_size, forecast_horizon)
        
    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # Use the output of the last time step
        out = self.fc(out[:, -1, :])
        return out  # shape: (batch_size, forecast_horizon)

# ------------------------------
# Global settings and parameters
# ------------------------------
# The features must match the order used during training:
features_to_normalize = ['open', 'high', 'low', 'close', 'volume',
                           'sin_hour', 'cos_hour', 'sin_minute', 'cos_minute',
                           'day_of_week', 'day_of_month', 'month', 'day_of_year', 'year']

INPUT_SIZE = len(features_to_normalize)    # should be 14 in this case
HIDDEN_SIZE = 128
NUM_LAYERS = 1

# For predicting the next day's minute-wise forecast:
# For example, if you want to predict every minute during a standard US trading day (~6.5 hours = 390 minutes)
FORECAST_HORIZON = 390

# Input window size (e.g., use the last 60 minutes to predict the next day)
WINDOW_SIZE = 60

# Set the device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the scaler that was used during training.
# (Make sure the file 'scaler_minute.pkl' is in the same folder or adjust the path as needed.)
scaler = joblib.load("scaler_minute.pkl")

# ------------------------------
# Define the prediction endpoint
# ------------------------------
@app.route("/predict", methods=["GET"])
def predict():
    # Get the stock symbol from the GET parameters (default to "AAPL" if not provided)
    symbol = request.args.get("symbol", default="AAPL")
    
    # Construct the model filename (e.g., "trained_model_minute_multistep_AAPL.pth")
    model_file = f"trained_model_minute_multistep_{symbol}.pth"
    if not os.path.exists(model_file):
        return jsonify({"error": f"Model file {model_file} not found for symbol {symbol}."})
    
    # Initialize and load the model state
    model = MultiStepLSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, FORECAST_HORIZON, dropout=0.2)
    try:
        model.load_state_dict(torch.load(model_file, map_location=device))
    except Exception as e:
        return jsonify({"error": f"Error loading model: {str(e)}"})
    model.to(device)
    model.eval()
    
    # Load the processed data CSV (this file should have been generated previously)
    processed_file = "processed_stock_data.csv"
    if not os.path.exists(processed_file):
        return jsonify({"error": f"Processed data file {processed_file} not found."})
    
    try:
        df = pd.read_csv(processed_file, index_col="timestamp", parse_dates=True)
    except Exception as e:
        return jsonify({"error": f"Error loading processed data: {str(e)}"})
    
    # Filter the DataFrame for the given stock symbol
    df = df[df['symbol'] == symbol]
    if df.empty:
        return jsonify({"error": f"No data found for symbol {symbol}."})
    
    # Ensure the data is sorted by time and that we have enough rows to create an input sequence
    df = df.sort_index()
    if len(df) < WINDOW_SIZE:
        return jsonify({"error": f"Not enough data for prediction. Require at least {WINDOW_SIZE} rows."})
    
    # Select the last WINDOW_SIZE rows and extract the features in the proper order.
    input_data = df.iloc[-WINDOW_SIZE:][features_to_normalize].values
    # Convert to a tensor of shape (1, WINDOW_SIZE, INPUT_SIZE)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Run the model to get predictions (normalized close values)
    with torch.no_grad():
        pred_norm = model(input_tensor).cpu().numpy().flatten()  # shape: (FORECAST_HORIZON,)
    
    # To inverse-transform the normalized predictions we create a placeholder array for all features.
    # We then set the "close" column (index 3) to the predicted normalized values.
    pred_placeholder = np.zeros((len(pred_norm), len(features_to_normalize)))
    pred_placeholder[:, 3] = pred_norm  # assuming "close" is at index 3
    pred_prices = scaler.inverse_transform(pred_placeholder)[:, 3]
    
    # Compute the growth (percentage change) relative to the last observed close price.
    last_close = input_data[-1, 3]  # last close price from the input sequence
    growth = ((pred_prices - last_close) / last_close) * 100.0
    
    # Generate forecast times starting from one minute after the last timestamp in the processed data.
    last_timestamp = df.index[-1]
    forecast_times = pd.date_range(start=last_timestamp + pd.Timedelta(minutes=1),
                                   periods=FORECAST_HORIZON, freq='T')
    forecast_times_str = forecast_times.strftime("%Y-%m-%d %H:%M:%S").tolist()
    
    # Construct the response as a JSON-serializable dictionary.
    response = {
        "symbol": symbol,
        "last_close": float(last_close),
        "forecast": [
            {
                "time": t,
                "predicted_price": float(price),
                "growth_percent": float(g)
            }
            for t, price, g in zip(forecast_times_str, pred_prices, growth)
        ]
    }
    return jsonify(response)

# ------------------------------
# Run the Flask application
# ------------------------------
if __name__ == "__main__":
    # The app will listen on all interfaces (0.0.0.0) and port 5000.
    app.run(host="0.0.0.0", port=5000)
