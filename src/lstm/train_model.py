# necessary imports
import glob
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, LSTM, Dropout, Embedding, Concatenate, Activation)
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt


############################################
# 1. Data Loading, Processing, and Saving
############################################

def load_stock_data(file_pattern: str) -> pd.DataFrame:
    """
    Loads JSON files with 1-minute stock data, extracts metadata, and ensures the correct time series key.

    Args:
        file_pattern: Glob pattern for JSON files (e.g., "data/*.json").

    Returns:
        Combined DataFrame with columns: [open, high, low, close, volume, symbol].
    """
    all_dfs = []
    files = glob.glob(file_pattern)

    for file in tqdm(files, desc="Loading JSON files"):
        try:
            with open(file, 'r') as f:
                data = json.load(f)

                # Extract metadata
                metadata = data["Meta Data"]
                symbol = metadata["2. Symbol"]

                # Verify the interval is "1min"
                interval = metadata["4. Interval"]
                if interval != "1min":
                    print(f"Warning: {file} has interval {interval} (expected '1min'). Skipping.")
                    continue

                # Construct the time series key
                ts_key = f"Time Series ({interval})"
                ts_data = data.get(ts_key)

                if not ts_data:
                    print(f"Warning: {ts_key} not found in {file}. Skipping.")
                    continue

                # Convert time series to DataFrame
                df = pd.DataFrame(ts_data).T
                df = df.apply(pd.to_numeric)
                df["symbol"] = symbol
                all_dfs.append(df)

        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
            continue

    if not all_dfs:
        raise ValueError("No valid data found in any files.")

    # Combine DataFrames
    combined_df = pd.concat(all_dfs)
    combined_df.index = pd.to_datetime(combined_df.index)
    combined_df = combined_df.sort_index()

    # Rename columns (e.g., "1. open" → "open")
    combined_df.columns = [col.split(". ")[-1] for col in combined_df.columns]

    return combined_df

def process_stock_data(
    df: pd.DataFrame,
    fill_method: str = 'ffill',
    filter_market_hours: bool = True,
    timezone: str = 'US/Eastern'
) -> pd.DataFrame:
    """
    Processes stock data:
    - Removes duplicate timestamps.
    - Fills missing intervals.
    - Optionally filters to market hours.

    Args:
        df: Raw DataFrame from `load_stock_data`.
        fill_method: Method to fill missing data ('ffill', 'bfill', 'interpolate').
        filter_market_hours: If True, retain only 9:30 AM - 4:00 PM ET timestamps.
        timezone: Time zone to localize timestamps.

    Returns:
        Processed DataFrame with continuous timestamps.
    """
    # Localize timezone if not already set
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone)

    processed_dfs = []

    # Process each symbol separately
    for symbol, group in df.groupby('symbol'):
        # Remove duplicates (keep last occurrence)
        group = group[~group.index.duplicated(keep='last')]

        # Create full time range for the symbol's period
        full_range = pd.date_range(
            start=group.index.min(),
            end=group.index.max(),
            freq='1T',
            tz=timezone
        )

        # Reindex to fill missing times
        group = group.reindex(full_range)

        # Fill missing values
        group['symbol'] = symbol  # Ensure symbol is preserved
        if fill_method == 'ffill':
            group = group.ffill()
        elif fill_method == 'bfill':
            group = group.bfill()
        elif fill_method == 'interpolate':
            group = group.interpolate()
        else:
            raise ValueError(f"Invalid fill_method: {fill_method}. Use 'ffill', 'bfill', or 'interpolate'.")

        # Filter market hours (9:30 AM to 4:00 PM ET)
        if filter_market_hours:
            group = group.between_time('09:30', '16:00')

        processed_dfs.append(group)

    # Combine all symbols
    processed_df = pd.concat(processed_dfs)
    return processed_df

def save_data(df: pd.DataFrame, output_path: str) -> None:
    """Saves DataFrame to a CSV file."""
    df.to_csv(output_path, index_label='timestamp')
    print(f"Data saved to {output_path}")

def verify_data(df: pd.DataFrame) -> None:
    """Prints basic checks for data integrity."""
    print("=== Data Verification ===")
    print(f"Time range: {df.index.min()} to {df.index.max()}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Unique symbols: {df['symbol'].unique()}")
    print(f"Total rows: {len(df)}")

# Step 1: Load raw data
raw_df = load_stock_data("../alphavantage/data/*.json")
# raw_df = load_stock_data("data/*.json")

# Step 2: Process data
processed_df = process_stock_data(
    raw_df,
    fill_method='ffill',  # Forward-fill missing values
    filter_market_hours=True
)

# Step 3: Verify
verify_data(processed_df)

# Step 4: Save (optional)
save_data(processed_df, "processed_stock_data.csv")


############################################
# 2. Normalization
############################################

class Normalizer:
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, data):
        """
        Normalizes the input data.
        Args:
            data: 2D numpy array or DataFrame (samples, features).
        Returns:
            Normalized data (numpy array).
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.sd = np.std(data, axis=0, keepdims=True)
        normalized_data = (data - self.mu) / self.sd
        return normalized_data

    def inverse_transform(self, normalized_data):
        """Converts normalized data back to original scale."""
        return (normalized_data * self.sd) + self.mu

# Define features to normalize
features_to_normalize = ['open', 'high', 'low', 'close', 'volume']

# Initialize a dictionary to store scalers for each symbol
scalers = {}
normalized_dfs = []

# Normalize each symbol's data
for symbol, group in processed_df.groupby('symbol'):
    scaler = Normalizer()
    normalized_data = scaler.fit_transform(group[features_to_normalize])
    scalers[symbol] = scaler
    normalized_df = pd.DataFrame(
        normalized_data,
        columns=features_to_normalize,
        index=group.index
    )
    # Preserve the symbol information
    normalized_df['symbol'] = symbol
    normalized_dfs.append(normalized_df)

# Combine normalized DataFrames
normalized_df = pd.concat(normalized_dfs)

print("Normalized DataFrame:")
print(normalized_df.head())


############################################
# 3. Preparing Sequences (Including Symbol Labels)
############################################

def prepare_data_x(x, window_size):
    """
    Create input sequences (X) using sliding windows.
    Args:
        x: 1D numpy array (normalized close prices).
        window_size: Size of the sliding window.
    Returns:
        data_x: Input sequences for training/validation.
        data_x_unseen: Final window for prediction.
    """
    n_row = x.shape[0] - window_size + 1
    # Use stride_tricks for efficient windowing
    output = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_row, window_size),
        strides=(x.strides[0], x.strides[0])
    )
    return output[:-1], output[-1:]  # All but last for training, last for prediction

def prepare_data_y(x, window_size):
    """
    Create target values (y) for each window.
    Args:
        x: 1D numpy array (normalized close prices).
        window_size: Size of the sliding window.
    Returns:
        output: Target values (the next time step after each window).
    """
    return x[window_size:]  # Next time step for each window

# Define parameters
WINDOW_SIZE = 60    # Number of time steps per sequence
TRAIN_SPLIT_SIZE = 0.8
EPOCHS = 50
BATCH_SIZE = 32

# Create lists to store sequences for all symbols.
# We now also prepare arrays to hold the corresponding symbol (as an integer) for each sequence.
all_data_x, all_data_y, all_symbol_x = [], [], []
all_data_x_unseen, all_symbol_x_unseen = [], []

# Create a mapping from symbol string to a unique integer (for the embedding).
symbols = sorted(normalized_df['symbol'].unique())
symbol_to_int = {s: i for i, s in enumerate(symbols)}

# For each symbol, prepare sliding window sequences.
for symbol, group in normalized_df.groupby('symbol'):
    # Extract the normalized close price values
    normalized_data_close_price = group['close'].values
    data_x, data_x_unseen = prepare_data_x(normalized_data_close_price, window_size=WINDOW_SIZE)
    data_y = prepare_data_y(normalized_data_close_price, window_size=WINDOW_SIZE)

    # Create an array for the symbol identifier (using our mapping)
    symbol_code = symbol_to_int[symbol]
    symbol_array = np.full((data_x.shape[0],), symbol_code, dtype=np.int32)
    symbol_array_unseen = np.array([symbol_code], dtype=np.int32)  # one per symbol for unseen data

    all_data_x.append(data_x)
    all_data_y.append(data_y)
    all_symbol_x.append(symbol_array)
    all_data_x_unseen.append(data_x_unseen)
    all_symbol_x_unseen.append(symbol_array_unseen)

# Combine sequences from all symbols
data_x = np.concatenate(all_data_x, axis=0)           # shape: (samples, WINDOW_SIZE)
data_y = np.concatenate(all_data_y, axis=0)           # shape: (samples,)
symbol_x = np.concatenate(all_symbol_x, axis=0)       # shape: (samples,)
data_x_unseen = np.concatenate(all_data_x_unseen, axis=0)  # shape: (num_symbols, WINDOW_SIZE)
symbol_x_unseen = np.concatenate(all_symbol_x_unseen, axis=0)  # shape: (num_symbols,)

print(f"Input shape: {data_x.shape}")
print(f"Target shape: {data_y.shape}")
print(f"Symbol shape: {symbol_x.shape}")
print(f"Unseen data shape: {data_x_unseen.shape}")

# Split the data into training and validation sets
split_index = int(data_y.shape[0] * TRAIN_SPLIT_SIZE)
data_x_train = data_x[:split_index]
data_x_val = data_x[split_index:]
data_y_train = data_y[:split_index]
data_y_val = data_y[split_index:]
symbol_x_train = symbol_x[:split_index]
symbol_x_val = symbol_x[split_index:]

# Reshape the sequence data for LSTM (adding feature dimension)
data_x_train = data_x_train.reshape((data_x_train.shape[0], data_x_train.shape[1], 1))
data_x_val = data_x_val.reshape((data_x_val.shape[0], data_x_val.shape[1], 1))
data_x_unseen = data_x_unseen.reshape((data_x_unseen.shape[0], data_x_unseen.shape[1], 1))

# The symbol inputs must be 2D (batch_size, 1) for the Embedding layer.
symbol_x_train = symbol_x_train.reshape(-1, 1)
symbol_x_val = symbol_x_val.reshape(-1, 1)
symbol_x_unseen = symbol_x_unseen.reshape(-1, 1)

print(f"Reshaped training data: {data_x_train.shape}, symbols: {symbol_x_train.shape}")
print(f"Reshaped validation data: {data_x_val.shape}, symbols: {symbol_x_val.shape}")
print(f"Reshaped unseen data: {data_x_unseen.shape}, symbols: {symbol_x_unseen.shape}")


############################################
# 4. Multi-Input LSTM Model with Symbol Embedding
############################################

class MultiInputLSTMModel(Model):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2,
                 output_size=1, dropout=0.2, num_symbols=10, embedding_dim=4):
        """
        Args:
            input_size: Number of features in the sequence (here 1 for close price).
            hidden_layer_size: Number of units in Dense and LSTM layers.
            num_layers: Number of stacked LSTM layers.
            output_size: Dimension of model output (1 for predicting a single value).
            dropout: Dropout rate.
            num_symbols: Total number of distinct symbols (for the Embedding layer).
            embedding_dim: Dimension of the symbol embedding.
        """
        super(MultiInputLSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        # Layers to process the sequence input
        self.linear_1 = Dense(hidden_layer_size, activation='relu')
        self.lstm_layers = []
        for i in range(num_layers):
            # Return sequences for all but the last LSTM layer
            return_seq = (i < num_layers - 1)
            self.lstm_layers.append(LSTM(hidden_layer_size, return_sequences=return_seq))
        self.dropout = Dropout(dropout)

        # Embedding layer for the stock symbol (categorical feature)
        self.symbol_embedding = Embedding(input_dim=num_symbols, output_dim=embedding_dim, input_length=1)

        # After concatenating LSTM output and symbol embedding, add a dense layer before final output.
        self.combined_dense = Dense(16, activation='relu')
        self.out_layer = Dense(output_size, activation='linear')

    def call(self, inputs):
        # Unpack the two inputs: sequence and symbol
        seq_input, symbol_input = inputs

        # Process the sequence input
        x = self.linear_1(seq_input)
        for lstm in self.lstm_layers:
            x = lstm(x)
        x = self.dropout(x)

        # Process the symbol input via the embedding
        symbol_emb = self.symbol_embedding(symbol_input)  # shape: (batch, 1, embedding_dim)
        symbol_emb = tf.squeeze(symbol_emb, axis=1)         # shape: (batch, embedding_dim)

        # Concatenate the LSTM features with the symbol embedding
        combined = Concatenate()([x, symbol_emb])
        combined = self.combined_dense(combined)
        output = self.out_layer(combined)
        return output

# Define model parameters
input_size = 1                 # Only the close price is used in the sequence
hidden_layer_size = 32
num_layers = 2
output_size = 1
dropout = 0.2
num_symbols = len(symbols)     # Based on the mapping we created
embedding_dim = 4

# Initialize the model
model = MultiInputLSTMModel(
    input_size=input_size,
    hidden_layer_size=hidden_layer_size,
    num_layers=num_layers,
    output_size=output_size,
    dropout=dropout,
    num_symbols=num_symbols,
    embedding_dim=embedding_dim
)

model.compile(optimizer='adam', loss='mse')

############################################
# 5. Training the Model
############################################

history = model.fit(
    [data_x_train, symbol_x_train],
    data_y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=([data_x_val, symbol_x_val], data_y_val)
)

# (Optional) Prediction on unseen sequences for each symbol
predictions = model.predict([data_x_unseen, symbol_x_unseen])
print("Predictions on unseen data (one per symbol):")
for sym, pred in zip(symbols, predictions):
    print(f"{sym}: {pred[0]}")


model.save("trained_model.h5")

# save weights
model.save_weights("trained_model_weights.h5")