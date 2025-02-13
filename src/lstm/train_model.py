# necessary imports
import glob
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# -------------------------------
# 1. Data Loading, Processing, and Saving
# -------------------------------

def load_stock_data(file_pattern: str) -> pd.DataFrame:
    all_dfs = []
    files = glob.glob(file_pattern)

    for file in tqdm(files, desc="Loading JSON files"):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                metadata = data["Meta Data"]
                symbol = metadata["2. Symbol"]
                interval = metadata["4. Interval"]
                if interval != "1min":
                    print(f"Warning: {file} has interval {interval} (expected '1min'). Skipping.")
                    continue
                ts_key = f"Time Series ({interval})"
                ts_data = data.get(ts_key)
                if not ts_data:
                    print(f"Warning: {ts_key} not found in {file}. Skipping.")
                    continue
                df = pd.DataFrame(ts_data).T
                df = df.apply(pd.to_numeric)
                df["symbol"] = symbol
                all_dfs.append(df)
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
            continue

    if not all_dfs:
        raise ValueError("No valid data found in any files.")

    combined_df = pd.concat(all_dfs)
    combined_df.index = pd.to_datetime(combined_df.index)
    combined_df = combined_df.sort_index()
    combined_df.columns = [col.split(". ")[-1] for col in combined_df.columns]
    return combined_df

def process_stock_data(df: pd.DataFrame, fill_method: str = 'ffill',
                       filter_market_hours: bool = True, timezone: str = 'US/Eastern') -> pd.DataFrame:
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone)
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
            raise ValueError(f"Invalid fill_method: {fill_method}. Use 'ffill', 'bfill', or 'interpolate'.")
        if filter_market_hours:
            group = group.between_time('09:30', '16:00')
        processed_dfs.append(group)
    processed_df = pd.concat(processed_dfs)
    return processed_df

def save_data(df: pd.DataFrame, output_path: str) -> None:
    df.to_csv(output_path, index_label='timestamp')
    print(f"Data saved to {output_path}")

def verify_data(df: pd.DataFrame) -> None:
    print("=== Data Verification ===")
    print(f"Time range: {df.index.min()} to {df.index.max()}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Unique symbols: {df['symbol'].unique()}")
    print(f"Total rows: {len(df)}")

# Step 1: Load raw data
raw_df = load_stock_data("../alphavantage/data/*.json")

# Step 2: Process data
processed_df = process_stock_data(raw_df, fill_method='ffill', filter_market_hours=True)

# Step 3: Verify and (optionally) save the processed data
verify_data(processed_df)
save_data(processed_df, "processed_stock_data.csv")

# -------------------------------
# 2. Normalization
# -------------------------------

class Normalizer:
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, data):
        # data is expected to be a NumPy array (or convertible)
        if isinstance(data, pd.DataFrame):
            data = data.values
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.sd = np.std(data, axis=0, keepdims=True)
        normalized_data = (data - self.mu) / self.sd
        return normalized_data

    def inverse_transform(self, normalized_data):
        return (normalized_data * self.sd) + self.mu

features_to_normalize = ['open', 'high', 'low', 'close', 'volume']
scalers = {}         # To store per-symbol scalers
normalized_dfs = []  # To collect normalized data for each symbol

for symbol, group in processed_df.groupby('symbol'):
    scaler = Normalizer()
    normalized_data = scaler.fit_transform(group[features_to_normalize])
    scalers[symbol] = scaler
    normalized_df_symbol = pd.DataFrame(normalized_data, columns=features_to_normalize, index=group.index)
    normalized_df_symbol['symbol'] = symbol
    normalized_dfs.append(normalized_df_symbol)

normalized_df = pd.concat(normalized_dfs)
print("Normalized DataFrame:")
print(normalized_df.head())

# -------------------------------
# 3. Preparing Sequences (Including Symbol Labels)
# -------------------------------

def prepare_data_x(x, window_size):
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_row, window_size),
        strides=(x.strides[0], x.strides[0])
    )
    # Use the entire sequence except for the final time step as input,
    # and reserve the final time step separately as unseen data if needed.
    return output[:-1], output[-1:]

def prepare_data_y(x, window_size):
    return x[window_size:]

WINDOW_SIZE = 60    # Number of time steps per sequence
TRAIN_SPLIT_SIZE = 0.8
EPOCHS = 50          # For demonstration; adjust as needed
BATCH_SIZE = 32

all_data_x, all_data_y, all_symbol_x = [], [], []
all_data_x_unseen, all_symbol_x_unseen = [], []

# Create a mapping from symbol string to a unique integer for embedding.
symbols = sorted(normalized_df['symbol'].unique())
symbol_to_int = {s: i for i, s in enumerate(symbols)}

for symbol, group in normalized_df.groupby('symbol'):
    # Use the normalized close price for sequence creation.
    normalized_data_close_price = group['close'].values
    data_x, data_x_unseen = prepare_data_x(normalized_data_close_price, window_size=WINDOW_SIZE)
    data_y = prepare_data_y(normalized_data_close_price, window_size=WINDOW_SIZE)
    symbol_code = symbol_to_int[symbol]
    symbol_array = np.full((data_x.shape[0],), symbol_code, dtype=np.int32)
    symbol_array_unseen = np.array([symbol_code], dtype=np.int32)
    all_data_x.append(data_x)
    all_data_y.append(data_y)
    all_symbol_x.append(symbol_array)
    all_data_x_unseen.append(data_x_unseen)
    all_symbol_x_unseen.append(symbol_array_unseen)

data_x = np.concatenate(all_data_x, axis=0)
data_y = np.concatenate(all_data_y, axis=0)
symbol_x = np.concatenate(all_symbol_x, axis=0)
data_x_unseen = np.concatenate(all_data_x_unseen, axis=0)
symbol_x_unseen = np.concatenate(all_symbol_x_unseen, axis=0)

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

# Reshape sequences to have an extra feature dimension (for close price)
data_x_train = data_x_train.reshape((data_x_train.shape[0], data_x_train.shape[1], 1))
data_x_val = data_x_val.reshape((data_x_val.shape[0], data_x_val.shape[1], 1))
data_x_unseen = data_x_unseen.reshape((data_x_unseen.shape[0], data_x_unseen.shape[1], 1))

# For embedding, symbol inputs are 2D: (samples, 1)
symbol_x_train = symbol_x_train.reshape(-1, 1)
symbol_x_val = symbol_x_val.reshape(-1, 1)
symbol_x_unseen = symbol_x_unseen.reshape(-1, 1)

print(f"Reshaped training data: {data_x_train.shape}, symbols: {symbol_x_train.shape}")
print(f"Reshaped validation data: {data_x_val.shape}, symbols: {symbol_x_val.shape}")
print(f"Reshaped unseen data: {data_x_unseen.shape}, symbols: {symbol_x_unseen.shape}")

# -------------------------------
# 4. PyTorch Multi-Input LSTM Model with Symbol Embedding
# -------------------------------

class StockDataset(Dataset):
    def __init__(self, sequences, symbols, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)  # (samples, WINDOW_SIZE, 1)
        self.symbols = torch.tensor(symbols, dtype=torch.long)         # (samples, 1)
        self.targets = torch.tensor(targets, dtype=torch.float32)        # (samples,)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.symbols[idx], self.targets[idx]

class MultiInputLSTMModelPT(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2,
                 output_size=1, dropout=0.2, num_symbols=10, embedding_dim=4):
        super(MultiInputLSTMModelPT, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        # Dense layer applied to each time step
        self.linear_1 = nn.Linear(input_size, hidden_layer_size)

        # Create LSTM layers
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=hidden_layer_size,
                    hidden_size=hidden_layer_size,
                    num_layers=1,
                    batch_first=True
                )
            )
        self.dropout = nn.Dropout(dropout)
        # Embedding layer for the symbol (categorical feature)
        self.symbol_embedding = nn.Embedding(num_symbols, embedding_dim)
        # Combined dense layers
        self.combined_dense = nn.Linear(hidden_layer_size + embedding_dim, 16)
        self.relu = nn.ReLU()
        self.out_layer = nn.Linear(16, output_size)

    def forward(self, seq_input, symbol_input):
        # seq_input: (batch, seq_len, 1)
        x = self.linear_1(seq_input)  # (batch, seq_len, hidden_layer_size)

        # Process through LSTM layers sequentially
        for i, lstm in enumerate(self.lstm_layers):
            out, (h_n, c_n) = lstm(x)
            if i < len(self.lstm_layers) - 1:
                x = out  # pass full sequence to next layer
            else:
                x = h_n[-1]  # use last hidden state of the final layer

        x = self.dropout(x)

        # Process symbol input: shape (batch, 1) -> squeeze to (batch,)
        symbol_input = symbol_input.squeeze(1)
        symbol_emb = self.symbol_embedding(symbol_input)  # (batch, embedding_dim)

        # Concatenate LSTM output and symbol embedding
        combined = torch.cat([x, symbol_emb], dim=1)
        combined = self.relu(self.combined_dense(combined))
        output = self.out_layer(combined)
        return output.squeeze(1)  # (batch,)

# Define model parameters
input_size = 1
hidden_layer_size = 32
num_layers = 2
output_size = 1
dropout = 0.2
num_symbols = len(symbols)
embedding_dim = 4

model_pt = MultiInputLSTMModelPT(
    input_size=input_size,
    hidden_layer_size=hidden_layer_size,
    num_layers=num_layers,
    output_size=output_size,
    dropout=dropout,
    num_symbols=num_symbols,
    embedding_dim=embedding_dim
)

# Set up optimizer and loss
optimizer = optim.Adam(model_pt.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Create Datasets and DataLoaders
train_dataset = StockDataset(data_x_train, symbol_x_train, data_y_train)
val_dataset   = StockDataset(data_x_val, symbol_x_val, data_y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------
# 5. Training the PyTorch Model
# -------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_pt.to(device)

def run_epoch(dataloader, model, criterion, optimizer=None, is_training=False):
    epoch_loss = 0.0
    if is_training:
        model.train()
    else:
        model.eval()

    for seq_batch, symbol_batch, target_batch in dataloader:
        seq_batch = seq_batch.to(device)         # (batch, WINDOW_SIZE, 1)
        symbol_batch = symbol_batch.to(device)     # (batch, 1)
        target_batch = target_batch.to(device)     # (batch,)

        if is_training:
            optimizer.zero_grad()

        outputs = model(seq_batch, symbol_batch)
        loss = criterion(outputs, target_batch)

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item() * seq_batch.size(0)

    return epoch_loss / len(dataloader.dataset)

for epoch in range(EPOCHS):
    train_loss = run_epoch(train_loader, model_pt, criterion, optimizer, is_training=True)
    val_loss = run_epoch(val_loader, model_pt, criterion, optimizer, is_training=False)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

# Save the trained model
torch.save(model_pt.state_dict(), "trained_model_pt.pth")
print("Model saved.")

# -------------------------------
# 6. Predicting on Unseen Data (Per Stock Symbol)
# -------------------------------
#
# For each symbol, we take the most recent WINDOW_SIZE close prices (from the normalized data),
# run the model in inference mode, and then inverse transform the predicted value.
#
# NOTE: Since the Normalizer was fit on all five features and 'close' is the 4th column (index 3),
# we perform the inverse transform manually using that column’s parameters.

model_pt.eval()  # set model to evaluation mode

predictions = {}

for symbol in symbols:
    # Get normalized data for this symbol (ensure sorted by time)
    group = normalized_df[normalized_df['symbol'] == symbol].sort_index()
    
    if len(group) < WINDOW_SIZE:
        print(f"Not enough data for symbol {symbol} to create a sequence.")
        continue

    # Prepare the input sequence: last WINDOW_SIZE values of the normalized close price
    seq = group['close'].values[-WINDOW_SIZE:]
    seq = seq.reshape(1, WINDOW_SIZE, 1)
    seq_tensor = torch.tensor(seq, dtype=torch.float32).to(device)

    # Prepare the symbol input for the embedding (shape: [1, 1])
    symbol_code = np.array([[symbol_to_int[symbol]]], dtype=np.int32)
    symbol_tensor = torch.tensor(symbol_code, dtype=torch.long).to(device)

    # Make prediction
    with torch.no_grad():
        prediction_norm = model_pt(seq_tensor, symbol_tensor)  # (batch,)
    # prediction_norm is a 1D tensor of size 1; convert to NumPy and reshape to (1, 1)
    prediction_norm = prediction_norm.cpu().numpy().reshape(1, -1)
    
    # Inverse transform the normalized prediction.
    # Since the scaler was fit on [open, high, low, close, volume],
    # and 'close' is the fourth feature (index 3), we use its mean and std.
    scaler = scalers[symbol]
    pred_original = prediction_norm[0, 0] * scaler.sd[0, 3] + scaler.mu[0, 3]
    
    predictions[symbol] = pred_original
    print(f"Predicted original close price for {symbol}: {pred_original}")

