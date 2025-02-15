import os
import time
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables from .env file
load_dotenv()

# Set up your Supabase URL and KEY via environment variables.
# Alternatively, you can directly assign the strings here.
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

if not url or not key:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set as environment variables.")

# Create the Supabase client.
supabase: Client = create_client(url, key)

# ------------------------------
# 1. Load CSV Data
# ------------------------------
csv_file = "../lstm/processed_stock_data.csv"

# Read the CSV file. Expecting columns: timestamp, open, high, low, close, volume, symbol
df = pd.read_csv(csv_file, parse_dates=["timestamp"])
print("CSV Data Loaded:")
print(df.head())

# ------------------------------
# 2. Prepare Data for Insertion
# ------------------------------
# Convert the DataFrame into a list of dictionaries (one per row)
# Ensure that the timestamp is converted to ISO format (which works well with Postgres timestamptz)
data = df.to_dict(orient="records")
for row in data:
    # Convert timestamp to ISO 8601 format if it's not already a string.
    if not isinstance(row["timestamp"], str):
        row["timestamp"] = row["timestamp"].isoformat()

# ------------------------------
# 3. Insert Data into Supabase Table
# ------------------------------
BATCH_SIZE = 50  # batch for better efficiency
num_rows = len(data)
print(f"Inserting {num_rows} rows into 'stock_data' table...")

MAX_RETRIES = 3

for start in range(0, num_rows, BATCH_SIZE):
    batch = data[start:start+BATCH_SIZE]
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = supabase.table("stock_data").insert(batch, upsert=True).execute()
            resp_dict = response.dict()
            if resp_dict.get("error"):
                print(f"Error inserting batch starting at row {start}: {resp_dict['error']}")
            else:
                print(f"Inserted rows {start} to {start + len(batch) - 1}")
            break  # exit the retry loop on success or error response
        except Exception as e:
            retries += 1
            print(f"Exception on batch starting at row {start}, retry {retries}: {e}")
            time.sleep(2)  # wait a moment before retrying
    else:
        print(f"Failed to insert batch starting at row {start} after {MAX_RETRIES} retries.")


print("Data insertion complete.")
