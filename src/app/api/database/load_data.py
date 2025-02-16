import os
import time
import glob
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables from .env file
load_dotenv()

# Set up your Supabase URL and KEY via environment variables.
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
if not url or not key:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set as environment variables.")

# Create the Supabase client.
supabase: Client = create_client(url, key)

# ------------------------------
# 1. Load CSV Data
# ------------------------------
# Get a list of all CSV files in the folder (adjust relative path as needed)
csv_files = glob.glob("../../../lstm/processed_data/*.csv")

# Read each CSV file into a DataFrame and store in a list
dfs = [pd.read_csv(file, parse_dates=["timestamp"]) for file in csv_files]

# Concatenate all DataFrames into one
df = pd.concat(dfs, ignore_index=True)
print("CSV Data Loaded:")
print(df.head())

# Ensure the 'timestamp' column is in datetime format
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ------------------------------
# 2. Check for New Data
# ------------------------------
# Query the database for the maximum timestamp already stored in the "stock_data" table.
existing_response = supabase.table("stock_data")\
    .select("timestamp")\
    .order("timestamp", desc=True)\
    .limit(1)\
    .execute()
existing_data = existing_response.dict().get("data", [])

if existing_data:
    max_timestamp = pd.to_datetime(existing_data[0]["timestamp"])
    print(f"Max timestamp in database: {max_timestamp}")
    # Filter the CSV data: only rows with timestamp > max_timestamp are new.
    df_new = df[df["timestamp"] > max_timestamp]
else:
    print("No existing data found in database. Inserting all rows.")
    df_new = df

num_new_rows = len(df_new)
print(f"Number of new rows to insert: {num_new_rows}")

if num_new_rows == 0:
    print("No new data to insert.")
    exit(0)

# ------------------------------
# 3. Prepare Data for Insertion
# ------------------------------
# Convert the DataFrame into a list of dictionaries.
data = df_new.to_dict(orient="records")
# Convert timestamp to ISO 8601 format for each record
for row in data:
    if not isinstance(row["timestamp"], str):
        row["timestamp"] = row["timestamp"].isoformat()

# ------------------------------
# 4. Insert Data into Supabase Table
# ------------------------------
BATCH_SIZE = 50  # Adjust batch size as needed
MAX_RETRIES = 3
num_rows = len(data)
print(f"Inserting {num_rows} new rows into 'stock_data' table...")

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
            break  # Exit retry loop on success or handled error
        except Exception as e:
            retries += 1
            print(f"Exception on batch starting at row {start}, retry {retries}: {e}")
            time.sleep(2)  # Delay before retrying
    else:
        print(f"Failed to insert batch starting at row {start} after {MAX_RETRIES} retries.")

print("Data insertion complete.")