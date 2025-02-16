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
# Get a list of all CSV files in the predictions folder
csv_files = glob.glob("../../../lstm/predictions/*.csv")

# Read each CSV file into a DataFrame and store in a list.
# We assume the CSV files have a "time" column.
dfs = [pd.read_csv(file, parse_dates=["time"]) for file in csv_files]

# Concatenate all DataFrames into one
df = pd.concat(dfs, ignore_index=True)
print("CSV Data Loaded:")
print(df.head())

# ------------------------------
# 2. Check for New Data
# ------------------------------
# Query the predicted_values table for the maximum time already stored.
existing_response = supabase.table("predicted_values")\
    .select("time")\
    .order("time", desc=True)\
    .limit(1)\
    .execute()
existing_data = existing_response.dict().get("data", [])

if existing_data:
    max_time = pd.to_datetime(existing_data[0]["time"])
    print(f"Max time in database: {max_time}")
    # Only keep rows where "time" > max_time.
    df_new = df[df["time"] > max_time]
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
for row in data:
    # Convert "time" column to ISO 8601 format if necessary.
    if not isinstance(row["time"], str):
        row["time"] = row["time"].isoformat()

# ------------------------------
# 4. Insert Data into Supabase Table
# ------------------------------
BATCH_SIZE = 50  # adjust as needed
MAX_RETRIES = 3
num_rows = len(data)
print(f"Inserting {num_rows} rows into 'predicted_values' table...")

for start in range(0, num_rows, BATCH_SIZE):
    batch = data[start:start+BATCH_SIZE]
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = supabase.table("predicted_values").insert(batch, upsert=True).execute()
            resp_dict = response.dict()
            if resp_dict.get("error"):
                print(f"Error inserting batch starting at row {start}: {resp_dict['error']}")
            else:
                print(f"Inserted rows {start} to {start + len(batch) - 1}")
            break  # exit retry loop
        except Exception as e:
            retries += 1
            print(f"Exception on batch starting at row {start}, retry {retries}: {e}")
            time.sleep(2)
    else:
        print(f"Failed to insert batch starting at row {start} after {MAX_RETRIES} retries.")

print("Data insertion complete.")
