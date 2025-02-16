#!/usr/bin/env python3
import os
import requests
import json
import pandas as pd
import time
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

def fetch_stock_data_only(api_key):
    """
    Fetch stock data from the API for each symbol and return a dictionary 
    mapping stock symbols to their JSON responses. This function does not insert
    data into the database.
    """
    stocks = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "NFLX", "AVGO", "PYPL"]
    fetched_data = {}
    
    for symbol in stocks:
        print(f"Fetching data for {symbol}...")
        url = (
            f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY'
            f'&symbol={symbol}&interval=1min&outputsize=full&apikey={api_key}'
        )
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            fetched_data[symbol] = data
            print(f"Data fetched for {symbol}.")
            # Pause to help avoid hitting the API rate limit (adjust the sleep time as needed).
            time.sleep(12)
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
    
    return fetched_data

def insert_fetched_data(fetched_data):
    """
    Process the fetched data and insert new records into the 'stock_data' table 
    in the database.
    """
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set as environment variables.")
    supabase: Client = create_client(url, key)

    for symbol, data in fetched_data.items():
        print(f"\nProcessing fetched data for {symbol}...")
        meta = data.get("Meta Data")
        if not meta:
            print(f"No 'Meta Data' found for {symbol}. Skipping.")
            continue

        interval = meta.get("4. Interval")
        ts_key = f"Time Series ({interval})"
        ts_data = data.get(ts_key)
        if not ts_data:
            print(f"No time series data ({ts_key}) for {symbol}. Skipping.")
            continue

        try:
            # Convert the JSON time series data into a DataFrame.
            df = pd.DataFrame(ts_data).T
            # Rename columns (e.g., "1. open" -> "open")
            df.columns = [col.split(". ")[-1] for col in df.columns]
            df = df.apply(pd.to_numeric)
            df.index = pd.to_datetime(df.index)
            df["symbol"] = symbol
            df.reset_index(inplace=True)
            df.rename(columns={"index": "timestamp"}, inplace=True)
        except Exception as e:
            print(f"Error processing data for {symbol}: {e}")
            continue

        # Query the DB for the latest timestamp for this symbol.
        try:
            response_db = supabase.table("stock_data")\
                .select("timestamp")\
                .eq("symbol", symbol)\
                .order("timestamp", desc=True)\
                .limit(1)\
                .execute()
            existing = response_db.dict().get("data", [])
            if existing:
                max_ts = pd.to_datetime(existing[0]["timestamp"])
                df_new = df[df["timestamp"] > max_ts]
                print(f"Existing data found for {symbol}, max timestamp: {max_ts}")
            else:
                print(f"No existing data for {symbol}. Inserting all rows.")
                df_new = df
        except Exception as e:
            print(f"Error querying DB for {symbol}: {e}")
            continue

        num_new = len(df_new)
        print(f"New rows to insert for {symbol}: {num_new}")
        if num_new == 0:
            continue

        # Prepare records for insertion.
        data_records = df_new.to_dict(orient="records")
        for row in data_records:
            if not isinstance(row["timestamp"], str):
                row["timestamp"] = row["timestamp"].isoformat()

        # Insert records in batches.
        BATCH_SIZE = 50
        MAX_RETRIES = 3
        num_rows = len(data_records)
        for start in range(0, num_rows, BATCH_SIZE):
            batch = data_records[start:start+BATCH_SIZE]
            retries = 0
            while retries < MAX_RETRIES:
                try:
                    res = supabase.table("stock_data").insert(batch, upsert=True).execute()
                    resp = res.dict()
                    if resp.get("error"):
                        print(f"Error inserting batch for {symbol} starting at row {start}: {resp['error']}")
                    else:
                        print(f"Inserted rows {start} to {start+len(batch)-1} for {symbol}")
                    break
                except Exception as ex:
                    retries += 1
                    print(f"Exception inserting batch for {symbol} at row {start}, retry {retries}: {ex}")
                    time.sleep(2)
            else:
                print(f"Failed to insert batch for {symbol} starting at row {start} after {MAX_RETRIES} retries.")

if __name__ == "__main__":
    
    API_KEY = "D3WKQGJTAEZJSVBI"
    data = fetch_stock_data_only(API_KEY)
    
    insert_fetched_data(data)
