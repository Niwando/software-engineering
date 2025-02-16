#!/usr/bin/env python3
import os
from app.api.alphavantage.fetch import fetch_stock_data_only, insert_fetched_data
from lstm.finetune import fine_tune_all_stocks
from lstm.pred import generate_and_insert_predictions

def main():
    # Umgebungsvariable auslesen
    API_KEY = os.environ.get("API_KEY")
    if not API_KEY:
        raise ValueError("API_KEY ist nicht gesetzt.")

    # 1.1 Fetch new data from the API and store it in memory.
    print("Step 1.1: Fetching new data from the API (fetch-only)...")
    fetched_data = fetch_stock_data_only(API_KEY)
    
    # 1.2 Insert the fetched data into the database.
    print("Step 1.2: Inserting fetched data into the database...")
    insert_fetched_data(fetched_data)
    
    # 2. Fine-tune models using data queried directly from the database.
    print("Step 2: Fine-tuning models using in-database data...")
    fine_tune_all_stocks()
    
    # 3. Generate predictions and insert them into the database.
    print("Step 3: Generating predictions and inserting them into the database...")
    generate_and_insert_predictions()
    
    print("All steps complete.")

if __name__ == "__main__":
    main()
