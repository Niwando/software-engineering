import os
import requests
import json

# Function to fetch and save data for a specific symbol
def fetch_and_save_data(symbol, api_key):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&outputsize=full&apikey={api_key}'
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        
        try:
            data = response.json()
            
            # Ensure the 'data' directory exists
            directory = 'data'
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            # Save the JSON file
            file_path = os.path.join(directory, f"{symbol}.json")
            with open(file_path, "w") as file:
                json.dump(data, file, indent=4)
            
            print(f"Data for {symbol} saved to {file_path}")
        
        except Exception as e:
            print(f"Error saving data for {symbol}: {e}")
    
    except requests.exceptions.RequestException as req_err:
        print(f"HTTP request failed for {symbol}: {req_err}")

# List of stock symbols
stocks = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "NFLX", "AVGO", "PYPL"]

# Set your API key here
api_key = "D3WKQGJTAEZJSVBI"

# Loop over all stocks and fetch data for each
for stock in stocks:
    fetch_and_save_data(stock, api_key)
