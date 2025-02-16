# src/tests/test_fetch_data.py

import sys
import os

# Insert the 'src' directory into sys.path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
from unittest.mock import MagicMock
import pytest
from app.api.alphavantage.fetch import fetch_stock_data_only

def test_fetch_stock_data_only(monkeypatch):
    # Dummy API response to simulate a real response.
    dummy_json = {
        "Meta Data": {
            "2. Symbol": "AAPL",
            "4. Interval": "1min"
        },
        "Time Series (1min)": {
            "2024-01-01 09:30:00": {
                "1. open": "150.0",
                "2. high": "151.0",
                "3. low": "149.5",
                "4. close": "150.5",
                "5. volume": "10000"
            }
        }
    }
    
    # Patch time.sleep to avoid actual delay.
    monkeypatch.setattr(time, "sleep", lambda x: None)
    
    # Patch requests.get to always return a dummy response.
    dummy_response = MagicMock()
    dummy_response.json.return_value = dummy_json
    dummy_response.raise_for_status.return_value = None
    monkeypatch.setattr("requests.get", lambda url: dummy_response)
    
    # Call the function with a dummy API key.
    api_key = "dummy_key"
    fetched_data = fetch_stock_data_only(api_key)
    
    expected_symbols = [
        "AAPL", "MSFT", "NVDA", "TSLA", "AMZN",
        "GOOGL", "META", "NFLX", "AVGO", "PYPL"
    ]
    
    # Verify that fetched_data is a dict and contains the expected symbols.
    assert isinstance(fetched_data, dict), "Returned object is not a dictionary."
    
    for symbol in expected_symbols:
        assert symbol in fetched_data, f"Missing symbol {symbol} in fetched data."
        assert fetched_data[symbol] == dummy_json, f"Data for {symbol} does not match the expected dummy JSON."
    
    print("Test passed: fetch_stock_data_only returns the correct dummy data for all symbols.")
