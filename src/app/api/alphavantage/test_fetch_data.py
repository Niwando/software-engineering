# test_fetch_data.py
import os
import json
import tempfile
import pytest
from unittest.mock import MagicMock
from fetch import fetch_and_save_data 

def test_fetch_and_save_data(tmp_path, monkeypatch):
    # Setup: use a temporary directory to simulate the 'data' folder
    temp_data_dir = tmp_path / "data"
    temp_data_dir.mkdir()

    # We'll override os.path.exists and os.makedirs for the 'data' folder.
    original_exists = os.path.exists
    def fake_exists(path):
        # If the path equals "data", return True (simulate that the folder exists)
        if path == "data":
            return True
        return original_exists(path)
    monkeypatch.setattr(os.path, "exists", fake_exists)
    
    # Override os.makedirs to do nothing if called with "data"
    monkeypatch.setattr(os, "makedirs", lambda path, exist_ok: None)

    # Also override open() so that files under "data" are written to our temp_data_dir.
    original_open = open
    def fake_open(file, mode, *args, **kwargs):
        # If file starts with "data", write to our temp directory.
        if file.startswith("data"):
            filename = os.path.basename(file)
            file = os.path.join(temp_data_dir, filename)
        return original_open(file, mode, *args, **kwargs)
    monkeypatch.setattr("builtins.open", fake_open)

    # Setup dummy API response.
    symbol = "AAPL"
    api_key = "dummy_key"
    dummy_json = {
        "Meta Data": {
            "2. Symbol": symbol,
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
    
    # Patch requests.get so that it returns a dummy response.
    dummy_response = MagicMock()
    dummy_response.json.return_value = dummy_json
    dummy_response.raise_for_status.return_value = None
    monkeypatch.setattr("requests.get", lambda url: dummy_response)
    
    # Call the function under test.
    fetch_and_save_data(symbol, api_key)
    
    # Check that the file was created in our temporary directory.
    expected_file = temp_data_dir / f"{symbol}.json"
    assert expected_file.exists(), f"Expected file {expected_file} does not exist."
    
    # Load the file and check its content.
    with open(expected_file, "r") as f:
        loaded_data = json.load(f)
    assert loaded_data == dummy_json, "The saved JSON data does not match the expected dummy JSON."

    print("Test passed: Data fetched and saved correctly.")
