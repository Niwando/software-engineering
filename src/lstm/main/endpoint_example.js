fetch("http://<your-server-address>:5000/predict?symbol=AAPL")
  .then(response => response.json())
  .then(data => {
      console.log(data);
      // Process the forecast data as needed.
  })
  .catch(error => console.error("Error:", error));
