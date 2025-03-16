## Description:
This project is a Stock Price Prediction Web Application that utilizes LSTM (Long Short-Term Memory) Neural Networks to forecast stock prices. The web interface is built using Streamlit and allows users to input a stock ticker symbol to visualize historical stock data, moving averages, and predicted stock prices.

## Features:
✅ Fetch Real-time Stock Data using Yahoo Finance API
✅ Predict Future Stock Prices using a trained LSTM model
✅ Visualize Moving Averages (50-day, 100-day, 200-day)
✅ Interactive Web UI using Streamlit
✅ Flask API Endpoint for external stock price predictions

## Technologies Used:
Python
TensorFlow/Keras (LSTM Model)
Pandas, NumPy (Data Handling)
Matplotlib, Streamlit (Visualization & UI)
Yahoo Finance API (Stock Data)
Flask (API for Predictions)

## Project Structure:
📁 Stock-Prediction-App  
│── 📄 app.py               # Streamlit-based Web App  
│── 📄 api.py               # Flask API for prediction  
│── 📄 model_training.py     # LSTM Model Training Script  
│── 📄 Stock Predictions Model.keras  # Pre-trained model file  
│── 📄 requirements.txt      # List of dependencies  
│── 📄 README.md             # Project Documentation  

## How to Run the Project?
1️ Clone the Repository
```
git clone https://github.com/yourusername/Stock-Prediction-App.git
cd Stock-Prediction-App
```
2️ Install Dependencies
```
pip install -r requirements.txt
```
3 Run the Streamlit Web App
```
streamlit run app.py
```
4 Run the Flask API
```
python api.py
```
## Usage:
Enter a stock symbol (e.g., AAPL, TSLA, GOOG).
View historical stock data and moving averages.
Get a predicted stock price for the next day.

## Contributing:
Feel free to fork this repo, raise issues, or submit pull requests to improve this project!
For any queries, reach out at kamalraj3106@gmail.com
