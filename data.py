# data.py
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from db_utils import get_last_prediction_error
from training_prediction import set_error_scaling_bounds

def get_last_trading_day():
    today = datetime.now() # grabs current date and time
    if today.weekday() >= 5:  # Weekend 0 is Mon and so on
        days_to_subtract = today.weekday() - 4 #5-4=1 so go 1 day back and 6-4=2 so go two days back
        today = today - timedelta(days=days_to_subtract)
    return today.strftime('%Y-%m-%d') #returns final date in format for yfinance 

def load_data(ticker, start="2020-01-01", end=None): #downloads historical data from yfinance
    end_date = get_last_trading_day() 
    df = yf.download(ticker, start=start, end=end_date, progress=False)

    if df.empty or len(df) < 91:
        raise ValueError("Not enough data for LSTM sequence. Need at least 91 days.") #make sure there are 91 timestamps available to predict

    # Features: Open, High, Low, Close, Volume
    features = df[["Open", "High", "Low", "Close", "Volume"]].values #grabs he 5 most important columns

    # Get last prediction error from DB
    last_error = get_last_prediction_error(ticker)
    error_column = np.zeros((features.shape[0], 1)) #creates an extra column where only latest row has prediction error
    error_column[-1] = last_error if last_error is not None else 0

    # Combine features + error column
    features = np.hstack((features, error_column)) #stacks the 6th feature onto the exsisting 5

    scaler = MinMaxScaler() #scales all 6 features to range [0, 1]
    scaled = scaler.fit_transform(features) 
    set_error_scaling_bounds( #stores min and max values of the error column for later use
    scaler.data_min_[-1],  
    scaler.data_max_[-1]
)

    sequence_length = 90
    X, y = [], []

    for i in range(len(scaled) - sequence_length):
        X.append(scaled[i:i+sequence_length])       #input sequence of shape [90,6] 90 timestamps and 6 features    
        y.append(scaled[i+sequence_length][3])      # output next day's closing price 

    return np.array(X), np.array(y), scaler, df #x is input t model, y traget values, scalar to reverse prediction back to price, df is raw price data 