# app.py
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data import load_data
from training_prediction import train_model, predict_next
from db_utils import insert_prediction, get_last_prediction_error
from earnings import get_todays_earnings

st.cache_resource.clear()

# Sidebar: earnings table
with st.sidebar:
    st.subheader("📊 Today's Earnings Releases")
    earnings_df = get_todays_earnings()
    st.dataframe(earnings_df, use_container_width=True)

st.title("📈 Stock Price Predictor")

# Stock selector
ticker = st.selectbox("Choose Stock:", ["AAPL", "GOOGL", "NVDA", "WMT","AMZN", "MSFT", "TSLA", "META", "AVGO","PLTR"])

if st.button("Train and Predict"):
    # Load and preprocess data
    X, y, scaler, raw_df = load_data(ticker, start="2020-01-01")

    # Train model
    model = train_model(X, y)

    # Get latest close and dates
    latest_close = float(raw_df["Close"].iloc[-1])
    last_trading_date = raw_df.index[-1].date()
    next_trading_date = last_trading_date + timedelta(days=1)
    while next_trading_date.weekday() >= 5:  # skip weekend
        next_trading_date += timedelta(days=1)

    # Predict for today (no error correction)
        # Predict for today (no error correction)
    today_pred = predict_next(model, X[-1], scaler)

    # Log prediction for today
    inserted_today = insert_prediction(ticker, last_trading_date, latest_close, today_pred)

    # Calculate correction error
    last_error = latest_close - today_pred

    # Make a fresh copy of X[-1] for tomorrow
    X_tomorrow = X[-1].copy()
    X_tomorrow[-1, -1] = scaler.transform([[0, 0, 0, 0, 0, last_error]])[0][-1]  # inject scaled error

    # Predict tomorrow using corrected input
    tomorrow_pred = predict_next(model, X_tomorrow, scaler)


    # Log forecast for tomorrow (actual_close = None)
    inserted_forecast = insert_prediction(ticker, next_trading_date, None, tomorrow_pred)

    # Feedback
    if inserted_today:
        st.info(f"✅ Logged today's prediction for {last_trading_date}.")
    else:
        st.warning(f"⚠️ Already logged or DB error for {last_trading_date}.")

    if inserted_forecast:
        st.info(f"🕒 Forecast for {next_trading_date} logged.")
    else:
        st.warning(f"⚠️ Forecast for {next_trading_date} already exists or DB error.")

    # Plot last 100 prices + prediction
    y_dates = raw_df.index[-len(y):]
    dummy = np.zeros((len(y[-100:]), 6))
    dummy[:, 3] = y[-100:]  # close price is 4th column
    true_close_prices = scaler.inverse_transform(dummy)[:, 3]

    fig, ax = plt.subplots()
    ax.plot(y_dates[-100:], true_close_prices, label="True Price")
    ax.plot([y_dates[-1]], [tomorrow_pred], label="Forecast", linestyle='--', marker='o', color='orange')
    ax.set_title(f"{ticker} - Last 100 Days Closing Prices + Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Show tomorrow’s forecast
    st.success(f"Predicted Next Closing Price for {ticker}: **${tomorrow_pred:.2f}**")
