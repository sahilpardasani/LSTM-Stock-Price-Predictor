# LSTM-Stock-Price-Predictor
This project uses a Long Short-Term Memory (LSTM) model to predict next-day closing prices for popular stocks based on the past 90 days of data.

## Project Structure
📁 stock-price-predictor-lstm/
├── app.py                     # Streamlit UI
├── data.py                   # Data loading & feature engineering
├── db_utils.py               # PostgreSQL DB interaction
├── earnings.py              # Pulls today's earnings via API
├── model.py                 # LSTM model architecture
├── training_prediction.py   # Model training + prediction logic
├── requirements.txt         # All dependencies
├── .env.example             # Template for env vars (exclude real secrets)

## 🧠 Features
- Uses 6 features: Open, High, Low, Close, Volume, and previous prediction error.
- Automatically logs actual vs predicted prices into a PostgreSQL database.
- Predicts future prices using corrected error from past performance.
- Optional earnings calendar sidebar using `finance_calendars`.

## 🔧 Tech Stack
- Streamlit (UI)
- PyTorch (LSTM model)
- yFinance (Stock data)
- PostgreSQL (Data storage)
- Python + NumPy + Scikit-Learn

## 🚀 How It Works
1. Select a stock (e.g. AAPL, NVDA)
2. Train model on past 90 days
3. Predict today's and tomorrow's price
4. Log predictions to database

# Future Improvements
Add sentiment-based news correction (via FinBERT)
Improve forecast speed
Add Dockerfile for easy deployment

## 📦 Setup (Optional for local run)
```bash
pip install -r requirements.txt
streamlit run app.py
