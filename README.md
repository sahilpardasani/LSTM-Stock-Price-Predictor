# LSTM-Stock-Price-Predictor
This project uses a Long Short-Term Memory (LSTM) model to predict next-day closing prices for popular stocks based on the past 90 days of data.

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
<img width="1542" alt="Screenshot 2025-06-17 at 12 22 47 AM" src="https://github.com/user-attachments/assets/364c0b7e-a6c0-455a-931b-dc79a37be8ec" />
<img width="1436" alt="Screenshot 2025-06-17 at 12 24 47 AM" src="https://github.com/user-attachments/assets/5c558bc3-9d89-45d0-b142-28081c2a77f3" />
<img width="505" alt="Screenshot 2025-06-17 at 12 25 15 AM" src="https://github.com/user-attachments/assets/46be2922-04dd-4f66-98b9-69bcf445071b" />

