# db_utils.py
import psycopg2
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def insert_prediction(ticker, date, actual, predicted): #if both actual and predicted prices are there it adds them
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        cur = conn.cursor()

        if actual is not None:
            insert_query = """
            INSERT INTO predictions (ticker, date, actual_close, predicted_close)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (ticker, date, actual_close, predicted_close) DO NOTHING; 
            """ 
            #on conflict helps not insert duplicate queries
            cur.execute(insert_query, (ticker, date, round(actual, 2), round(predicted, 2)))
        else:
            insert_query = """
            INSERT INTO predictions (ticker, date, actual_close, predicted_close)
            VALUES (%s, %s, NULL, %s)
            ON CONFLICT (ticker, date, actual_close, predicted_close) DO NOTHING;
            """
            cur.execute(insert_query, (ticker, date, round(predicted, 2)))

        conn.commit()
        cur.close()
        conn.close()
        return True

    except Exception as e: #returns false if there is an error
        print(f"[DB ERROR] {e}")
        return False

def get_last_prediction_error(ticker): #finds the most recent prediction that includes both actual and predicted price and then computes error
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        cur = conn.cursor()

        query = """
        SELECT actual_close, predicted_close
        FROM predictions
        WHERE ticker = %s
        AND actual_close IS NOT NULL
        ORDER BY date DESC
        LIMIT 1;
        """
        #filters queries only which has actual closing price
        cur.execute(query, (ticker,))
        result = cur.fetchone()

        cur.close()
        conn.close()

        if result and result[0] is not None and result[1] is not None:
            actual, predicted = result
            return actual - predicted
        else:
            return 0.0
# if nothing valid is there then return 0.0
    except Exception as e:
        print(f"[DB ERROR - get_last_prediction_error] {e}")
        return 0.0
