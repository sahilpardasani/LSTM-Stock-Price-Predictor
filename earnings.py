from finance_calendars import finance_calendars as fc
import pandas as pd

def get_todays_earnings(): #func that fetches today's earnings data
    try:
        df = fc.get_earnings_today() #dataframe to store earnings data
        print("Earnings Columns:", df.columns)  # Print actual columns

        if df.empty:
            return pd.DataFrame({"Message": ["No earnings data today."]}) #if no earnings print this msg

        return df  # Temporarily return raw dataframe to inspect it in Streamlit
    except Exception as e:
        return pd.DataFrame({"Error": [f"Failed to load earnings: {e}"]}) # catch any error like API down or something else
