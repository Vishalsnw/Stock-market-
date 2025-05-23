import os
import pandas as pd
import numpy as np
import requests
import joblib
import datetime
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
from nsepython import nse_optionchain_scrapper

# === LOAD .env VARIABLES ===
load_dotenv("details.env")

# === CONFIG ===
TOP_SYMBOLS = ["RELIANCE.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "LT.NS", "SBIN.NS", "AXISBANK.NS", "ITC.NS", "HINDUNILVR.NS"]
INDEX_SYMBOLS = ["NIFTY", "BANKNIFTY"]
MODEL_PATH = "intraday_model.pkl"
SCALER_PATH = "intraday_scaler.pkl"
OUTPUT_PARQUET = "combined_data.parquet"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# === FETCH EQUITY OPTIONS VIA YFINANCE ===
def fetch_yfinance_options(symbol):
    try:
        ticker = yf.Ticker(symbol)
        expiry = ticker.options[0]
        opt_chain = ticker.option_chain(expiry)
    except:
        return pd.DataFrame()

    calls = opt_chain.calls
    puts = opt_chain.puts
    calls["Option type"] = "CE"
    puts["Option type"] = "PE"
    df = pd.concat([calls, puts], ignore_index=True)
    df["Underlying"] = symbol
    df["Underlying Value"] = ticker.info.get("regularMarketPrice", 0)
    return df

# === FETCH INDEX OPTIONS VIA NSEPYTHON ===
def fetch_nse_options():
    all_data = []
    for sym in INDEX_SYMBOLS:
        try:
            df = nse_optionchain_scrapper(sym)
            df["Underlying"] = sym
            all_data.append(df)
        except Exception as e:
            print(f"Failed NSE fetch: {sym} | {e}")
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# === TELEGRAM NOTIFIER ===
def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    requests.post(url, data=payload)

# === ML INFERENCE ===
def run_prediction():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Model or Scaler file missing!")
        return

    # Fetch all option data
    all_df = []

    # Equity Options
    for sym in TOP_SYMBOLS:
        df = fetch_yfinance_options(sym)
        if not df.empty:
            all_df.append(df)

    # Index Options
    df_index = fetch_nse_options()
    if not df_index.empty:
        all_df.append(df_index)

    if not all_df:
        print("No option data found.")
        return

    df = pd.concat(all_df, ignore_index=True)

    # Features
    features = ['strike', 'open', 'high', 'low', 'lastPrice', 'impliedVolatility', 'volume', 'openInterest', 'Underlying Value']
    df = df.dropna(subset=features)
    X = df[features]

    # Predict
    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X)

    model = load_model(MODEL_PATH)
    preds = model.predict(X_scaled)
    df['Predicted Close'] = preds
    df['Signal'] = np.where(df['Predicted Close'] > df['lastPrice'], 'BUY', 'SELL')
    df['Confidence'] = abs(df['Predicted Close'] - df['lastPrice']) / df['lastPrice']

    high_conf = df[df['Confidence'] > 0.05]

    # Telegram
    if not high_conf.empty:
        now = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        msg = f"<b>Options ML Signal @ {now}</b>\n"
        for _, row in high_conf.iterrows():
            msg += (
                f"{row['Underlying']} {row.get('Option type', '')} {row['strike']} | "
                f"{row['Signal']} | LTP: {row['lastPrice']} | "
                f"Target: {row['Predicted Close']:.2f}\n"
            )
        send_telegram_message(msg)

    df.to_parquet(OUTPUT_PARQUET, index=False)

if __name__ == "__main__":
    run_prediction()
