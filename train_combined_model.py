import os
import pandas as pd
import numpy as np
import requests
import joblib
import datetime
import yfinance as yf
from dotenv import load_dotenv

# === LOAD .env VARIABLES ===
load_dotenv("details.env")

# === CONFIG ===
SYMBOLS = ["^NSEI", "^NSEBANK", "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "LT.NS", "SBIN.NS", "ITC.NS", "AXISBANK.NS"]
MODEL_PATH = "models/intraday_model.pkl"
SCALER_PATH = "models/scaler.pkl"
OUTPUT_PARQUET = "combined_data.parquet"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# === FETCH OPTION DATA ===
def fetch_all_options(symbols):
    combined_df = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            expiry = ticker.options[0]
            opt_chain = ticker.option_chain(expiry)
            calls = opt_chain.calls
            puts = opt_chain.puts
            calls["Option type"] = "CE"
            puts["Option type"] = "PE"
            df = pd.concat([calls, puts], ignore_index=True)
            df["Underlying"] = symbol
            df["Underlying Value"] = ticker.info.get("regularMarketPrice", 0)
            combined_df.append(df)
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
    return pd.concat(combined_df, ignore_index=True) if combined_df else pd.DataFrame()

# === TELEGRAM ALERT ===
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

# === PREDICTION LOGIC ===
def run_prediction():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Model or Scaler missing.")
        return

    df = fetch_all_options(SYMBOLS)
    if df.empty:
        print("No data.")
        return

    features = ['strike', 'open', 'high', 'low', 'lastPrice', 'impliedVolatility', 'volume', 'openInterest', 'Underlying Value']
    df = df.dropna(subset=features)
    X = df[features]

    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X)

    model = joblib.load(MODEL_PATH)
    preds = model.predict(X_scaled)

    df['Predicted Close'] = preds
    df['Signal'] = np.where(df['Predicted Close'] > df['lastPrice'], 'BUY', 'SELL')
    df['Confidence'] = abs(df['Predicted Close'] - df['lastPrice']) / df['lastPrice']
    high_conf = df[df['Confidence'] > 0.05]

    if not high_conf.empty:
        now = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        msg = f"<b>Options ML Signals @ {now}</b>\n"
        for _, row in high_conf.iterrows():
            msg += (
                f"{row['Underlying']} {row['Option type']} {row['strike']} | "
                f"{row['Signal']} | LTP: {row['lastPrice']} | "
                f"Target: {row['Predicted Close']:.2f}\n"
            )
        send_telegram_message(msg)

    df.to_parquet(OUTPUT_PARQUET, index=False)

if __name__ == "__main__":
    run_prediction()
