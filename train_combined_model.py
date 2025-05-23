import os
import joblib
import numpy as np
import pandas as pd
import datetime
import requests
from nsepython import nse_optionchain_scrapper

# Set your Telegram bot token and chat ID here
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SYMBOLS = ["NIFTY", "BANKNIFTY", "FINNIFTY"]

def fetch_all_options(symbols):
    all_data = []
    for sym in symbols:
        try:
            df = nse_optionchain_scrapper(sym)
            df['Underlying'] = sym
            all_data.append(df)
        except Exception as e:
            print(f"Failed to fetch data for {sym}: {e}")
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def send_telegram_message(message):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram credentials missing.")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("Failed to send message:", e)

def run_prediction():
    # Load model from repo root (no downloading)
    if not os.path.exists("intraday_model.pkl") or not os.path.exists("scaler.pkl"):
        print("Model or Scaler file missing in repo root!")
        return

    model = joblib.load("intraday_model.pkl")
    scaler = joblib.load("scaler.pkl")

    df = fetch_all_options(SYMBOLS)
    if df.empty:
        print("No option data found.")
        return

    features = ['strike', 'open', 'high', 'low', 'lastPrice', 'impliedVolatility',
                'volume', 'openInterest', 'Underlying Value']
    df = df.dropna(subset=features)
    X = df[features]
    X_scaled = scaler.transform(X)

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

    df.to_parquet("combined_data.parquet", index=False)

if __name__ == "__main__":
    run_prediction()
