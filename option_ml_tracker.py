import os
import pandas as pd
import numpy as np
import requests
import gdown
import joblib
import datetime
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# === CONFIG ===
SYMBOL = "RELIANCE.NS"
MODEL_PATH = "intraday_model.h5"
SCALER_PATH = "intraday_scaler.pkl"
OUTPUT_CSV = "live_signals.csv"
TELEGRAM_TOKEN = "7121966371:AAEKHVrsqLRswXg64-6Nf3nid-Mbmlmmw5M"
TELEGRAM_CHAT_ID = "7621883960"

# === DOWNLOAD MODELS ===
def download_model():
    gdown.download("https://drive.google.com/uc?id=1saHEBDvVA_rGEolcmKfvMjTN8OtmJMZv", MODEL_PATH, quiet=False)
    gdown.download("https://drive.google.com/uc?id=1XXYYZZ", SCALER_PATH, quiet=False)  # Replace with actual scaler ID

# === FETCH OPTION DATA FROM YFINANCE ===
def fetch_yfinance_options(symbol):
    ticker = yf.Ticker(symbol)
    try:
        expiry = ticker.options[0]
        opt_chain = ticker.option_chain(expiry)
    except:
        return pd.DataFrame()

    calls = opt_chain.calls
    puts = opt_chain.puts

    calls["Option type"] = "CE"
    puts["Option type"] = "PE"

    df = pd.concat([calls, puts], ignore_index=True)
    df["Underlying Value"] = ticker.info.get("regularMarketPrice", 0)
    return df

# === TELEGRAM NOTIFIER ===
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    requests.post(url, data=payload)

# === RUN ML INFERENCE ===
def run_prediction():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        download_model()

    df = fetch_yfinance_options(SYMBOL)
    if df.empty:
        print("No option data found.")
        return

    features = ['strike', 'open', 'high', 'low', 'lastPrice', 'impliedVolatility', 'volume', 'openInterest', 'Underlying Value']
    df = df.dropna(subset=features)
    X = df[features]

    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X)

    model = load_model(MODEL_PATH)
    preds = model.predict(X_scaled)
    df['Predicted Close'] = preds
    df['Signal'] = np.where(df['Predicted Close'] > df['lastPrice'], 'BUY', 'SELL')

    df['Confidence'] = abs(df['Predicted Close'] - df['lastPrice']) / df['lastPrice']
    high_conf = df[df['Confidence'] > 0.05]

    if not high_conf.empty:
        now = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        msg = f"<b>Options ML Signal @ {now}</b>\n"
        for _, row in high_conf.iterrows():
            msg += f"{row['Option type']} {row['strike']} | {row['Signal']} | LTP: {row['lastPrice']} | Target: {row['Predicted Close']:.2f}\n"
        send_telegram_message(msg)

    df.to_csv(OUTPUT_CSV, index=False)

if __name__ == "__main__":
    run_prediction()
