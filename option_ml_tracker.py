# option_ml_tracker.py

import os
import pandas as pd
import numpy as np
import requests
import gdown
import joblib
import datetime
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# === CONFIG ===
SYMBOL = "RELIANCE"
MODEL_PATH = "intraday_model.h5"
SCALER_PATH = "intraday_scaler.pkl"
OUTPUT_CSV = "live_signals.csv"
TELEGRAM_TOKEN = "7121966371:AAEKHVrsqLRswXg64-6Nf3nid-Mbmlmmw5M"
TELEGRAM_CHAT_ID = "7621883960"

# === GDOWN FOR MODELS ===
def download_model():
    gdown.download("https://drive.google.com/uc?id=1saHEBDvVA_rGEolcmKfvMjTN8OtmJMZv", MODEL_PATH, quiet=False)
    gdown.download("https://drive.google.com/uc?id=1XXYYZZ", SCALER_PATH, quiet=False)  # Replace with actual scaler file id

# === FETCH NSE DATA ===
def fetch_nse_option_chain(symbol="RELIANCE"):
    url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol.upper()}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": f"https://www.nseindia.com/option-chain"
    }
    session = requests.Session()
    session.get("https://www.nseindia.com", headers=headers)
    response = session.get(url, headers=headers)
    data = response.json()

    rows = []
    for item in data['records']['data']:
        for option_type in ['CE', 'PE']:
            if option_type in item:
                opt = item[option_type]
                rows.append({
                    'Strike Price': opt['strikePrice'],
                    'Expiry': opt['expiryDate'],
                    'Option type': option_type,
                    'Open': opt.get('openPrice', 0),
                    'High': opt.get('highPrice', 0),
                    'Low': opt.get('lowPrice', 0),
                    'Close': opt.get('closePrice', 0),
                    'LTP': opt.get('lastPrice', 0),
                    'Settle Price': opt.get('settlementPrice', 0),
                    'No. of contracts': opt.get('numberOfContractsTraded', 0),
                    'Turnover': opt.get('turnover', 0),
                    'OI': opt.get('openInterest', 0),
                    'Chng in OI': opt.get('changeinOpenInterest', 0),
                    'Underlying Value': data['records']['underlyingValue']
                })
    return pd.DataFrame(rows)

# === TELEGRAM NOTIFIER ===
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    requests.post(url, data=payload)

# === MAIN INFERENCE FUNCTION ===
def run_prediction():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        download_model()

    df = fetch_nse_option_chain(SYMBOL)
    if df.empty:
        print("No data fetched")
        return

    features = [
        'Strike Price', 'Open', 'High', 'Low', 'LTP', 'Settle Price',
        'No. of contracts', 'Turnover', 'OI', 'Chng in OI', 'Underlying Value'
    ]
    df = df.dropna()
    X = df[features]

    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X)

    model = load_model(MODEL_PATH)
    preds = model.predict(X_scaled)
    df['Predicted Close'] = preds
    df['Signal'] = np.where(df['Predicted Close'] > df['LTP'], 'BUY', 'SELL')

    # Filter only high confidence suggestions (difference > 5%)
    df['Confidence'] = abs(df['Predicted Close'] - df['LTP']) / df['LTP']
    high_conf = df[df['Confidence'] > 0.05]

    if not high_conf.empty:
        now = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        msg = f"<b>Options ML Signal @ {now}</b>\n"
        for _, row in high_conf.iterrows():
            msg += f"{row['Option type']} {row['Strike Price']} | {row['Signal']} | LTP: {row['LTP']} | Target: {row['Predicted Close']:.2f}\n"
        send_telegram_message(msg)

    df.to_csv(OUTPUT_CSV, index=False)

if __name__ == "__main__":
    run_prediction()
