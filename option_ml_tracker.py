import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings("ignore")
load_dotenv("details.env")

# === CONFIG ===
MODEL_PATH = "intraday_model.pkl"
SCALER_PATH = "scaler.pkl"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
ACTIVE_TRADES_FILE = "active_trades.json"
DEBUG_MODE = os.getenv("DEBUG", "0") == "1"

INDEX_SYMBOLS = ["NIFTY", "BANKNIFTY"]
STOCK_SYMBOLS = ["RELIANCE", "INFY", "TCS", "HDFCBANK", "ICICIBANK", "LT", "SBIN", "AXISBANK", "ITC", "HINDUNILVR"]

# === TELEGRAM ===
def send_telegram(msg):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
        try:
            requests.post(url, data=payload)
        except:
            pass

# === DATA FETCHING ===
def fetch_option_chain(symbol, is_index=True):
    base_url = "https://www.nseindia.com"
    api_url = f"https://www.nseindia.com/api/option-chain-{'indices' if is_index else 'equities'}?symbol={symbol}"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "*/*",
        "Referer": "https://www.nseindia.com/option-chain",
        "Connection": "keep-alive"
    }

    session = requests.Session()
    session.headers.update(headers)
    try:
        session.get(base_url, timeout=5)
        response = session.get(api_url, timeout=10)
        data = response.json()
        records = []

        for item in data["records"]["data"]:
            if 'CE' in item and 'PE' in item:
                ce, pe = item['CE'], item['PE']
                records.append({
                    'strike': ce['strikePrice'],
                    'lastPrice': ce['lastPrice'],
                    'open': ce.get('openPrice', 0),
                    'high': ce.get('highPrice', 0),
                    'low': ce.get('lowPrice', 0),
                    'impliedVolatility': ce.get('impliedVolatility', 0),
                    'volume': ce.get('totalTradedVolume', 0),
                    'openInterest': ce.get('openInterest', 0),
                    'Underlying Value': ce.get('underlyingValue', 0),
                    'Option type': 'CE',
                    'Symbol': symbol
                })
                records.append({
                    'strike': pe['strikePrice'],
                    'lastPrice': pe['lastPrice'],
                    'open': pe.get('openPrice', 0),
                    'high': pe.get('highPrice', 0),
                    'low': pe.get('lowPrice', 0),
                    'impliedVolatility': pe.get('impliedVolatility', 0),
                    'volume': pe.get('totalTradedVolume', 0),
                    'openInterest': pe.get('openInterest', 0),
                    'Underlying Value': pe.get('underlyingValue', 0),
                    'Option type': 'PE',
                    'Symbol': symbol
                })
        return pd.DataFrame(records)
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

# === LOAD MODEL AND SCALER ===
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def load_active_trades():
    return json.load(open(ACTIVE_TRADES_FILE)) if os.path.exists(ACTIVE_TRADES_FILE) else {}

def save_active_trades(data):
    with open(ACTIVE_TRADES_FILE, "w") as f:
        json.dump(data, f, indent=2)

def run_prediction():
    print("Fetching data and predicting...")
    all_df = []
    for symbol in INDEX_SYMBOLS + STOCK_SYMBOLS:
        df = fetch_option_chain(symbol, is_index=(symbol in INDEX_SYMBOLS))
        if not df.empty:
            all_df.append(df)

    if not all_df:
        print("No data fetched.")
        return

    df = pd.concat(all_df, ignore_index=True)
    features = ['strike', 'open', 'high', 'low', 'lastPrice', 'impliedVolatility', 'volume', 'openInterest', 'Underlying Value']
    df = df.dropna(subset=features)
    df = df[df['lastPrice'] > 0]

    X = df[features].astype(float)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    df['Predicted Close'] = preds
    df['Confidence'] = abs(df['Predicted Close'] - df['lastPrice']) / df['lastPrice']
    df['Signal'] = np.where((df['Predicted Close'] > df['lastPrice']) & (df['Confidence'] >= 0.05), 'BUY', 'HOLD')

    active_trades = load_active_trades()

    for _, row in df[df['Signal'] == 'BUY'].iterrows():
        key = f"{row['Symbol']}_{row['strike']}_{row['Option type']}"
        if key not in active_trades:
            buy_price = float(row['lastPrice'])
            target = round(buy_price * 1.1, 2)
            sl = round(buy_price * 0.95, 2)
            active_trades[key] = {
                "symbol": row['Symbol'],
                "strike": row['strike'],
                "type": row['Option type'],
                "buy_price": buy_price,
                "target": target,
                "sl": sl,
                "entry_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            msg = (
                f"<b>NEW BUY SIGNAL</b>\n"
                f"{row['Symbol']} {row['Option type']} {row['strike']}\n"
                f"Buy Price: {buy_price}\nTarget: {target} | SL: {sl}"
            )
            send_telegram(msg)

    save_active_trades(active_trades)

# === MAIN LOOP ===
if __name__ == "__main__":
    send_telegram("NSE Option ML Tracker Started.")
    while True:
        try:
            run_prediction()
        except Exception as e:
            send_telegram(f"Error in prediction loop: {str(e)}")
        if DEBUG_MODE:
            break
        time.sleep(60)
