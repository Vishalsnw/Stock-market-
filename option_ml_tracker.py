import os
import json
import time
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from nsepython import nse_optionchain_scrapper
import joblib

# === CONFIG ===
load_dotenv("details.env")
MODEL_PATH = "intraday_model.pkl"
SCALER_PATH = "intraday_scaler.pkl"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TOP_SYMBOLS = ["RELIANCE.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "LT.NS", "SBIN.NS", "AXISBANK.NS", "ITC.NS", "HINDUNILVR.NS"]
INDEX_SYMBOLS = ["NIFTY", "BANKNIFTY"]
ACTIVE_TRADES_FILE = "active_trades.json"

# === LOAD MODEL ===
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# === TELEGRAM ===
def send_telegram(message):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
        requests.post(url, data=payload)

# === FETCH OPTIONS ===
def fetch_yfinance_options(symbol):
    try:
        ticker = yf.Ticker(symbol)
        expiry = ticker.options[0]
        opt_chain = ticker.option_chain(expiry)
        calls, puts = opt_chain.calls, opt_chain.puts
        calls["Option type"], puts["Option type"] = "CE", "PE"
        df = pd.concat([calls, puts], ignore_index=True)
        df["Underlying"] = symbol
        df["Underlying Value"] = ticker.info.get("regularMarketPrice", 0)
        return df
    except:
        return pd.DataFrame()

def fetch_nse_options():
    all_data = []
    for sym in INDEX_SYMBOLS:
        try:
            df = nse_optionchain_scrapper(sym)
            df["Underlying"] = sym
            all_data.append(df)
        except:
            pass
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# === ACTIVE TRADES ===
def load_active_trades():
    if os.path.exists(ACTIVE_TRADES_FILE):
        with open(ACTIVE_TRADES_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_active_trades(data):
    with open(ACTIVE_TRADES_FILE, 'w') as f:
        json.dump(data, f, indent=2)

# === MAIN PREDICTION LOGIC ===
def run_prediction():
    df_all = []
    for sym in TOP_SYMBOLS:
        df = fetch_yfinance_options(sym)
        if not df.empty:
            df_all.append(df)
    df_index = fetch_nse_options()
    if not df_index.empty:
        df_all.append(df_index)
    if not df_all:
        return

    df = pd.concat(df_all, ignore_index=True)
    features = ['strike', 'open', 'high', 'low', 'lastPrice', 'impliedVolatility', 'volume', 'openInterest', 'Underlying Value']
    df = df.dropna(subset=features)
    X = df[features].astype(float)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    df['Predicted Close'] = preds
    df['Confidence'] = abs(df['Predicted Close'] - df['lastPrice']) / df['lastPrice']
    df['Signal'] = np.where((df['Predicted Close'] > df['lastPrice']) & (df['Confidence'] >= 0.9), 'BUY', 'HOLD')

    active_trades = load_active_trades()

    for _, row in df[df['Signal'] == 'BUY'].iterrows():
        key = f"{row['Underlying']}_{row['strike']}_{row['Option type']}"
        if key not in active_trades:
            buy_price = float(row['lastPrice'])
            target = round(buy_price * 1.10, 2)
            sl = round(buy_price * 0.95, 2)
            active_trades[key] = {
                "symbol": row['Underlying'],
                "strike": row['strike'],
                "type": row['Option type'],
                "buy_price": buy_price,
                "target": target,
                "sl": sl,
                "entry_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            msg = (
                f"<b>NEW BUY SIGNAL</b>\n"
                f"{row['Underlying']} {row['Option type']} {row['strike']}\n"
                f"Buy Price: {buy_price}\nTarget: {target} | SL: {sl}"
            )
            send_telegram(msg)

    save_active_trades(active_trades)
    check_exit_conditions(active_trades)

# === EXIT LOGIC ===
def check_exit_conditions(active_trades):
    updated = {}
    for key, trade in active_trades.items():
        symbol = trade['symbol']
        df = fetch_yfinance_options(symbol)
        if df.empty:
            updated[key] = trade
            continue
        match = df[(df['strike'] == float(trade['strike'])) & (df['Option type'] == trade['type'])]
        if match.empty:
            updated[key] = trade
            continue
        ltp = float(match.iloc[0]['lastPrice'])
        if ltp >= trade['target']:
            send_telegram(f"<b>TARGET HIT:</b> {symbol} {trade['type']} {trade['strike']}\nPrice: {ltp}")
        elif ltp <= trade['sl']:
            send_telegram(f"<b>SL HIT:</b> {symbol} {trade['type']} {trade['strike']}\nPrice: {ltp}")
        elif model.predict(scaler.transform([[float(trade['strike']), ltp, ltp, ltp, ltp, 0, 0, 0, ltp]]))[0] < ltp:
            send_telegram(f"<b>ML EXIT SIGNAL:</b> {symbol} {trade['type']} {trade['strike']}\nPrice: {ltp}")
        else:
            updated[key] = trade
    save_active_trades(updated)

if __name__ == "__main__":
    while True:
        run_prediction()
        time.sleep(60)
