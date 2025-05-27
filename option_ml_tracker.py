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
import warnings

warnings.filterwarnings("ignore")

# === CONFIG ===
load_dotenv("details.env")
MODEL_PATH = "intraday_model.pkl"
SCALER_PATH = "scaler.pkl"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TOP_SYMBOLS = ["RELIANCE.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "LT.NS", "SBIN.NS", "AXISBANK.NS", "ITC.NS", "HINDUNILVR.NS"]
INDEX_SYMBOLS = ["NIFTY", "BANKNIFTY"]
ACTIVE_TRADES_FILE = "active_trades.json"
DEBUG_MODE = os.getenv("DEBUG", "0") == "1"

# === TELEGRAM ===
def send_telegram(message):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
            requests.post(url, data=payload)
        except Exception as e:
            print(f"Telegram Error: {e}")

# === LOGGING ===
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# === LOAD MODEL AND SCALER ===
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    send_telegram("Model or Scaler file not found.")
    raise FileNotFoundError("Model or Scaler file missing.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# === FETCH OPTIONS ===
def fetch_yfinance_options(symbol):
    try:
        ticker = yf.Ticker(symbol)
        if not ticker.options:
            log(f"YFinance Error for {symbol}: No options data available")
            return pd.DataFrame()
        expiry = ticker.options[0]
        opt_chain = ticker.option_chain(expiry)
        calls, puts = opt_chain.calls, opt_chain.puts
        calls["Option type"], puts["Option type"] = "CE", "PE"
        df = pd.concat([calls, puts], ignore_index=True)
        df["Underlying"] = symbol
        df["Underlying Value"] = ticker.info.get("regularMarketPrice", 0)
        return df
    except Exception as e:
        log(f"YFinance Error for {symbol}: {e}")
        return pd.DataFrame()

def fetch_nse_options():
    all_data = []
    for sym in INDEX_SYMBOLS:
        try:
            df = nse_optionchain_scrapper(sym)
            df["Underlying"] = sym
            all_data.append(df)
        except Exception as e:
            log(f"NSE fetch failed for {sym}: {e}")
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
    log("Starting prediction cycle...")
    send_telegram("Prediction cycle started.")

    df_all = []
    for sym in TOP_SYMBOLS:
        df = fetch_yfinance_options(sym)
        if not df.empty:
            df_all.append(df)

    df_index = fetch_nse_options()
    if not df_index.empty:
        df_all.append(df_index)

    if not df_all:
        log("No data fetched. Skipping this cycle.")
        return

    df = pd.concat(df_all, ignore_index=True)

    features = ['strike', 'open', 'high', 'low', 'lastPrice', 'impliedVolatility', 'volume', 'openInterest', 'Underlying Value']
    df = df.dropna(subset=features)

    if df.empty:
        log("DataFrame empty after dropping NA.")
        return

    X = df[features].astype(float)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    df['Predicted Close'] = preds
    df['Confidence'] = abs(df['Predicted Close'] - df['lastPrice']) / df['lastPrice']
    df['Signal'] = np.where((df['Predicted Close'] > df['lastPrice']) & (df['Confidence'] >= 0.05), 'BUY', 'HOLD')

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
        test_input = [[float(trade['strike']), ltp, ltp, ltp, ltp, 0, 0, 0, ltp]]
        test_scaled = scaler.transform(test_input)
        pred_close = model.predict(test_scaled)[0]

        if ltp >= trade['target']:
            send_telegram(f"<b>TARGET HIT:</b> {symbol} {trade['type']} {trade['strike']}\nPrice: {ltp}")
        elif ltp <= trade['sl']:
            send_telegram(f"<b>SL HIT:</b> {symbol} {trade['type']} {trade['strike']}\nPrice: {ltp}")
        elif pred_close < ltp:
            send_telegram(f"<b>ML EXIT SIGNAL:</b> {symbol} {trade['type']} {trade['strike']}\nPrice: {ltp}")
        else:
            updated[key] = trade
    save_active_trades(updated)

# === MAIN LOOP ===
if __name__ == "__main__":
    log("Option ML Tracker started.")
    send_telegram("Option ML Tracker started successfully.")

    while True:
        try:
            run_prediction()
        except Exception as e:
            err_msg = f"Error in main loop: {str(e)}"
            log(err_msg)
            send_telegram(err_msg)

        if DEBUG_MODE:
            break  # Exit after 1 loop in debug mode
        time.sleep(60)
