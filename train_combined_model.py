import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

def clean_column_names(df):
    df.columns = [col.encode('ascii', 'ignore').decode().strip() for col in df.columns]
    return df

def preprocess_data(df):
    df = clean_column_names(df)

    required_cols = [
        'Strike Price', 'Open', 'High', 'Low', 'Close', 'LTP', 'Settle Price',
        'No. of contracts', 'Turnover * in', 'Premium Turnover ** in',
        'Open Int', 'Change in OI', 'Underlying Value'
    ]

    col_map = {}
    for col in required_cols:
        match = [c for c in df.columns if c.startswith(col)]
        if match:
            col_map[col] = match[0]
        else:
            raise ValueError(f"Required column not found: {col}")

    df = df[list(col_map.values())]
    df.columns = list(col_map.keys())

    df = df.replace("-", np.nan).dropna()
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    X = df.drop('Close', axis=1)
    y = df['Close']
    return X, y

def train_combined_model():
    if not os.path.exists("combined_data.parquet"):
        print("combined_data.csv file not found!")
        return

    print("Loading combined_data.csv ...")
    df = pd.read_csv("combined_data.csv")

    try:
        X, y = preprocess_data(df)
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return

    print(f"Total records: {len(X)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print("Training XGBoost model ...")
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    print(f"Validation MAE: {mae:.4f}")

    os.makedirs("models", exist_ok=True)
    print("Saving model and scaler ...")
    joblib.dump(model, "models/intraday_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("✅ Model and scaler saved to 'models/' and ready for prediction.")

if __name__ == "__main__":
    train_combined_model()
