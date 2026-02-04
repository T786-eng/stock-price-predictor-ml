import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import time

# 1. Auto-detect CSV dataset in the script folder
cwd = os.path.dirname(__file__)
preferred = os.path.join(cwd, 'stock_data.csv')
if os.path.exists(preferred):
    file_path = preferred
else:
    csvs = [f for f in os.listdir(cwd) if f.lower().endswith('.csv')]
    if not csvs:
        print("✗ No CSV dataset found in folder. Place a CSV and rerun.")
        raise SystemExit(1)
    file_path = os.path.join(cwd, csvs[0])

start_time = time.time()
df = pd.read_csv(file_path)
load_time = time.time() - start_time
print(f"✓ Loaded: {os.path.basename(file_path)} — shape={df.shape} in {load_time:.2f}s")

# Normalize column names for lookup
cols_lower = {c.lower(): c for c in df.columns}
def col(key):
    return cols_lower.get(key.lower())

# Require Close column
if col('close') is None:
    print("✗ Dataset missing required 'Close' column.")
    raise SystemExit(1)

# Basic cleanup
if col('date'):
    dcol = col('date')
    # parse datetimes defensively (mixed tzs -> coerce to UTC, then drop tz)
    date_series = pd.to_datetime(df[dcol], utc=True, errors='coerce')
    try:
        date_series = date_series.dt.tz_convert(None)
    except Exception:
        try:
            date_series = date_series.dt.tz_localize(None)
        except Exception:
            pass
    df[dcol] = date_series
    df = df.sort_values(by=dcol).ffill()
else:
    df = df.ffill()

# 2. Fast technical indicators (safe guards if columns missing)
start_indicators = time.time()
c_close = col('close')
df['MA10'] = df[c_close].rolling(10).mean()
df['MA50'] = df[c_close].rolling(50).mean()
delta = df[c_close].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['RSI'] = 100 - (100 / (1 + gain / loss))
df['Target'] = df[c_close].shift(-1)
df.dropna(inplace=True)
indicators_time = time.time() - start_indicators
print(f"✓ Indicators calculated in {indicators_time:.2f}s, final shape={df.shape}")

# Sample a subset to speed up training (adjust frac as needed)
df = df.sample(frac=0.1, random_state=42)
print(f"✓ Sampled dataset to {len(df)} rows for faster training")

# 3. Choose features that exist in the dataset
candidate = ['open', 'high', 'low', 'close', 'volume', 'ma10', 'ma50', 'rsi']
available = [cols_lower.get(c) if c in cols_lower else (c.upper() if c.upper() in df.columns else None) for c in candidate]
features = [f for f in available if f and f in df.columns]
if not features:
    print('✗ No usable features found in dataset.')
    raise SystemExit(1)

X = df[features]
y = df['Target']

# 4. Split, train and evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
start_train = time.time()
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
train_time = time.time() - start_train
print(f"✓ Model trained in {train_time:.2f}s")
start_pred = time.time()
preds = model.predict(X_test)
pred_time = time.time() - start_pred
print(f"✓ Predictions made in {pred_time:.2f}s")

print('--- Results ---')
print(f"R2 Score: {r2_score(y_test, preds):.4f}")
print(f"MAE: ${mean_absolute_error(y_test, preds):.2f}")

# 5. Plot results
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual')
plt.plot(preds, label='Predicted', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show() 