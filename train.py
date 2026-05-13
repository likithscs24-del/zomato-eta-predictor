"""
Zomato Delivery ETA — Model Training Script
============================================
Reproduces the exact feature engineering from the original notebook so that
app.py predictions match what the model was trained on.

Run:
    python train.py

Output:
    zomato_model.pkl   ← upload this to Google Drive
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv('Zomato_Dataset.csv')
print(f"  Raw shape: {df.shape}")

# ── 2. Drop rows with nulls in critical columns ───────────────────────────────
critical = [
    'Delivery_person_Age', 'Delivery_person_Ratings',
    'Weather_conditions', 'Road_traffic_density',
    'multiple_deliveries', 'Festival', 'City',
    'Time_Orderd', 'Time_Order_picked',
]
df = df.dropna(subset=critical).copy()
print(f"  After dropping nulls: {df.shape}")

# ── 3. Haversine distance (km) ────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

df['distance_km'] = haversine(
    df['Restaurant_latitude'],  df['Restaurant_longitude'],
    df['Delivery_location_latitude'], df['Delivery_location_longitude'],
)

# ── 4. Time features ──────────────────────────────────────────────────────────
def time_to_mins(t):
    """Convert 'HH:MM' string to minutes since midnight."""
    try:
        h, m = str(t).strip().split(':')
        return int(h) * 60 + int(m)
    except Exception:
        return np.nan

df['order_mins']  = df['Time_Orderd'].apply(time_to_mins)
df['pickup_mins'] = df['Time_Order_picked'].apply(time_to_mins)

# Handle midnight rollover (pickup next day)
mask = df['pickup_mins'] < df['order_mins']
df.loc[mask, 'pickup_mins'] += 24 * 60

df['pickup_delay'] = df['pickup_mins'] - df['order_mins']
df['order_hour']   = df['order_mins'] // 60

# Drop rows where time parsing failed or pickup_delay is unreasonable
df = df.dropna(subset=['order_mins', 'pickup_mins'])
df = df[(df['pickup_delay'] >= 0) & (df['pickup_delay'] <= 60)]
print(f"  After time cleaning: {df.shape}")

# ── 5. Label encode categoricals ──────────────────────────────────────────────
# Using manual maps so app.py stays in sync (alphabetical order = LabelEncoder default)
weather_map  = {'Cloudy': 0, 'Fog': 1, 'Sandstorms': 2, 'Stormy': 3, 'Sunny': 4, 'Windy': 5}
traffic_map  = {'High': 0, 'Jam': 1, 'Low': 2, 'Medium': 3}
vehicle_map  = {'bicycle': 0, 'electric_scooter': 1, 'motorcycle': 2, 'scooter': 3}
festival_map = {'No': 0, 'Yes': 1}
city_map     = {'Metropolitian': 0, 'Semi-Urban': 1, 'Urban': 2}
order_map    = {'Buffet': 0, 'Drinks': 1, 'Meal': 2, 'Snack': 3}

df['Weather_conditions']   = df['Weather_conditions'].map(weather_map)
df['Road_traffic_density'] = df['Road_traffic_density'].map(traffic_map)
df['Type_of_vehicle']      = df['Type_of_vehicle'].map(vehicle_map)
df['Festival']             = df['Festival'].map(festival_map)
df['City']                 = df['City'].map(city_map)
df['Type_of_order']        = df['Type_of_order'].map(order_map)

# ── 6. Build feature matrix ───────────────────────────────────────────────────
# Column order MUST match app.py's build_features() exactly
FEATURES = [
    'Delivery_person_Age',      # age
    'Delivery_person_Ratings',  # rating
    'Weather_conditions',       # weather
    'Road_traffic_density',     # traffic
    'Vehicle_condition',        # vehicle_cond
    'Type_of_order',            # order_type
    'Type_of_vehicle',          # vehicle
    'multiple_deliveries',      # multi
    'Festival',                 # festival
    'City',                     # city
    'distance_km',              # distance
    'order_mins',               # order_mins
    'pickup_mins',              # pickup_mins
    'pickup_delay',             # pickup_delay
    'order_hour',               # hour
]
TARGET = 'Time_taken (min)'

df = df.dropna(subset=FEATURES + [TARGET])
X = df[FEATURES]
y = df[TARGET]
print(f"  Final training shape: {X.shape}")
print(f"  Target range: {y.min()}–{y.max()} min, mean={y.mean():.1f}")

# ── 7. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 8. Train XGBoost ──────────────────────────────────────────────────────────
print("\nTraining XGBoost model...")
model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ── 9. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f"\n── Results ──────────────────────────")
print(f"  MAE : {mae:.2f} min")
print(f"  R²  : {r2:.3f}")
print(f"  Max error: {abs(y_test.values - y_pred).max():.1f} min")

# ── 10. Feature importance ────────────────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(f"\n── Top features ─────────────────────")
for feat, imp in importances.head(10).items():
    print(f"  {feat:<30} {imp:.4f}")

# ── 11. Save model ────────────────────────────────────────────────────────────
OUT_PATH = 'zomato_model.pkl'
joblib.dump(model, OUT_PATH)
size_mb = os.path.getsize(OUT_PATH) / 1e6
print(f"\n✅ Model saved → {OUT_PATH}  ({size_mb:.1f} MB)")
print("   Upload this file to Google Drive, then copy the File ID into app.py")
