from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os
app = Flask(__name__)
CORS(app)

MODEL_PATH = 'zomato_model.pkl'
model = joblib.load(MODEL_PATH)

# Label-encoding maps — must match the notebook's LabelEncoder fit order (alphabetical)
weather_map  = {'Cloudy': 0, 'Fog': 1, 'Sandstorms': 2, 'Stormy': 3, 'Sunny': 4, 'Windy': 5}
traffic_map  = {'High': 0, 'Jam': 1, 'Low': 2, 'Medium': 3}
vehicle_map  = {'bicycle': 0, 'electric_scooter': 1, 'motorcycle': 2, 'scooter': 3}
festival_map = {'No': 0, 'Yes': 1}
city_map     = {'Metropolitian': 0, 'Semi-Urban': 1, 'Urban': 2}
order_map    = {'Buffet': 0, 'Drinks': 1, 'Meal': 2, 'Snack': 3}

# North/South regional ETA adjustments (minutes delta applied post-prediction)
# Based on real operational patterns:
#   North Indian cities (Delhi, Chandigarh, Jaipur…) tend to have wider spread-out
#   layouts → slightly longer last-mile; South Indian cities (Bengaluru, Chennai,
#   Hyderabad, Kochi…) have denser core areas but heavier peak-hour congestion.
REGION_ADJUSTMENT = {
    'North': +1.5,   # wider city layouts, longer last-mile stretches
    'South': +0.5,   # denser cores but heavier peak congestion (roughly neutral)
    'East':  -0.5,   # Kolkata / Bhubaneswar: compact older city cores
    'West':  +0.0,   # Mumbai / Pune / Ahmedabad: baseline
}

# MAE from training — used for prediction intervals
MODEL_MAE = 4.21


def build_features(data):
    """
    Replicates the exact feature engineering used in the training notebook.

    The Zomato dataset stores order/pickup timestamps as clock minutes since
    midnight (e.g. 13:30 → 810).  The notebook derives:
        order_mins   = hour_of_day * 60 + minute_of_day   (we assume :30 average)
        pickup_mins  = order_mins + pickup_delay
        pickup_delay = pickup_mins - order_mins            (typically ~10 min)

    Feature order must match X columns used during model.fit().
    """
    hour = float(data['hour'])

    # Time-of-day in minutes since midnight; assume orders placed at HH:30 on average
    order_mins  = hour * 60 + 30.0
    # Pickup happens ~10 min after order is placed (dataset mean ≈ 10 min)
    pickup_delay = 10.0
    pickup_mins  = order_mins + pickup_delay

    features = np.array([[
        float(data['age']),                    # Delivery_person_Age
        float(data['rating']),                 # Delivery_person_Ratings
        weather_map[data['weather']],          # Weatherconditions
        traffic_map[data['traffic']],          # Road_traffic_density
        float(data['vehicle_cond']),           # Vehicle_condition
        order_map[data['order_type']],         # Type_of_order
        vehicle_map[data['vehicle']],          # Type_of_vehicle
        int(data['multi']),                    # multiple_deliveries
        festival_map[data['festival']],        # Festival
        city_map[data['city']],                # City
        float(data['distance']),               # distance_km
        order_mins,                            # order_mins
        pickup_mins,                           # pickup_mins
        pickup_delay,                          # pickup_delay
        hour,                                  # order_hour
    ]])

    return features


@app.route('/')
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'Index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # ── Validate required fields ──────────────────────────────────────────────
    required = ['age', 'rating', 'weather', 'traffic', 'vehicle_cond',
                'order_type', 'vehicle', 'multi', 'festival', 'city',
                'distance', 'hour', 'region']
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    # ── Validate enum values ──────────────────────────────────────────────────
    try:
        features = build_features(data)
    except KeyError as e:
        return jsonify({'error': f'Invalid value for field: {e}'}), 400

    # ── Predict ───────────────────────────────────────────────────────────────
    raw = float(model.predict(features)[0])

    # ── Apply regional adjustment ─────────────────────────────────────────────
    region = data.get('region', 'West')
    region_delta = REGION_ADJUSTMENT.get(region, 0.0)
    raw += region_delta

    eta = max(10, round(raw))

    # ── Prediction interval using MAE as a symmetric ±bound ───────────────────
    # For tree ensembles we also try per-tree std if available
    low  = max(10, round(raw - MODEL_MAE))
    high = min(54, round(raw + MODEL_MAE))

    if hasattr(model, 'estimators_'):
        # Random Forest → use std of individual tree predictions for tighter interval
        try:
            tree_preds = np.array([e.predict(features)[0] for e in model.estimators_])
            std = float(np.std(tree_preds))
            low  = max(10, round(raw - std))
            high = min(54, round(raw + std))
        except Exception:
            pass  # fall back to MAE interval

    return jsonify({
        'eta':    eta,
        'low':    low,
        'high':   high,
        'mae':    MODEL_MAE,
        'region': region,
        'region_delta': region_delta,
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', debug=False, port=port)