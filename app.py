from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os
 
app = Flask(__name__)
CORS(app)
 
model = joblib.load('zomato_model.pkl')
 
# Encodings match exact LabelEncoder order used during training
weather_map  = {'Cloudy': 0, 'Fog': 1, 'Sandstorms': 2, 'Stormy': 3, 'Sunny': 4, 'Windy': 5}
traffic_map  = {'High': 0, 'Jam': 1, 'Low': 2, 'Medium': 3}
vehicle_map  = {'bicycle': 0, 'electric_scooter': 1, 'motorcycle': 2, 'scooter': 3}
festival_map = {'No': 0, 'Yes': 1}
city_map     = {'Metropolitian': 0, 'Semi-Urban': 1, 'Urban': 2}
order_map    = {'Buffet': 0, 'Drinks': 1, 'Meal': 2, 'Snack': 3}
 
@app.route('/')
def index():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(base_dir, 'Index.html')
 
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
 
    hour = float(data['hour'])
 
    # Convert hour to minutes since midnight (matches training data)
    # e.g. 13:00 → 780 mins, pickup ~10 mins later → 790 mins
    order_mins   = hour * 60       # e.g. 13 → 780
    pickup_mins  = order_mins + 10 # pickup ~10 mins after order
    pickup_delay = 10.0            # median from training data
 
    # Exact feature order matching training:
    # Delivery_person_Age, Delivery_person_Ratings, Weather_conditions,
    # Road_traffic_density, Vehicle_condition, Type_of_order,
    # Type_of_vehicle, multiple_deliveries, Festival, City,
    # distance_km, order_mins, pickup_mins, pickup_delay, order_hour
    features = np.array([[
        float(data['age']),              # 0:  Delivery_person_Age
        float(data['rating']),           # 1:  Delivery_person_Ratings
        weather_map[data['weather']],    # 2:  Weather_conditions
        traffic_map[data['traffic']],    # 3:  Road_traffic_density
        float(data['vehicle_cond']),     # 4:  Vehicle_condition
        order_map[data['order_type']],   # 5:  Type_of_order
        vehicle_map[data['vehicle']],    # 6:  Type_of_vehicle
        int(data['multi']),              # 7:  multiple_deliveries
        festival_map[data['festival']],  # 8:  Festival
        city_map[data['city']],          # 9:  City
        float(data['distance']),         # 10: distance_km
        order_mins,                      # 11: order_mins (mins since midnight)
        pickup_mins,                     # 12: pickup_mins
        pickup_delay,                    # 13: pickup_delay (median = 10.0)
        hour,                            # 14: order_hour
    ]])
 
    prediction = model.predict(features)[0]
    prediction = max(5, round(float(prediction)))  # minimum realistic ETA
    return jsonify({'eta': prediction})
 
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', debug=False, port=port)
