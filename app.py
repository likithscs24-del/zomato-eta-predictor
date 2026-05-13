from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

model = joblib.load('zomato_model.pkl')

weather_map  = {'Cloudy':0, 'Fog':1, 'Sandstorms':2, 'Stormy':3, 'Sunny':4, 'Windy':5}
traffic_map  = {'High':0, 'Jam':1, 'Low':2, 'Medium':3}
vehicle_map  = {'bicycle':0, 'electric_scooter':1, 'motorcycle':2, 'scooter':3}
festival_map = {'No':0, 'Yes':1}
city_map     = {'Metropolitian':0, 'Semi-Urban':1, 'Urban':2}
order_map    = {'Buffet':0, 'Drinks':1, 'Meal':2, 'Snack':3}

@app.route('/')
def index():
    return send_from_directory('.', 'Index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[
        float(data['age']),              # Delivery_person_Age
        float(data['rating']),           # Delivery_person_Ratings
        weather_map[data['weather']],    # Weather_conditions
        traffic_map[data['traffic']],    # Road_traffic_density
        float(data['vehicle_cond']),     # Vehicle_condition
        order_map[data['order_type']],   # Type_of_order
        vehicle_map[data['vehicle']],    # Type_of_vehicle
        int(data['multi']),              # multiple_deliveries
        festival_map[data['festival']],  # Festival
        city_map[data['city']],          # City
        float(data['distance']),         # distance_km
        float(data['hour']),             # order_hour (just the hour, not × 60)
        float(data['hour']) + 0.17,      # pickup time (order_hour + ~10 min in decimal)
        10.0,                            # pickup_delay (median)
        float(data['hour']),             # order_hour again
    ]])
    prediction = model.predict(features)[0]
    prediction = max(1, round(float(prediction)))  # prevent negative values
    return jsonify({'eta': prediction})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', debug=False, port=port)