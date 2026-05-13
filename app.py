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
    return send_from_directory('.', 'zomato-eta-vibrant(1).html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[
        float(data['age']),
        float(data['rating']),
        weather_map[data['weather']],
        traffic_map[data['traffic']],
        float(data['vehicle_cond']),
        order_map[data['order_type']],
        vehicle_map[data['vehicle']],
        int(data['multi']),
        festival_map[data['festival']],
        city_map[data['city']],
        float(data['distance']),
        float(data['hour']) * 60,
        float(data['hour']) * 60 + 10,
        10.0,
        float(data['hour']),
    ]])
    prediction = model.predict(features)[0]
    return jsonify({'eta': round(float(prediction))})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', debug=False, port=port)