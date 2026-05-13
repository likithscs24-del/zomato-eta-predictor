# Zomato Delivery ETA Predictor

A machine learning web app that predicts food delivery time.

## Run Locally
pip install -r requirements.txt
python app.py

## API - POST /predict
{
  "age": 25, "rating": 4.5, "weather": "Sunny",
  "traffic": "Medium", "vehicle_cond": 2, "order_type": "Meal",
  "vehicle": "motorcycle", "multi": 0, "festival": "No",
  "city": "Urban", "distance": 5.2, "hour": 13
}