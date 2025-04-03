from flask import Flask, request, jsonify
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
import pytz

app = Flask(__name__)

# API Configuration
API_KEY = "68902669ec8942fea23142156250204"
BASE_URL = "http://api.weatherapi.com/v1/current.json"

# Function to Fetch Current Weather Data
def get_current_weather(city):
    try:
        url = f"{BASE_URL}?key={API_KEY}&q={city}&aqi=yes"
        response = requests.get(url)
        if response.status_code != 200:
            return None

        data = response.json()
        return {
            'city': data['location']['name'],
            'country': data['location']['country'],
            'current_temp': round(data['current']['temp_c']),
            'feels_like': round(data['current']['feelslike_c']),
            'temp_min': round(data['current']['temp_c'] - 2),
            'temp_max': round(data['current']['temp_c'] + 2),
            'humidity': round(data['current']['humidity']),
            'description': data['current']['condition']['text'],
            'wind_gust_dir': data['current']['wind_degree'],
            'pressure': data['current']['pressure_mb'],
            'Wind_Gust_Speed': data['current']['wind_kph']
        }
    except:
        return None

# Read Historical Data
def read_historical_data(filename):
    return pd.read_csv(filename).dropna().drop_duplicates()

# Prepare Data for Training
def prepare_data(data):
    le = LabelEncoder()
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])
    X = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
    y = data['RainTomorrow']
    return X, y, le

# Train Rain Prediction Model
def train_rain_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Train Regression Model
def train_regression_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Predict Future Values
def predict_future(model, current_value):
    predictions = [current_value]
    for _ in range(5):
        next_value = model.predict(np.array([[predictions[-1]]]))[0]
        predictions.append(next_value)
    return predictions[1:]

# Prepare Regression Data
def prepare_regression_data(data, feature):
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i+1])
    return np.array(X).reshape(-1, 1), np.array(y)


@app.route('/predict', methods=['GET'])
def predict_weather():
    city = request.args.get('city')
    if not city:
        return jsonify({"error": "City parameter is required"}), 400
    
    current_weather = get_current_weather(city)
    if not current_weather:
        return jsonify({"error": "Unable to fetch weather data"}), 500
    
    historical_data = read_historical_data("Historical_data.csv")
    X, y, le = prepare_data(historical_data)
    rain_model = train_rain_model(X, y)
    
    # Prepare Current Weather Data
    current_data = pd.DataFrame([{
        "MinTemp": current_weather['temp_min'],
        "MaxTemp": current_weather['temp_max'],
        "WindGustDir": 0,  # Placeholder
        "WindGustSpeed": current_weather['Wind_Gust_Speed'],
        "Humidity": current_weather['humidity'],
        "Pressure": current_weather['pressure'],
        "Temp": current_weather['current_temp']
    }])
    

    
    # Rain Prediction
    rain_prediction = rain_model.predict(current_data)[0]
    
    # Train Regression Models
    temp_model = train_regression_model(*prepare_regression_data(historical_data, "Temp"))
    hum_model = train_regression_model(*prepare_regression_data(historical_data, "Humidity"))
    
    future_temp = predict_future(temp_model, current_weather['temp_max'])
    future_humidity = predict_future(hum_model, current_weather['humidity'])
    
    now = datetime.now(pytz.timezone("Asia/Kolkata"))
    future_times = [(now + timedelta(hours=i+1)).strftime("%H:%M") for i in range(5)]
    
    prediction_result = {
        "city": current_weather['city'],
        "country": current_weather['country'],
        "current_temp": current_weather['current_temp'],
        "feels_like": current_weather['feels_like'],
        "humidity": current_weather['humidity'],
        "weather": current_weather['description'],
        "rain_prediction": "Yes" if rain_prediction else "No",
        "future_predictions": {
            "times": future_times,
            "temperature": [round(temp, 1) for temp in future_temp],
            "humidity": [round(hum, 1) for hum in future_humidity]
        }}
    
    
    return jsonify(prediction_result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)