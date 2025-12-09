import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

API_KEY="db5232745a4f7e7bee3738f58828b267"



# --------- 1. LOAD & TRAIN MODEL ---------
def train_model(csv_path="../data/weather_data.csv"):
    data = pd.read_csv(csv_path)
    feature_cols = ["temp_c", "humidity", "wind_kph", "pressure_hpa", "cloud_pct"]

    X = data[feature_cols]
    y = data["rain"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"Model trained. Accuracy on test set: {score:.4f}")
    return model, feature_cols


# --------- 2. FETCH LIVE WEATHER FROM API ---------
def get_weather_from_city(city_name, api_key=API_KEY):
    """
    Call OpenWeatherMap current weather API and return
    our 5 features as a dict.
    """
    url = (
        f"http://api.openweathermap.org/data/2.5/weather"
        f"?q={city_name}&appid={api_key}&units=metric"
    )

    resp = requests.get(url)
    if resp.status_code != 200:
        print("❌ Failed to fetch data. Check city name or API key.")
        print("Response:", resp.text)
        return None

    data = resp.json()

    temp_c = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    pressure_hpa = data["main"]["pressure"]
    wind_ms = data["wind"].get("speed", 0.0)  # meters/sec
    wind_kph = wind_ms * 3.6
    cloud_pct = data.get("clouds", {}).get("all", 0)

    print("\nLive Weather from API:")
    print(f"  Temp (°C)     : {temp_c}")
    print(f"  Humidity (%)  : {humidity}")
    print(f"  Wind (km/h)   : {wind_kph:.1f}")
    print(f"  Pressure (hPa): {pressure_hpa}")
    print(f"  Cloud cover % : {cloud_pct}")

    return {
        "temp_c": temp_c,
        "humidity": humidity,
        "wind_kph": wind_kph,
        "pressure_hpa": pressure_hpa,
        "cloud_pct": cloud_pct,
    }


# --------- 3. PREDICT USING LIVE DATA ---------
def predict_rain_from_city(model, feature_cols, city_name):
    features = get_weather_from_city(city_name)
    if features is None:
        return

    df = pd.DataFrame([[features[col] for col in feature_cols]], columns=feature_cols)

    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]  # prob of rain

    if pred == 1:
        print(f"\n🌧 It is likely to RAIN in {city_name}. (Confidence: {proba*100:.1f}%)")
    else:
        print(
            f"\n☀ No rain expected in {city_name}. "
            f"(Rain probability: {proba*100:.1f}%)"
        )


def main():
    city = input("Enter city name for live rain prediction: ")
    model, feature_cols = train_model()
    predict_rain_from_city(model, feature_cols, city)


if __name__ == "__main__":
    main()
