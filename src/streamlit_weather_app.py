import pandas as pd
import requests
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --------- API KEY (use secrets/env if possible) ---------
API_KEY = st.secrets["OPENWEATHER_API_KEY"]



# --------- MODEL SETUP ---------
@st.cache_data
def load_data(csv_path="../data/weather_data.csv"):
    data = pd.read_csv(csv_path)
    return data


@st.cache_resource
def train_model(data):
    feature_cols = ["temp_c", "humidity", "wind_kph", "pressure_hpa", "cloud_pct"]

    X = data[feature_cols]
    y = data["rain"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    # feature importances for chart
    importances = pd.Series(model.feature_importances_, index=feature_cols)

    return model, feature_cols, acc, importances


def get_weather_from_city(city_name, api_key=API_KEY):
    if api_key in (None, "", "YOUR_API_KEY_HERE"):
        st.error("API key not set. Configure OPENWEATHER_API_KEY in Streamlit secrets.")
        return None

    url = (
        f"http://api.openweathermap.org/data/2.5/weather"
        f"?q={city_name}&appid={api_key}&units=metric"
    )
    resp = requests.get(url)
    if resp.status_code != 200:
        st.error(f"Failed to fetch data for '{city_name}'. Response: {resp.text}")
        return None

    data = resp.json()
    temp_c = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    pressure_hpa = data["main"]["pressure"]
    wind_ms = data["wind"].get("speed", 0.0)
    wind_kph = wind_ms * 3.6
    cloud_pct = data.get("clouds", {}).get("all", 0)

    return {
        "temp_c": temp_c,
        "humidity": humidity,
        "wind_kph": wind_kph,
        "pressure_hpa": pressure_hpa,
        "cloud_pct": cloud_pct,
    }


# --------- STREAMLIT UI ---------
def main():
    st.title("🌦 Rain Prediction App (ML + Live API)")
    st.write(
        "Predict whether it will rain based on weather conditions. "
        "You can enter values manually or fetch live data from OpenWeather."
    )

    data = load_data()
    model, feature_cols, acc, importances = train_model(data)

    # ----- SIDEBAR -----
    with st.sidebar:
        st.header("Prediction Mode")
        mode = st.radio("Choose input method:", ["Manual Input", "Live City (API)"])

        st.markdown(f"**Model test accuracy:** `{acc:.2%}`")
        st.markdown("---")
        st.markdown("**Dataset preview:**")
        st.dataframe(data.head())

        st.markdown("---")
        st.markdown("**Feature importance:**")
        st.bar_chart(importances)

    # ----- MAIN AREA -----
    st.subheader("Weather-based Rain Prediction")

    if mode == "Manual Input":
        st.markdown("### Enter weather values manually")

        col1, col2 = st.columns(2)
        with col1:
            temp_c = st.slider("Temperature (°C)", -10.0, 50.0, 25.0)
            humidity = st.slider("Humidity (%)", 0, 100, 70)
            cloud_pct = st.slider("Cloud cover (%)", 0, 100, 50)
        with col2:
            wind_kph = st.slider("Wind speed (km/h)", 0.0, 150.0, 10.0)
            pressure_hpa = st.slider("Pressure (hPa)", 950, 1050, 1010)

        if st.button("Predict Rain"):
            df = pd.DataFrame(
                [[temp_c, humidity, wind_kph, pressure_hpa, cloud_pct]],
                columns=feature_cols,
            )
            pred = model.predict(df)[0]
            proba = model.predict_proba(df)[0][1]

            if pred == 1:
                st.error(f"🌧 Rain likely! (Confidence: {proba*100:.1f}%)")
            else:
                st.success(f"☀ No rain expected. (Rain probability: {proba*100:.1f}%)")

    else:
        st.markdown("### Use live data from OpenWeatherMap")
        city = st.text_input("Enter city name", value="Hyderabad")

        if st.button("Fetch & Predict"):
            features = get_weather_from_city(city)
            if features is not None:
                st.write("Live weather data:", features)

                df = pd.DataFrame(
                    [[features[col] for col in feature_cols]],
                    columns=feature_cols,
                )
                pred = model.predict(df)[0]
                proba = model.predict_proba(df)[0][1]

                if pred == 1:
                    st.error(f"🌧 Rain likely in {city}! (Confidence: {proba*100:.1f}%)")
                else:
                    st.success(
                        f"☀ No rain expected in {city}. "
                        f"(Rain probability: {proba*100:.1f}%)"
                    )

    # ----- EXTRA VISUALS -----
    st.markdown("---")
    st.markdown("### Data Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Rain vs No-Rain count**")
        rain_counts = data["rain"].value_counts().rename(index={0: "No rain", 1: "Rain"})
        st.bar_chart(rain_counts)

    with col2:
        st.markdown("**Humidity distribution**")
        st.line_chart(data["humidity"])

    st.caption("Toy dataset for demo purposes. In a real project, use a large historical dataset.")


if __name__ == "__main__":
    main()
