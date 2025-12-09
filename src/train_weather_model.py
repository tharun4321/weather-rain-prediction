import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------- 1. LOAD DATA ----------
def load_data(csv_path="../data/weather_data.csv"):
    """Load weather data from CSV."""
    data = pd.read_csv(csv_path)

    feature_cols = ["temp_c", "humidity", "wind_kph", "pressure_hpa", "cloud_pct"]
    X = data[feature_cols]
    y = data["rain"]   # 1 = rain, 0 = no rain

    return X, y, feature_cols


# ---------- 2. TRAIN MODEL ----------
def train_model(X, y, test_size=0.2, random_state=42):
    """Train a Random Forest classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test


# ---------- 3. EVALUATE MODEL ----------
def evaluate_model(model, X_test, y_test):
    """Print accuracy and classification metrics."""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Model Performance:")
    print(f"  Accuracy: {acc:.4f}")
    print("  Confusion Matrix:")
    print(cm)
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


# ---------- 4. USER INPUT PREDICTION ----------
def predict_from_user(model, feature_cols):
    """Ask user for weather values and predict rain or no rain."""
    print("\nEnter today's weather details to predict rain (0 = No, 1 = Yes):")

    try:
        temp_c = float(input("  Temperature (°C): "))
        humidity = float(input("  Humidity (%): "))
        wind_kph = float(input("  Wind speed (km/h): "))
        pressure_hpa = float(input("  Pressure (hPa): "))
        cloud_pct = float(input("  Cloud cover (%): "))

        new_row = pd.DataFrame(
            [[temp_c, humidity, wind_kph, pressure_hpa, cloud_pct]],
            columns=feature_cols
        )

        pred = model.predict(new_row)[0]
        prob = model.predict_proba(new_row)[0][1]  # probability of rain

        if pred == 1:
            print(f"\n🌧 It is likely to RAIN. (Confidence: {prob*100:.1f}%)")
        else:
            print(f"\n☀ No rain expected. (Rain probability: {prob*100:.1f}%)")

    except ValueError:
        print("❌ Invalid input. Please enter numeric values only.")


# ---------- 5. MAIN ----------
def main():
    X, y, feature_cols = load_data()
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    evaluate_model(model, X_test, y_test)
    predict_from_user(model, feature_cols)


if __name__ == "__main__":
    main()
