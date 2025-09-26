# test_model.py
import joblib
import pandas as pd
from simulate_jacket_data import haversine_km, eta_minutes_from_distance_km

# thresholds for class buckets
def score_to_class(score):
    if score >= 80: return "High"
    if score >= 50: return "Medium"
    return "Low"

# load model
model = joblib.load("priority_model.joblib")
FEATURES = ['heart_rate','spo2','humidity','state','distance_km','eta_min','last_packet_s']

# Example single sample (edit values for quick tests)
sample1 = {
    "id": "demo_A",
    "lat": 31.25, "lon": 29.85,
    "heart_rate": 140,
    "spo2": 82,
    "humidity": 85,
    "state": 1,
    "last_packet_s": 5
}

# Example batch (list of dicts)
batch = [
    sample1,
    {"id":"demo_B","lat":31.21,"lon":29.91,"heart_rate":80,"spo2":96,"humidity":45,"state":0,"last_packet_s":5},
    {"id":"demo_C","lat":31.40,"lon":30.10,"heart_rate":65,"spo2":88,"humidity":70,"state":0,"last_packet_s":50},
    {"id":"demo_D","lat":31.10,"lon":29.50,"heart_rate":50,"spo2":78,"humidity":92,"state":1,"last_packet_s":130}
]

def predict_from_dicts(dicts):
    # normalize to DataFrame
    df = pd.json_normalize(dicts)
    
    # Calculate distance and ETA features
    CENTER_LAT, CENTER_LON = 31.2, 29.9  # Same center coordinates as in simulate_jacket_data.py
    for i, row in df.iterrows():
        distance = haversine_km(row['lat'], row['lon'], CENTER_LAT, CENTER_LON)
        eta = eta_minutes_from_distance_km(distance)
        df.at[i, 'distance_km'] = distance
        df.at[i, 'eta_min'] = eta
    
    # Select features for prediction
    X = df[FEATURES].fillna(0)
    preds = model.predict(X)
    
    outputs = []
    for i, row in df.iterrows():
        score = float(max(0.0, min(100.0, preds[i])))
        cls = score_to_class(score)
        outputs.append({
            "id": row.get("id", f"row_{i}"),
            "heart_rate": int(row['heart_rate']),
            "spo2": int(row['spo2']),
            "humidity": float(row['humidity']),
            "state": int(row['state']),
            "distance_km": round(float(row['distance_km']),3),
            "eta_min": round(float(row['eta_min']),2),
            "last_packet_s": float(row['last_packet_s']),
            "predicted_priority": round(score,2),
            "priority_class": cls
        })
    return outputs

if __name__ == "__main__":
    out = predict_from_dicts(batch)
    import json
    print(json.dumps(out, indent=2))
