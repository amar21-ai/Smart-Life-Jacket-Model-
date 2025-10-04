# serve_priority.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from simulate_jacket_data import haversine_km, eta_minutes_from_distance_km

app = Flask(__name__)
CORS(app)  # allow cross-origin requests (useful for browser dashboards)

# Load model once at startup
MODEL_PATH = "priority_model.joblib"
model = joblib.load(MODEL_PATH)

# The features the model was trained on, keep order consistent
FEATURES = ['heart_rate','spo2','humidity','state','distance_km','eta_min','last_packet_s']

def score_to_class(score):
    if score >= 75: return "High"
    if score >= 45: return "Medium"
    return "Low"

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok","model":MODEL_PATH}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts either a JSON object (single sample) or a JSON array (batch).
    Each input may include lat/lon OR distance_km & eta_min already computed.
    Required numeric fields ideally: heart_rate, spo2, humidity, state
    Optional: lat, lon, last_packet_s
    """
    payload = request.get_json(force=True)
    if payload is None:
        return jsonify({"error":"no json payload"}), 400

    # Normalize into DataFrame (single dict -> list)
    if isinstance(payload, dict):
        df = pd.json_normalize([payload])
    elif isinstance(payload, list):
        df = pd.json_normalize(payload)
    else:
        return jsonify({"error":"invalid payload format, must be dict or list"}), 400

    # compute features (adds distance_km & eta_min if lat/lon present)
    try:
        # Calculate distance and ETA if lat/lon are provided
        CENTER_LAT, CENTER_LON = 31.2, 29.9  # Same center coordinates as in simulate_jacket_data.py
        
        if 'lat' in df.columns and 'lon' in df.columns:
            for i, row in df.iterrows():
                distance = haversine_km(row['lat'], row['lon'], CENTER_LAT, CENTER_LON)
                eta = eta_minutes_from_distance_km(distance)
                df.at[i, 'distance_km'] = distance
                df.at[i, 'eta_min'] = eta
    except Exception as e:
        return jsonify({"error":"feature computation failed","detail": str(e)}), 500

    # Ensure feature columns exist and are in the same order
    for c in FEATURES:
        if c not in df.columns:
            df[c] = 0
    X = df[FEATURES].fillna(0)

    # Predict
    preds = model.predict(X)  # pass DataFrame to avoid feature-name warning

    # build response
    out = []
    for i, score in enumerate(preds):
        score = float(max(0.0, min(100.0, score)))
        cls = score_to_class(score)
        row = df.iloc[i].to_dict()
        out.append({
            "id": row.get("id", None),
            "priority_score": round(score, 2),
            "priority_class": cls,
            # echo some input fields for convenience
            "heart_rate": int(row.get("heart_rate", 0)),
            "spo2": int(row.get("spo2", 0)),
            "humidity": float(row.get("humidity", 0)),
            "state": int(row.get("state", 0)),
            "distance_km": round(float(row.get("distance_km", 0)), 3),
            "eta_min": round(float(row.get("eta_min", 0)), 2),
            "last_packet_s": float(row.get("last_packet_s", 0)),
            # Include coordinates in response
            "lat": float(row.get("lat", 0)) if "lat" in row else None,
            "lon": float(row.get("lon", 0)) if "lon" in row else None
        })

    # if input was single dict, return single object
    if isinstance(payload, dict):
        return jsonify(out[0]), 200
    return jsonify(out), 200

if __name__ == "__main__":
    # Using Waitress WSGI server instead of Flask's development server
    from waitress import serve
    print("Starting production server on http://0.0.0.0:5000")
    serve(app, host="0.0.0.0", port=5000)