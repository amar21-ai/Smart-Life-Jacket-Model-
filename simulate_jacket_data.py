# save as simulate_jacket_data.py
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime, timedelta, timezone
import random

# ---------- Helpers ----------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def eta_minutes_from_distance_km(dist_km, speed_kmh=20.0):
    if speed_kmh <= 0:
        return 9999.0
    return (dist_km / speed_kmh) * 60.0

def generate_sea_coordinates():
    """Generate random coordinates that fall within Mediterranean Sea zones"""
    sea_zones = [
        # Western Mediterranean
        {'lat': (36.0, 43.0), 'lon': (3.0, 12.0)},
        # Central Mediterranean
        {'lat': (31.0, 40.0), 'lon': (12.0, 20.0)},
        # Eastern Mediterranean
        {'lat': (31.0, 37.0), 'lon': (20.0, 35.0)},
        # Aegean Sea
        {'lat': (35.0, 41.0), 'lon': (23.0, 28.0)}
    ]
    
    zone = random.choice(sea_zones)
    lat = random.uniform(zone['lat'][0], zone['lat'][1])
    lon = random.uniform(zone['lon'][0], zone['lon'][1])
    
    return lat, lon

# ---------- Simulation function ----------
def simulate_dataset(n=3000, seed=42, center_lat=31.2, center_lon=29.9, boat_speed_kmh=20.0):
    """
    Simulate dataset with correlated vitals and sea-only coordinates
    """
    np.random.seed(seed)
    rows = []
    base_time = datetime.now(timezone.utc)
    
    for i in range(n):
        jid = f"jacket_{i+1:04d}"
        ts = base_time - timedelta(seconds=int(np.random.uniform(0, 2*3600)))
        
        # Generate coordinates only in Mediterranean Sea
        lat, lon = generate_sea_coordinates()
        distance = haversine_km(lat, lon, center_lat, center_lon)
        eta = eta_minutes_from_distance_km(distance, speed_kmh=boat_speed_kmh)

        # State first
        state = int(np.random.choice([0, 1], p=[0.85, 0.15]))  # same prior

        # --- Correlated sampling (heart_rate, spo2, humidity) ---
        # Means, stds and correlations tuned by state
        if state == 1:
            means = np.array([120.0, 82.0, 78.0])   # hr, spo2, humidity
            stds  = np.array([18.0, 6.0, 12.0])
            corr  = np.array([
                [1.0, -0.7,  0.5],   # HR negatively correlated with SpO2, positively with humidity
                [-0.7, 1.0, -0.6],   # SpO2 negatively correlated with humidity in critical cases
                [0.5, -0.6,  1.0]
            ])
        else:
            means = np.array([80.0, 98.5, 36.0])
            stds  = np.array([8.0, 1.2, 7.0])
            corr  = np.array([
                [1.0, -0.25, 0.1],
                [-0.25, 1.0, -0.15],
                [0.1, -0.15, 1.0]
            ])

        # Build covariance matrix
        cov = np.outer(stds, stds) * corr

        # Draw a single correlated sample
        try:
            hr_val, spo2_val, hum_val = np.random.multivariate_normal(means, cov)
        except Exception:
            # Fallback: in case cov is not PD (shouldn't usually happen), fall back to independent draws:
            hr_val = np.random.normal(means[0], stds[0])
            spo2_val = np.random.normal(means[1], stds[1])
            hum_val = np.random.normal(means[2], stds[2])

        # Clip & convert to realistic ranges
        heart_rate = int(np.clip(hr_val, 40, 220))
        spo2 = int(np.clip(spo2_val, 60, 100))
        humidity = float(np.clip(hum_val, 5.0, 100.0))

        # Last packet frequency: more frequent (smaller scale) when sinking
        last_packet_scale = 8.0 if state == 1 else 30.0
        last_packet_s = float(np.random.exponential(scale=last_packet_scale))

        # Extra logical enforcement rules (keeps realism and enforces invariants)
        # - If sinking ensure oxygen is in dangerous range or HR is elevated
        if state == 1:
            if spo2 > 92:
                # lower spo2 into a dangerous range but keep some variability
                spo2 = int(np.clip(np.random.normal(82, 6), 65, 92))
            if heart_rate < 90:
                heart_rate = int(np.clip(np.random.normal(110, 18), 80, 200))

        # Enforce inverse relation if HR very high
        if heart_rate > 150 and spo2 > 90:
            spo2 = int(np.clip(spo2 - int((heart_rate - 150)*0.6), 60, 100))

        # Risk flags (adjusted thresholds to match physiology)
        hr_risk = 1 if (heart_rate < 50 or heart_rate > 120) else 0
        spo2_risk = 1 if spo2 < 92 else 0
        humidity_risk = 1 if humidity > 60 else 0
        num_risks = hr_risk + spo2_risk + humidity_risk + state

        # Baseline priority (you can tune weights)
        S_H = 0.0
        if hr_risk: S_H += 35.0
        if spo2_risk: S_H += 40.0
        if state == 1: S_H += 30.0
        S_H = min(100.0, S_H)

        S_D = max(0.0, min(100.0, (humidity - 30.0) * 2.0))
        S_R = max(0.0, min(100.0, (eta - 5.0) * (100.0/55.0)))

        priority = 0.55 * S_H + 0.15 * S_D + 0.30 * S_R

        # Expert overrides (same idea as before)
        if (spo2 < 85 and state == 1) or (heart_rate < 45 and state == 1) or (spo2 < 80):
            priority = max(priority, 95.0)
        if humidity > 90:
            priority = max(priority, 90.0)
        if last_packet_s > 120:
            priority = max(priority, 80.0)

        priority += np.random.normal(loc=0.0, scale=3.0)
        priority = float(np.clip(priority, 0.0, 100.0))

        rows.append({
            'id': jid,
            'ts': ts.isoformat() + 'Z',
            'lat': round(lat,6),
            'lon': round(lon,6),
            'heart_rate': int(heart_rate),
            'spo2': int(spo2),
            'humidity': round(humidity,2),
            'state': state,
            'distance_km': round(distance,3),
            'eta_min': round(eta,2),
            'last_packet_s': round(last_packet_s,2),
            'hr_risk': hr_risk,
            'spo2_risk': spo2_risk,
            'humidity_risk': humidity_risk,
            'num_risks': num_risks,
            'priority': round(priority,2)
        })

    df = pd.DataFrame(rows)
    return df
# ---------- Main ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Simulate jacket dataset')
    parser.add_argument('--n', type=int, default=3000, help='number of rows to simulate')
    parser.add_argument('--out', type=str, default='sim_jacket_data.csv', help='output csv file')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--center-lat', type=float, default=31.2, help='rescue center latitude')
    parser.add_argument('--center-lon', type=float, default=29.9, help='rescue center longitude')
    parser.add_argument('--speed', type=float, default=20.0, help='boat speed km/h for ETA')
    args = parser.parse_args()

    df = simulate_dataset(n=args.n, seed=args.seed, center_lat=args.center_lat,
                          center_lon=args.center_lon, boat_speed_kmh=args.speed)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")
    print(df.describe(include='all').transpose().loc[['heart_rate','spo2','humidity','state','priority']])
