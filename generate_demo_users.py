import random
import json
from simulate_jacket_data import haversine_km, eta_minutes_from_distance_km

def generate_sea_coordinates():
    """Generate random coordinates within a safe radius of a point in Mediterranean Sea"""
    # Center point in decimal degrees
    CENTER_LAT = 33.64544  # 33°38'43.6"N
    CENTER_LON = 30.00117  # 30°00'04.2"E
    
    # Generate point within 20km radius to ensure they're in the sea
    # but not too far for rescue operations
    radius_km = random.uniform(2, 20)  # Between 2-20km from center
    bearing = random.uniform(0, 360)  # Random direction
    
    # Convert bearing and distance to lat/lon difference
    # Rough approximation (sufficient for small distances)
    lat_km = 111  # 1 degree latitude = ~111km
    lon_km = 111 * cos(radians(CENTER_LAT))  # 1 degree longitude = ~111km * cos(lat)
    
    delta_lat = (radius_km * cos(radians(bearing))) / lat_km
    delta_lon = (radius_km * sin(radians(bearing))) / lon_km
    
    lat = CENTER_LAT + delta_lat
    lon = CENTER_LON + delta_lon
    
    return round(lat, 5), round(lon, 5)

def generate_demo_users(n_users=100):
    users = []
    
    for i in range(n_users):
        lat, lon = generate_sea_coordinates()
        
        # Randomly decide if user is in distress
        is_distress = random.random() < 0.15  # 15% chance of distress
        
        if is_distress:
            heart_rate = random.randint(120, 160)
            spo2 = random.randint(75, 85)
            humidity = random.randint(75, 95)
            state = 1
        else:
            heart_rate = random.randint(60, 100)
            spo2 = random.randint(95, 100)
            humidity = random.randint(30, 60)
            state = 0
        
        user = {
            "id": f"user_{i+1:03d}",
            "lat": lat,
            "lon": lon,
            "heart_rate": heart_rate,
            "spo2": spo2,
            "humidity": humidity,
            "state": state,
            "last_packet_s": random.randint(5, 60)
        }
        users.append(user)
    
    return users

if __name__ == "__main__":
    from math import cos, sin, radians  # Add these imports at the top
    
    # Generate users
    demo_users = generate_demo_users(100)
    
    # Save to JSON file
    output_file = "demo_users.json"
    with open(output_file, 'w') as f:
        json.dump(demo_users, f, indent=2)
    
    print(f"Generated {len(demo_users)} users and saved to {output_file}")