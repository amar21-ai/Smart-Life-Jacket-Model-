# 🛟 Smart Life Jacket – Rescue Priority Model

This project provides a **machine learning model and dashboard** to support rescue operations using smart life jackets.  
It processes real-time telemetry (heart rate, SpO₂, humidity, sinking state, location, and communication metrics) and predicts a **priority score (0–100)** to help rescuers identify and respond to the most urgent cases.

## 🚀 Features
- Cleans and preprocesses telemetry data  
- Random Forest model for rescue priority prediction  
- Flask API for real-time scoring  
- Interactive dashboard to visualize users and rescue paths
  
🔑 Tech Stack

Python · scikit-learn · Flask · pandas · Leaflet.js · Bootstrap

## ⚡ Usage
```bash
# clone repo
git clone https://github.com/YOUR_USERNAME/Smart-Life-Jacket-Model.git
cd Smart-Life-Jacket-Model

# install dependencies
pip install -r requirements.txt

# train model
python train_model.py

# run API
python serve_priority.py

