# 🛟 Smart Life Jacket – Rescue Priority System

This project presents a **smart life jacket monitoring and rescue prioritization system**.  
By processing real-time telemetry from life jackets — such as heart rate, blood oxygen (SpO₂), humidity, sinking state, location, and communication metrics — the system predicts a **priority score (0–100)** for each user. This score helps rescuers allocate resources effectively during emergency operations.

---

## 🚀 Features
- **Data Collection & Cleaning**: Ensures valid, consistent telemetry for model input.  
- **Machine Learning Model**: Random Forest Regressor trained to predict rescue priority.  
- **Flask API**: Serves the trained model for real-time scoring.  
- **Interactive Dashboard**: Visualizes users, priority scores, and rescue paths.  

---

## ⚡ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/amar21-ai/Smart-Life-Jacket-Model-.git
cd Smart-Life-Jacket-Model-
