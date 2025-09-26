# train_model_improved.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import sys

CSV_PATH = "sim_jacket_data.csv"
FEATURES = ['heart_rate', 'spo2', 'humidity', 'state', 'distance_km', 'eta_min', 'last_packet_s']
TARGET = 'priority'
RANDOM_STATE = 42

def main():
    df = pd.read_csv(CSV_PATH)
    print("Loaded", len(df), "rows from", CSV_PATH)

    # Basic checks
    if df[FEATURES + [TARGET]].isnull().any().any():
        print("⚠️ Missing values detected. Filling numeric NaNs with column median.")
        for col in FEATURES + [TARGET]:
            if df[col].isnull().any():
                median = df[col].median()
                df[col] = df[col].fillna(median)

    # Optional: group-split by id if present (prevents leakage from same jacket)
    if 'id' in df.columns:
        print("Found 'id' column — performing geakageroup-aware split to avoid l by jacket.")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        groups = df['id']
        train_idx, test_idx = next(gss.split(df, groups=groups))
        X_train = df.loc[train_idx, FEATURES]
        X_test  = df.loc[test_idx, FEATURES]
        y_train = df.loc[train_idx, TARGET]
        y_test  = df.loc[test_idx, TARGET]
    else:
        X = df[FEATURES]
        y = df[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

    print("Train rows:", len(X_train), "Test rows:", len(X_test))

    # Model: Random Forest (use all cores). Use oob_score for an extra check.
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        oob_score=True,
        bootstrap=True
    )

    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Train performance (to detect overfitting)
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    print("✅ Model trained")
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test MAE:  {mae:.3f}")
    print(f"Test R²:   {r2:.3f}")
    print(f"Train RMSE: {train_rmse:.3f}")
    print(f"Train R²:   {train_r2:.3f}")

    if hasattr(model, "oob_score_"):
        print("OOB R²:", round(model.oob_score_, 3))

    # Feature importances (sorted)
    importances = model.feature_importances_
    fi = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)
    print("\nFeature importances:")
    for f, imp in fi:
        print(f"  {f}: {imp:.4f}")

    # Cross-validated score (optional)
    try:
        cv_scores = cross_val_score(model, df[FEATURES], df[TARGET], cv=5, scoring='r2', n_jobs=-1)
        print("\n5-fold CV R²:", np.round(cv_scores, 3), "mean:", np.round(cv_scores.mean(),3))
    except Exception as e:
        print("Could not run cross_val_score:", e)

    # Save model and feature list
    joblib.dump(model, "priority_model.joblib")
    joblib.dump(FEATURES, "priority_model_features.joblib")
    print("\n💾 Saved model as priority_model.joblib and feature list as priority_model_features.joblib")

if __name__ == "__main__":
    main()
