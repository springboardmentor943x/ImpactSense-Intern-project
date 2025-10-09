import os
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load preprocessed data with one-hot encoded columns
df = pd.read_csv("data/train_data.csv")

# Select feature columns (original numeric + one-hot encoded categorical)
features = [
    "Latitude",
    "Longitude",
    "Depth",
    "Magnitude",
    "Root Mean Square",
    "Magnitude Type_MD",
    "Magnitude Type_MH",
    "Magnitude Type_ML",
    "Magnitude Type_MS",
    "Magnitude Type_MW",
    "Magnitude Type_MWB",
    "Magnitude Type_MWC",
    "Magnitude Type_MWR",
    "Magnitude Type_MWW",
    "Status_Reviewed"
]

# Prepare feature matrix and target vector
X = df[features]
y = df["Damage_Potential"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train LightGBM regressor
model = lgb.LGBMRegressor(
    n_estimators=600,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.3f}, R²: {r2:.3f}")

# Save model and feature list
os.makedirs("models", exist_ok=True)
joblib.dump(
    {"model": model, "features": features},
    "models/lgb_damage_model.pkl"
)
print("Saved model and feature list")
