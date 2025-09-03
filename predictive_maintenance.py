"""
Predictive Maintenance Project
Author: Your Name
Description:
  This script trains a machine learning model (XGBoost) to predict
  machine failures using synthetic sensor data (temperature, vibration, etc.).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# ----------------------------
# Step 1: Generate Synthetic Dataset
# ----------------------------
np.random.seed(42)

data = pd.DataFrame({
    "temperature": np.random.normal(70, 5, 1000),  # in Celsius
    "vibration": np.random.normal(0.5, 0.1, 1000), # in mm/s
    "pressure": np.random.normal(30, 3, 1000),     # in bar
    "rpm": np.random.normal(1500, 100, 1000),      # rotations per minute
})

# Label: 1 = failure, 0 = healthy
data["failure"] = np.where(
    (data["temperature"] > 75) | (data["vibration"] > 0.6), 1, 0
)

# Save dataset (optional for GitHub)
data.to_csv("data/sensor_data.csv", index=False)

print("✅ Dataset created with shape:", data.shape)
print(data.head())

# ----------------------------
# Step 2: Preprocessing
# ----------------------------
X = data.drop("failure", axis=1)
y = data["failure"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ----------------------------
# Step 3: Train Model
# ----------------------------
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# ----------------------------
# Step 4: Evaluation
# ----------------------------
y_pred = model.predict(X_test)

print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Healthy", "Failure"], yticklabels=["Healthy", "Failure"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ----------------------------
# Step 5: Predict on New Data
# ----------------------------
new_data = np.array([[78, 0.65, 32, 1600]])  # Example input
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)[0]

status = "🚨 FAILURE" if prediction == 1 else "✅ HEALTHY"
print("\n🔮 Prediction for new machine data:", status)
