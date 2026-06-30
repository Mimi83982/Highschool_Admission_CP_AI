import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib

print("🚀 Generating sequence data for forecasting...")

# 1. Generate Dummy Sequence Data (Sem 1 to Sem 12 averages)
# We create 1000 simulated student grade progressions
np.random.seed(42)
data = []
for _ in range(1000):
    # Start with a random base average between 60 and 90
    base = np.random.uniform(60, 90)

    # Add some random fluctuation (trend) across 12 semesters
    trend = np.random.uniform(-2, 3)

    # Generate 12 semesters (Sem 1-6 Middle School, Sem 7-12 High School)
    semesters = [min(100, max(0, base + (i * trend) + np.random.normal(0, 2))) for i in range(12)]
    data.append(semesters)

# Define column names for all 12 semesters
columns = [f'Sem{i}' for i in range(1, 13)]
df_seq = pd.DataFrame(data, columns=columns)

# 2. Split into Features (X: Sem 1-6) and Targets (y: Sem 7-12)
X = df_seq[['Sem1', 'Sem2', 'Sem3', 'Sem4', 'Sem5', 'Sem6']]
y = df_seq[['Sem7', 'Sem8', 'Sem9', 'Sem10', 'Sem11', 'Sem12']]

# 3. Scale the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Build the Deep Learning Forecasting Model
print("🧠 Training the Deep Learning Forecasting Model (MLP Regressor)...")
forecaster = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
forecaster.fit(X_scaled, y)

# 5. Save the Model and Scaler
joblib.dump(forecaster, 'grade_forecaster.pkl')
joblib.dump(scaler, 'forecaster_scaler.pkl')

print("✅ Forecasting Models Saved! (grade_forecaster.pkl & forecaster_scaler.pkl)")