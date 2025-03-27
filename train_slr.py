import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Define the path to your dataset
dataset_path = os.path.join(os.path.dirname(__file__), 'data/slr.csv')

# Load the dataset
data = pd.read_csv(dataset_path)

# Clean Runtime data - remove 'min' and convert to numeric
data['Runtime'] = data['Runtime'].str.replace(' min', '').astype(float)

# Remove invalid ratings and runtimes
data = data[
    (data['IMDB_Rating'] > 0) & 
    (data['IMDB_Rating'] <= 10) & 
    (data['Runtime'] >= 30) & 
    (data['Runtime'] <= 300)
]

# Extract independent (X) and dependent (y) variables
X = data[['Runtime']]
y = data['IMDB_Rating']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
# Clip predictions to valid range
y_pred = np.clip(y_pred, 1, 10)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model trained successfully!")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Model Accuracy (R² Score): {r2:.4f}")

# Test some sample predictions
test_samples = [[90], [120], [150]]  # Sample runtimes
test_scaled = scaler.transform(test_samples)
test_pred = model.predict(test_scaled)
test_pred = np.clip(test_pred, 1, 10)
print("\nSample predictions:")
for runtime, pred in zip(test_samples, test_pred):
    print(f"Runtime: {runtime[0]} minutes -> IMDb Rating: {pred:.1f}")

# Save the model and scaler
model_data = {
    'model': model,
    'scaler': scaler
}
model_path = os.path.join(os.path.dirname(__file__), 'slr_model.pkl')
joblib.dump(model_data, model_path)
print(f"\nModel and scaler saved at {model_path}")
