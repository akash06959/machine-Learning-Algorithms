import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, r2_score
import pickle

# Load dataset
data = pd.read_csv('data/lgr.csv')

# Clean data - remove rows with missing Gross values and convert to numeric
data['Gross'] = pd.to_numeric(data['Gross'], errors='coerce')
data = data.dropna(subset=['Gross'])

# Create Hit/Flop label based on multiple criteria
# A movie is considered a "Hit" if:
# 1. Gross revenue is above the 40th percentile AND
# 2. IMDb rating is above 6.5
gross_threshold = data['Gross'].quantile(0.4)
data['Hit'] = ((data['Gross'] > gross_threshold) & (data['IMDB_Rating'] > 6.5)).astype(int)

# Prepare features
X = data[['IMDB_Rating', 'No_of_Votes', 'Gross']]
y = data['Hit']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate probabilities for R² score
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred_proba)

print("Model trained successfully!")
print(f"Accuracy: {accuracy:.4f}")
print(f"R² Score: {r2:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print thresholds used for classification
print(f"\nClassification Thresholds:")
print(f"Gross Revenue > ${gross_threshold:,.2f}")
print(f"IMDb Rating > 6.5")

# Test some sample predictions
print("\nSample Predictions:")
sample_inputs = [
    [8.0, 5000000, 3000000000],  # High rating, high votes, very high gross
    [6.0, 100000, 1000000],      # Average rating, moderate votes, low gross
    [9.0, 1000000, 500000000]    # Excellent rating, good votes, high gross
]

for input_data in sample_inputs:
    scaled_input = scaler.transform([input_data])
    pred = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0]
    print(f"\nInput: IMDb={input_data[0]}, Votes={input_data[1]:,}, Gross=${input_data[2]:,}")
    print(f"Prediction: {'Hit' if pred == 1 else 'Flop'} (Confidence: {max(prob):.2%})")

# Save the model and scaler
model_data = {
    'model': model,
    'scaler': scaler
}

with open('lr_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\nModel saved successfully at: lr_model.pkl")
