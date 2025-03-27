import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
data = pd.read_csv("data/knn.csv")

# Clean data - remove rows with missing values and convert to numeric
data['Gross'] = pd.to_numeric(data['Gross'], errors='coerce')
data['IMDB_Rating'] = pd.to_numeric(data['IMDB_Rating'], errors='coerce')
data['Meta_score'] = pd.to_numeric(data['Meta_score'], errors='coerce')
data = data.dropna(subset=['Gross', 'IMDB_Rating', 'Meta_score'])

# Create Success categories based on multiple criteria
def classify_success(row):
    gross = row['Gross']
    imdb = row['IMDB_Rating']
    meta = row['Meta_score']
    
    # Low success criteria (more strict)
    if (gross < 10000000 or  # Less than $10M gross
        imdb < 5.0 or  # Very poor IMDb rating
        meta < 40 or  # Very poor Meta score
        (gross < 5000000 and imdb < 6.0) or  # Low gross with poor rating
        (imdb < 3.0 and meta < 30)):  # Extremely poor ratings
        return 'Low'
    
    # High success criteria (more strict)
    elif (gross > 100000000 and imdb > 7.0 and meta > 60 or  # Blockbuster with good ratings
          (gross > 50000000 and imdb > 7.5 and meta > 70) or  # High gross with excellent ratings
          (imdb > 8.0 and meta > 80 and gross > 20000000)):  # Critical success with decent gross
        return 'High'
    
    # Medium success (everything else)
    else:
        return 'Medium'

# Apply classification
data['Success'] = data.apply(classify_success, axis=1)

# Prepare features
X = data[['Gross', 'IMDB_Rating', 'Meta_score']]
y = data['Success']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling using RobustScaler for better handling of outliers
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model with more neighbors for better generalization
model = KNeighborsClassifier(n_neighbors=7, weights='distance')
model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = model.predict(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Model trained successfully!")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Test some sample predictions
test_samples = [
    [100014, 4.8, 32],     # Low success example (your case)
    [5000000, 4.5, 40],    # Low success example
    [20000000, 6.0, 60],   # Medium success example
    [200000000, 9.0, 90]   # High success example
]
test_scaled = scaler.transform(test_samples)
test_pred = model.predict(test_scaled)
print("\nSample predictions:")
for sample, pred in zip(test_samples, test_pred):
    print(f"Revenue: ${sample[0]:,.0f}, IMDb: {sample[1]}, Meta: {sample[2]} -> {pred}")

# Save model and scaler as a dictionary
model_data = {
    'model': model,
    'scaler': scaler
}
joblib.dump(model_data, "knn_model.pkl")
print("\nModel and scaler saved as knn_model.pkl")
