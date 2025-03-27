import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
print("Loading data...")
df = pd.read_csv('data/plr.csv')

# Data cleaning and type conversion
df = df.dropna()
df['Gross'] = pd.to_numeric(df['Gross'], errors='coerce')
df = df[df['Gross'] > 0]  # Remove zero or negative gross values
df = df.dropna()  # Remove any rows with NaN values after conversion

# Remove outliers using IQR method
Q1 = df['Gross'].quantile(0.25)
Q3 = df['Gross'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['Gross'] < (Q1 - 1.5 * IQR)) | (df['Gross'] > (Q3 + 1.5 * IQR)))]

# Feature engineering
print("Engineering features...")

# Target variable transformation
df['Log_Gross'] = np.log1p(df['Gross'])

# Prepare features
X = df[['IMDB_Rating']]
y = df['Log_Gross']

# Split data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features using RobustScaler for better handling of outliers
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create polynomial features (degree=3 for better curve fitting)
poly = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Train model with Ridge regression and stronger regularization
print("Training model...")
model = Ridge(alpha=10.0)  # Increased alpha for stronger regularization
model.fit(X_train_poly, y_train)

# Evaluate model
train_pred = model.predict(X_train_poly)
test_pred = model.predict(X_test_poly)

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

print(f"\nModel Performance:")
print(f"Training R² Score: {train_r2:.4f}")
print(f"Test R² Score: {test_r2:.4f}")
print(f"Test RMSE (log scale): {test_rmse:.4f}")

# Calculate feature importance
feature_names = poly.get_feature_names_out(['IMDB_Rating'])
importance = np.abs(model.coef_)
feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importance})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Save model and preprocessing objects
print("\nSaving model...")
model_data = {
    'model': model,
    'scaler': scaler,
    'poly': poly,
    'r2_score': test_r2
}
joblib.dump(model_data, 'pr_model.pkl')

print("Model training complete!")

# Test some sample predictions
print("\nTesting sample predictions...")
test_ratings = [6.5, 7.5, 8.5, 9.0]

for rating in test_ratings:
    # Create a sample input
    sample_input = pd.DataFrame({
        'IMDB_Rating': [rating]
    })
    
    # Transform and predict
    sample_scaled = scaler.transform(sample_input)
    sample_poly = poly.transform(sample_scaled)
    pred_log = model.predict(sample_poly)[0]
    pred_gross = np.expm1(pred_log)
    
    # Apply rating-based limits for more logical predictions
    if rating <= 4.0:
        max_revenue = 5_000_000  # $5M max for very poor ratings
    elif rating <= 5.0:
        max_revenue = 20_000_000  # $20M max for poor ratings
    elif rating <= 6.0:
        max_revenue = 50_000_000  # $50M max for below average ratings
    elif rating <= 7.0:
        max_revenue = 150_000_000  # $150M max for average ratings
    elif rating <= 8.0:
        max_revenue = 400_000_000  # $400M max for good ratings
    elif rating <= 9.0:
        max_revenue = 800_000_000  # $800M max for excellent ratings
    else:
        max_revenue = 1_500_000_000  # $1.5B max for exceptional ratings
    
    # Apply minimum revenue based on rating
    min_revenue = max(100_000, (rating - 1) * 50_000)
    
    # Clip prediction to reasonable range
    pred_gross = np.clip(pred_gross, min_revenue, max_revenue)
    
    print(f"\nIMDb Rating: {rating}")
    print(f"Predicted Gross: ${pred_gross:,.2f}")

