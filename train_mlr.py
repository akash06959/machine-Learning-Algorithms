import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
from sklearn.ensemble import VotingRegressor
from sklearn.svm import SVR
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("Loading and preparing data...")

# Load the dataset
data = pd.read_csv('data/mlr.csv')

# Convert Released_Year to numeric, coercing errors to NaN
data['Released_Year'] = pd.to_numeric(data['Released_Year'], errors='coerce')

# Enhanced data cleaning and outlier removal
valid_data = data[
    (data['IMDB_Rating'] > 0) & 
    (data['IMDB_Rating'] <= 10) &
    (data['Gross'] >= 1000) &  
    (data['No_of_Votes'] > 0) &  
    (data['Gross'] <= data['Gross'].quantile(0.99)) &  
    (data['Released_Year'].notna()) &  
    (data['Released_Year'] >= 1900) &  
    (data['Released_Year'] <= 2024) &
    (data['No_of_Votes'] >= 1000)  # Remove movies with very few votes
].copy()

# Remove extreme outliers using IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

valid_data = remove_outliers(valid_data, 'Gross')
valid_data = remove_outliers(valid_data, 'No_of_Votes')

print(f"Valid data shape after cleaning: {valid_data.shape}")

# Advanced Feature Engineering
print("Performing advanced feature engineering...")

# 1. Enhanced Transformations
valid_data['Log_Gross'] = np.log1p(valid_data['Gross'])
valid_data['Log_Votes'] = np.log1p(valid_data['No_of_Votes'])
valid_data['Votes_Per_Year'] = valid_data['No_of_Votes'] / (datetime.now().year - valid_data['Released_Year'] + 1)

# 2. Director Features with Enhanced Statistics
director_encoder = LabelEncoder()
valid_data['Director_encoded'] = director_encoder.fit_transform(valid_data['Director'])

# Calculate comprehensive director statistics
director_stats = valid_data.groupby('Director').agg({
    'Gross': ['mean', 'std', 'count', 'max', 'min', 'median'],
    'IMDB_Rating': ['mean', 'std', 'min', 'max'],
    'No_of_Votes': ['mean', 'sum'],
    'Log_Gross': ['mean', 'std']
}).fillna(0)

# Flatten column names
director_stats.columns = [
    'Director_Avg_Gross', 'Director_Std_Gross', 'Director_Movie_Count',
    'Director_Max_Gross', 'Director_Min_Gross', 'Director_Median_Gross',
    'Director_Avg_Rating', 'Director_Rating_Std', 'Director_Min_Rating',
    'Director_Max_Rating', 'Director_Avg_Votes', 'Director_Total_Votes',
    'Director_Avg_Log_Gross', 'Director_Log_Gross_Std'
]

# Calculate director success rate
director_stats['Director_Success_Rate'] = (
    director_stats['Director_Avg_Gross'] / director_stats['Director_Movie_Count']
)

# Convert to serializable format
director_stats_dict = {
    'index': director_stats.index.tolist(),
    'data': director_stats.to_dict(orient='index')
}

# Map director statistics back to the main dataframe
for col in director_stats.columns:
    valid_data[col] = valid_data['Director'].map(director_stats[col])

# 3. Enhanced Time-based Features
current_year = datetime.now().year
valid_data['Years_Since_Release'] = current_year - valid_data['Released_Year']
valid_data['Is_Recent'] = (valid_data['Years_Since_Release'] <= 5).astype(int)
valid_data['Release_Decade'] = (valid_data['Released_Year'] // 10) * 10
valid_data['Is_Golden_Age'] = ((valid_data['Released_Year'] >= 1990) & 
                              (valid_data['Released_Year'] <= 2010)).astype(int)

# 4. Advanced Rating-based Features
valid_data['Rating_Squared'] = valid_data['IMDB_Rating'] ** 2
valid_data['Rating_Cube'] = valid_data['IMDB_Rating'] ** 3
valid_data['Rating_Votes_Interaction'] = valid_data['IMDB_Rating'] * valid_data['Log_Votes']
valid_data['High_Rating_Impact'] = (valid_data['IMDB_Rating'] >= 8.0).astype(int) * valid_data['Log_Votes']

# 5. Normalize Year with Enhanced Method
year_mean = float(valid_data['Released_Year'].mean())
year_std = float(valid_data['Released_Year'].std())
valid_data['Year_normalized'] = (valid_data['Released_Year'] - year_mean) / year_std

# 6. Advanced Success Metrics
valid_data['Recent_Success'] = valid_data['Director_Avg_Gross'] * (1 / (1 + np.exp(-valid_data['Year_normalized'])))
valid_data['Director_Experience'] = np.log1p(valid_data['Director_Movie_Count'])
valid_data['Director_Consistency'] = valid_data['Director_Avg_Rating'] / (valid_data['Director_Rating_Std'] + 1)
valid_data['Director_Trend'] = valid_data['Director_Max_Gross'] / (valid_data['Director_Median_Gross'] + 1)
valid_data['Vote_Weight'] = np.log1p(valid_data['No_of_Votes']) / valid_data['Years_Since_Release']

# Define final feature set
numeric_features = [
    'IMDB_Rating', 'Log_Votes', 'Year_normalized', 'Director_encoded',
    'Director_Avg_Gross', 'Director_Movie_Count', 'Director_Max_Gross',
    'Director_Median_Gross', 'Director_Avg_Rating', 'Director_Success_Rate',
    'Years_Since_Release', 'Is_Recent', 'Is_Golden_Age', 'Rating_Squared',
    'Rating_Cube', 'Rating_Votes_Interaction', 'High_Rating_Impact',
    'Recent_Success', 'Director_Experience', 'Director_Consistency',
    'Director_Trend', 'Vote_Weight', 'Votes_Per_Year'
]

print(f"Total features: {len(numeric_features)}")

# Prepare features
X = valid_data[numeric_features]
y = valid_data['Log_Gross']

# Split the data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Use RobustScaler for better handling of outliers
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Ensemble Model...")

# Create base models
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.01,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    random_state=42,
    n_jobs=-1
)

svr_model = SVR(
    kernel='rbf',
    C=10.0,
    epsilon=0.1,
    gamma='scale'
)

# Create voting regressor
model = VotingRegressor([
    ('xgb', xgb_model),
    ('svr', svr_model)
])

# Train the model with cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring='r2')

print("Cross-validation scores:", cv_scores)
print(f"Average CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Train XGBoost model first for feature importance
xgb_model.fit(X_train_scaled, y_train)

# Train final ensemble model
model.fit(X_train_scaled, y_train)

# Evaluate on test set
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nFinal Model Performance:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# Feature importance analysis (using XGBoost component)
feature_importance = pd.DataFrame({
    'feature': numeric_features,
    'importance': xgb_model.feature_importances_
})
print("\nTop 10 Most Important Features:")
print(feature_importance.sort_values('importance', ascending=False).head(10))

print("\nSaving model data...")

# Save all necessary data
model_data = {
    'model': model,
    'scaler': scaler,
    'director_encoder': director_encoder,
    'year_mean': year_mean,
    'year_std': year_std,
    'director_stats': director_stats_dict,
    'features': numeric_features,
    'r2_score': r2
}

with open('mlr_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model saved successfully at: mlr_model.pkl")

# Test predictions
print("\nTesting sample predictions...")
sample_data = pd.DataFrame({
    'Director': ['Quentin Tarantino', 'Christopher Nolan', 'Peter Jackson'],
    'IMDB_Rating': [6.7, 8.5, 9.0],
    'No_of_Votes': [500000, 1000000, 2000000],
    'Released_Year': [2020, 2021, 2022]
})

for _, row in sample_data.iterrows():
    try:
        # Prepare features for prediction
        features = {}
        
        # Basic features
        features['IMDB_Rating'] = row['IMDB_Rating']
        features['Log_Votes'] = np.log1p(row['No_of_Votes'])
        features['Year_normalized'] = (row['Released_Year'] - year_mean) / year_std
        features['Director_encoded'] = director_encoder.transform([row['Director']])[0]
        
        # Add director statistics
        if row['Director'] in director_stats.index:
            for col in director_stats.columns:
                features[col] = director_stats.loc[row['Director'], col]
        else:
            for col in director_stats.columns:
                features[col] = director_stats[col].mean()
        
        # Calculate additional features
        features['Years_Since_Release'] = current_year - row['Released_Year']
        features['Is_Recent'] = int(features['Years_Since_Release'] <= 5)
        features['Is_Golden_Age'] = int((row['Released_Year'] >= 1990) and (row['Released_Year'] <= 2010))
        features['Rating_Squared'] = features['IMDB_Rating'] ** 2
        features['Rating_Cube'] = features['IMDB_Rating'] ** 3
        features['Rating_Votes_Interaction'] = features['IMDB_Rating'] * features['Log_Votes']
        features['High_Rating_Impact'] = int(features['IMDB_Rating'] >= 8.0) * features['Log_Votes']
        features['Recent_Success'] = features['Director_Avg_Gross'] * (1 / (1 + np.exp(-features['Year_normalized'])))
        features['Director_Experience'] = np.log1p(features['Director_Movie_Count'])
        features['Director_Consistency'] = features['Director_Avg_Rating'] / (features['Director_Rating_Std'] + 1)
        features['Director_Trend'] = features['Director_Max_Gross'] / (features['Director_Median_Gross'] + 1)
        features['Vote_Weight'] = features['Log_Votes'] / features['Years_Since_Release']
        features['Votes_Per_Year'] = row['No_of_Votes'] / (current_year - row['Released_Year'] + 1)
        
        # Prepare feature vector
        feature_vector = np.array([[features[f] for f in numeric_features]])
        
        # Scale features
        scaled_features = scaler.transform(feature_vector)
        
        # Make prediction
        pred_log = model.predict(scaled_features)[0]
        prediction = np.expm1(pred_log)
        
        print(f"\nPrediction for {row['Director']}:")
        print(f"IMDb Rating: {row['IMDB_Rating']}")
        print(f"Votes: {row['No_of_Votes']:,}")
        print(f"Year: {row['Released_Year']}")
        print(f"Predicted Gross: ${prediction:,.2f}")
        
    except Exception as e:
        print(f"Error making prediction: {str(e)}")

print("\nModel training and evaluation complete!")
