import joblib
import pandas as pd

def predict(file_path):
    model_path = '/Users/arnav/Desktop/predictive_maintainence/models/model.pkl'
    scaler_path = '/Users/arnav/Desktop/predictive_maintainence/models/scaler.pkl'
    imputer_path = '/Users/arnav/Desktop/predictive_maintainence/models/imputer.pkl'
    features_path = '/Users/arnav/Desktop/predictive_maintainence/models/features.pkl'
    
    # Load model, scaler, imputer, and feature names
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    imputer = joblib.load(imputer_path)
    feature_names = joblib.load(features_path)
    
    # Load new data
    new_data = pd.read_csv(file_path)
    
    # Drop 'RUL' column if it exists
    if 'RUL' in new_data.columns:
        new_data = new_data.drop(columns=['RUL'])
    
    # Identify missing and extra columns
    missing_cols = set(feature_names) - set(new_data.columns)
    extra_cols = set(new_data.columns) - set(feature_names)
    print("Missing columns:", missing_cols)
    print("Extra columns:", extra_cols)
    
    # Add missing columns with NaN values
    for col in missing_cols:
        new_data[col] = pd.NA
    
    # Ensure columns are in the same order as feature_names
    new_data = new_data[feature_names]
    
    # Handle missing values using the same imputer
    new_data_imputed = imputer.transform(new_data)
    
    # Scale data
    X_new_scaled = scaler.transform(new_data_imputed)
    
    # Predict
    predictions = model.predict(X_new_scaled)
    
    return predictions

if __name__ == "__main__":
    new_data_path = '/Users/arnav/Desktop/predictive_maintainence/data/new_data.csv'
    predictions = predict(new_data_path)
    print("Predictions:\n", predictions)
