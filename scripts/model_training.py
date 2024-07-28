import sys
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from data_preprocessing import preprocess_data

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training completed.")
    return model

def save_model(model, scaler, imputer, feature_names):
    model_path = '/Users/arnav/Desktop/predictive_maintainence/models/model.pkl'
    scaler_path = '/Users/arnav/Desktop/predictive_maintainence/models/scaler.pkl'
    imputer_path = '/Users/arnav/Desktop/predictive_maintainence/models/imputer.pkl'
    features_path = '/Users/arnav/Desktop/predictive_maintainence/models/features.pkl'
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(imputer, imputer_path)
    joblib.dump(feature_names, features_path)  # Save feature names
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Imputer saved to {imputer_path}")
    print(f"Feature names saved to {features_path}")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, imputer, feature_names = preprocess_data('/Users/arnav/Desktop/predictive_maintainence/data/turbofan.csv')
    print("Data preprocessing completed. Starting model training.")
    model = train_model(X_train, y_train)
    print("Model training completed. Saving model.")
    save_model(model, scaler, imputer, feature_names)
    print("Model training completed and saved.")
