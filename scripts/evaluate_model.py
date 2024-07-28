import joblib
from sklearn.metrics import mean_squared_error
from data_preprocessing import preprocess_data

def evaluate_model(X_test, y_test):
    model = joblib.load('/Users/arnav/Desktop/predictive_maintainence/models/model.pkl')
    scaler = joblib.load('/Users/arnav/Desktop/predictive_maintainence/models/scaler.pkl')
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _ = preprocess_data('/Users/arnav/Desktop/predictive_maintainence/data/turbofan.csv')
    evaluate_model(X_test, y_test)
