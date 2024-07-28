import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(file_path):
    # Load the data
    data = pd.read_csv(file_path)
    print("Initial data shape:", data.shape)
    print("Initial data preview:\n", data.head())

    # Drop rows with missing target values
    data = data.dropna(subset=['RUL'])
    print("Data shape after dropping missing values:", data.shape)

    # Split features and target variable
    X = data.drop(columns=['RUL'])
    y = data['RUL']
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    print("Imputer feature names:", X.columns.tolist())

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    print("Feature means after scaling:", scaler.mean_)
    print("Feature std deviations after scaling:", scaler.scale_)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print("Training features shape:", X_train.shape)
    print("Testing features shape:", X_test.shape)
    print("Training target shape:", y_train.shape)
    print("Testing target shape:", y_test.shape)
    
    return X_train, X_test, y_train, y_test, scaler, imputer, X.columns.tolist()  # Return feature names
