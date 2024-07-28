# Predictive Maintenance API

This application provides a RESTful API for predictive maintenance using machine learning. It allows users to submit new data for prediction and receive the results. The application uses a pre-trained machine learning model to predict the remaining useful life (RUL) or other maintenance-related metrics.

## Features

- **Predictive Maintenance**: Uses a pre-trained model to predict maintenance needs based on input data.
- **Data Handling**: Accepts data in CSV format, processes it, and performs predictions.
- **Model Integration**: Utilizes a trained machine learning model along with a scaler and imputer for data processing.

## Components

### 1. `predict.py`

This script contains the `predict` function that:

- Loads the pre-trained model, scaler, imputer, and feature names.
- Processes new data, ensuring it matches the expected format.
- Handles missing values, scales data, and generates predictions.

### 2. `app.py`

This script sets up a Flask web server with an endpoint to handle predictions:

- **Endpoint**: `/predict`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Form Field**: `file` (CSV file with new data)

### 3. Model Files

Ensure you have the following files in the `models` directory:

- `model.pkl` - The trained machine learning model.
- `scaler.pkl` - The scaler used to normalize data.
- `imputer.pkl` - The imputer used to handle missing values.
- `features.pkl` - The list of feature names expected by the model.

## Setup

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**

   Create a virtual environment and install the required packages:

   ```bash
   python -m venv venv
   source venv/bin/activate  # For Unix/MacOS
   venv\Scripts\activate     # For Windows
   pip install -r requirements.txt
   ```

3. **Run the Flask App**

   ```bash
   python app.py
   ```

   The server will start on `http://localhost:5000`.

## Usage

### Sending a Prediction Request

To send a prediction request, use `curl` or a tool like Postman to POST a CSV file to the `/predict` endpoint.

#### Example Using `curl`

```bash
curl -X POST http://localhost:5000/predict -F "file=@/path/to/new_data.csv"
```

Replace `/path/to/new_data.csv` with the path to your CSV file.

### Example Request

```json
{
  "file": "<path-to-your-csv-file>"
}
```

## Response

The API will return the predictions based on the input data. The format will be a JSON array of predicted values.

## Notes

- Ensure the CSV file matches the format expected by the model (excluding the 'RUL' column if present).
- Missing columns in the input data will be filled with NaN values.

## Troubleshooting

- **404 Error**: Check if the server is running and the URL is correct.
- **400 Error**: Ensure the file is properly formatted and the correct field name is used.

## How is this Useful

1. **Prevents Unexpected Breakdowns**: By predicting when maintenance is needed, it helps prevent sudden equipment failures, reducing downtime and preventing costly repairs.

2. **Optimizes Maintenance Schedules**: It helps in scheduling maintenance activities more effectively, ensuring that they are performed when necessary and avoiding unnecessary maintenance.

3. **Reduces Costs**: By predicting failures before they happen, it reduces the cost of emergency repairs and extends the lifespan of equipment.

4. **Improves Efficiency**: It helps maintain equipment in optimal condition, ensuring smooth operation and increasing overall efficiency.

5. **Informs Decision-Making**: The predictions provide valuable insights that can help in making informed decisions about equipment management and resource allocation.
