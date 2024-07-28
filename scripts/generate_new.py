import pandas as pd
import numpy as np

# Define the column names that match the training features
columns = [
    'unit', 'time', 'op_setting_1', 'op_setting_2', 'op_setting_3',
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
    'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
    'sensor_21'
]

# Create a DataFrame with placeholder values
data = {
    'unit': np.random.randint(1, 10, size=5),
    'time': np.arange(1, 6),
    'op_setting_1': np.random.uniform(-0.01, 0.01, size=5),
    'op_setting_2': np.random.uniform(-0.01, 0.01, size=5),
    'op_setting_3': np.random.uniform(100, 100, size=5),
    'sensor_1': np.random.uniform(500, 550, size=5),
    'sensor_2': np.random.uniform(600, 650, size=5),
    'sensor_3': np.random.uniform(500, 550, size=5),
    'sensor_4': np.random.uniform(1500, 1600, size=5),
    'sensor_5': np.random.uniform(1400, 1500, size=5),
    'sensor_6': np.random.uniform(15, 20, size=5),
    'sensor_7': np.random.uniform(20, 25, size=5),
    'sensor_8': np.random.uniform(500, 600, size=5),
    'sensor_9': np.random.uniform(2300, 2400, size=5),
    'sensor_10': np.random.uniform(100, 150, size=5),
    'sensor_11': np.random.uniform(30, 35, size=5),
    'sensor_12': np.random.uniform(390, 400, size=5),
    'sensor_13': np.random.uniform(2400, 2500, size=5),
    'sensor_14': np.random.uniform(100, 150, size=5),
    'sensor_15': np.random.uniform(35, 45, size=5),
    'sensor_16': np.random.uniform(30, 40, size=5),
    'sensor_17': np.random.uniform(2300, 2400, size=5),
    'sensor_18': np.random.uniform(100, 150, size=5),
    'sensor_19': np.random.uniform(35, 45, size=5),
    'sensor_20': np.random.uniform(30, 40, size=5),
    'sensor_21': np.random.uniform(10, 15, size=5)
}

df_new = pd.DataFrame(data, columns=columns)

# Save to CSV
df_new.to_csv('new_data.csv', index=False)

print("new_data.csv created successfully.")
