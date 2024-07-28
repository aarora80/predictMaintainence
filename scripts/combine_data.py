# scripts/combine_data.py
import pandas as pd

def load_data(train_path, test_path, rul_path):
    # Load training data
    train_data = pd.read_csv(train_path, sep=' ', header=None)
    train_data.drop(train_data.columns[[26, 27]], axis=1, inplace=True)
    train_data.columns = ['unit', 'time', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    
    # Load test data
    test_data = pd.read_csv(test_path, sep=' ', header=None)
    test_data.drop(test_data.columns[[26, 27]], axis=1, inplace=True)
    test_data.columns = ['unit', 'time', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    
    # Load RUL data
    rul_data = pd.read_csv(rul_path, sep=' ', header=None)
    rul_data.drop(rul_data.columns[[1]], axis=1, inplace=True)
    rul_data.columns = ['RUL']
    
    # Add RUL to test data
    max_time = test_data.groupby('unit')['time'].max().reset_index()
    max_time.columns = ['unit', 'max_time']
    test_data = test_data.merge(max_time, on='unit')
    test_data = test_data.merge(rul_data, left_on='unit', right_index=True)
    test_data['RUL'] = test_data['RUL'] + test_data['max_time'] - test_data['time']
    test_data.drop(['max_time'], axis=1, inplace=True)
    
    return train_data, test_data

def save_to_csv(train_data, test_data, output_path):
    combined_data = pd.concat([train_data, test_data], axis=0)
    combined_data.to_csv(output_path, index=False)

if __name__ == "__main__":
    train_path = '/Users/arnav/Desktop/predictive_maintainence/data/CMAPSSData/train_FD001.txt'
    test_path = '/Users/arnav/Desktop/predictive_maintainence/data/CMAPSSData/test_FD001.txt'
    rul_path = '/Users/arnav/Desktop/predictive_maintainence/data/CMAPSSData/RUL_FD001.txt'
    output_path = '/Users/arnav/Desktop/predictive_maintainence/data/turbofan.csv'

    
    train_data, test_data = load_data(train_path, test_path, rul_path)
    save_to_csv(train_data, test_data, output_path)
    print(f"Data saved to {output_path}")
