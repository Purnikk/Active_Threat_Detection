import pandas as pd
from app import load_and_preprocess_data

def check_target_values():
    # Load the dataset
    data = load_and_preprocess_data('cybersecurity_intrusion_data (1).csv')
    
    # Get unique values in target column
    unique_values = data['target'].unique()
    print(f"Unique target values: {unique_values}")
    
    # Count occurrences of each value
    value_counts = data['target'].value_counts()
    print("\nValue counts:")
    print(value_counts)

if __name__ == '__main__':
    check_target_values() 