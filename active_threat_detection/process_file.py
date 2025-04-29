import pandas as pd
import joblib
import os
from app import load_and_preprocess_data, predict

def process_file(file_path):
    try:
        # Load and preprocess the data
        data = load_and_preprocess_data(file_path)
        
        # Load the model
        if not os.path.exists('model.joblib'):
            print("Error: Model not found. Please make sure model.joblib exists.")
            return
            
        model = joblib.load('model.joblib')
        
        # Make predictions
        results, summary = predict(model, data)
        
        # Print results
        print("\n=== Prediction Results ===")
        print(f"Total Records Processed: {summary['total_records']}")
        print(f"Safe Records Detected: {summary['safe_count']}")
        print(f"Threats Detected: {summary['threat_count']}")
        print(f"Model Accuracy: {summary['accuracy']}")
        
        print("\n=== Detailed Results ===")
        for result in results:
            print(f"\nRecord {result['index']}:")
            print(f"Prediction: {result['prediction_text']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print("Features:")
            for feature, value in result['features'].items():
                print(f"  {feature}: {value}")
                
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == '__main__':
    file_path = input("Enter the path to your CSV file: ")
    process_file(file_path) 