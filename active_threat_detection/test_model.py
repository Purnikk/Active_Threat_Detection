import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from app import load_and_preprocess_data, train_model, predict

def test_model_accuracy():
    # Load the dataset
    data = load_and_preprocess_data('cybersecurity_intrusion_data (1).csv')
    
    # Split the data into training and testing sets
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(pd.concat([X_train, y_train], axis=1))
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    # Print results
    print(f"\nModel Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, predictions)
    print(f"True Negatives (Safe correctly identified): {cm[0,0]}")
    print(f"False Positives (Safe incorrectly marked as threat): {cm[0,1]}")
    print(f"False Negatives (Threat incorrectly marked as safe): {cm[1,0]}")
    print(f"True Positives (Threat correctly identified): {cm[1,1]}")
    
    # Calculate and print threat detection rate
    total_records = len(predictions)
    threat_count = sum(predictions)
    safe_count = total_records - threat_count
    threat_percentage = (threat_count/total_records)*100
    
    print(f"\nThreat Detection Summary:")
    print(f"Total Records: {total_records}")
    print(f"Threats Detected: {threat_count}")
    print(f"Safe Records: {safe_count}")
    print(f"Threat Rate: {threat_percentage:.1f}%")

if __name__ == '__main__':
    test_model_accuracy() 