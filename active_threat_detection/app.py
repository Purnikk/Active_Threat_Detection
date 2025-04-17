import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, flash, redirect, url_for
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for flash messages

# Model file path
MODEL_PATH = 'model.joblib'

def load_and_preprocess_data(file):
    try:
        # Read the CSV file
        df = pd.read_csv(file)
        
        # If 'target' column doesn't exist, assume it's the last column
        if 'target' not in df.columns:
            df.columns = list(df.columns[:-1]) + ['target']
            
        # Convert all columns to numeric, replacing errors with 0
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        return df
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")

def train_model(data):
    try:
        # Prepare features and target
        features = data.drop(['target'], axis=1)  # Drop only target column
        target = data['target']

        # Create preprocessing pipeline for all numeric features
        numeric_features = features.columns  # All columns are numeric now
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features)
            ])

        # Create and train pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        pipeline.fit(features, target)
        
        # Save the model
        joblib.dump(pipeline, MODEL_PATH)
        return pipeline
    except Exception as e:
        raise ValueError(f"Error training model: {str(e)}")

def predict(model, data):
    try:
        # Remove target and timestamp if they exist
        features = data.drop(['target'], axis=1, errors='ignore')

        # Make predictions
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        
        # Get total counts
        total_records = len(predictions)
        threat_count = sum(predictions)
        safe_count = total_records - threat_count
        
        # Prepare results
        results = []
        for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
            result = {
                'index': idx + 1,  # Start from 1 instead of using original index
                'prediction': int(pred),
                'prediction_text': 'Potential Threat' if pred == 1 else 'Safe',
                'confidence': float(prob[int(pred)]),
                'features': {
                    name: f"{value:.4f}" for name, value in features.iloc[idx].items()
                }
            }
            results.append(result)
        
        # Add summary statistics
        summary = {
            'total_records': total_records,
            'threat_count': int(threat_count),
            'safe_count': int(safe_count),
            'threat_percentage': f"{(threat_count/total_records)*100:.1f}%"
        }
        
        return results, summary
    except Exception as e:
        raise ValueError(f"Error making predictions: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if not file.filename.endswith('.csv'):
        flash('Please upload a CSV file', 'error')
    return redirect(url_for('index'))

    try:
        # Load and preprocess the data
        data = load_and_preprocess_data(file)
        
        # Load or train the model
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            flash('Training new model...', 'info')
            model = train_model(data)
        
        # Make predictions
        results, summary = predict(model, data)
        
        # Flash summary message
        flash(f"Analysis complete: {summary['threat_count']} threats detected out of {summary['total_records']} records ({summary['threat_percentage']} threat rate)", 'info')
        
        return render_template('index.html', results=results, summary=summary, filename=file.filename)
    
    except Exception as e:
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=8080)