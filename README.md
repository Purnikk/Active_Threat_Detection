# Active Threat Detection in Cyber Forensics

A web-based system for detecting potential cybersecurity threats using machine learning.

## Features

- Upload and analyze network traffic data
- Real-time threat detection using Random Forest Classifier
- Detailed feature analysis for each record
- User-friendly web interface
- Confidence scores for predictions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/active_threat_detection.git
cd active_threat_detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python active_threat_detection/app.py
```

2. Open your web browser and go to:
```
http://localhost:5000
```

3. Upload your CSV file containing network traffic data

## Project Structure

```
active_threat_detection/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html         # Web interface template
├── model.joblib           # Trained model (created after first run)
└── requirements.txt       # Python dependencies
```

## Data Format

The system expects CSV files with the following structure:
- Each row represents a network session
- The last column should be the target variable (0 for safe, 1 for threat)
- All other columns should be numeric features

## Requirements

- Python 3.7+
- Flask
- scikit-learn
- pandas
- numpy
- joblib

## License

This project is licensed under the MIT License - see the LICENSE file for details. 