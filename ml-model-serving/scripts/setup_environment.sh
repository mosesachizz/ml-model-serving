#!/bin/bash
# Setup script for ML Model Serving environment

set -e

echo "Setting up ML Model Serving environment..."

# Create required directories
mkdir -p data/raw data/processed models logs

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements/dev.txt

# Download sample data
echo "Downloading sample data..."
python -c "
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
df.to_csv('data/raw/iris.csv', index=False)
print('Sample data saved to data/raw/iris.csv')
"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file from .env.example"
fi

echo "Setup completed successfully!"
echo "To activate virtual environment: source venv/bin/activate"
echo "To run the API: uvicorn app.main:app --reload"