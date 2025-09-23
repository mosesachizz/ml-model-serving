# Data Directory

This directory contains the data used for training and testing ML models.

## Structure

- `raw/`: Raw, unprocessed data files
  - `iris.csv`: Sample Iris dataset (for demonstration)
  
- `processed/`: Processed and cleaned data files
  - `iris_processed.csv`: Processed Iris dataset with numerical targets

## Usage

1. Place raw data files in the `raw/` directory
2. Use the training pipeline to process raw data into `processed/` directory
3. Models are trained using data from the `processed/` directory

## Data Privacy

- Never commit sensitive or personal data
- Use sample data for demonstration purposes
- For production, use proper data storage solutions (S3, database, etc.)

## Sample Data

The included `iris.csv` is the famous Iris dataset for demonstration purposes.