import json
import os
import sys
from src.feature_engineering import generate_training_dataset
from src.svm_model import train_test_predictor
from src.ci_integration import get_travis_build_data

# Load config
with open('config/perceptor_config.json', 'r') as f:
    config = json.load(f)

# Ensure directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Fetch data from Travis CI
print("Fetching Travis CI build data...")
travis_data = get_travis_build_data(
    config['travis_ci']['repo_slug'],
    config['travis_ci']['api_token'],
    build_limit=200  # Adjust as needed
)

# Generate training dataset
print("Generating training dataset...")
dataset = generate_training_dataset(
    config['paths']['repo_path'],
    travis_data
)

# Save dataset
dataset.to_csv(config['paths']['dataset_path'], index=False)
print(f"Dataset saved to {config['paths']['dataset_path']}")

# Train model
print("Training SVM model...")
model, metrics = train_test_predictor(
    config['paths']['dataset_path'],
    config['paths']['model_path']
)

print("Training complete!")
print(f"Model saved to {config['paths']['model_path']}")
print(f"Accuracy: {metrics['classification_report']['accuracy']:.4f}")