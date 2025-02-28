import argparse
import json
import sys
from perceptor.src.ci_integration import get_changed_files, create_selective_test_config
from perceptor.src.svm_model import predict_test_failures

# Parse arguments
parser = argparse.ArgumentParser(description='Predict which tests to run')
parser.add_argument('--commit', help='Specific commit to analyze (default: HEAD)')
parser.add_argument('--config', default='config/perceptor_config.json', 
                    help='Path to config file')
args = parser.parse_args()

# Load config
with open(args.config, 'r') as f:
    config = json.load(f)

# Get changed files
print("Analyzing code changes...")
changes = get_changed_files(config['paths']['repo_path'], args.commit)

if not changes['files']:
    print("No changes detected.")
    sys.exit(0)

print(f"Detected {len(changes['files'])} changed files")

# Predict test failures
print("Predicting test failures...")
predictions = predict_test_failures(config['paths']['model_path'], changes)

# Create test configuration
config_path = create_selective_test_config(predictions, config['paths']['output_config'])

print(f"Generated test configuration at {config_path}")
print("Done!")