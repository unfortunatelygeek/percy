import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_test_predictor(dataset_path, output_model_path):
    """
    Train and evaluate the SVM model for test prediction.
    
    Args:
        dataset_path: Path to the dataset CSV generated by feature engineering
        output_model_path: Path to save the trained model
        
    Returns:
        Trained model and evaluation metrics
    """
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # If dataset is empty or too small, return early
    if len(df) < 10:
        print("Warning: Dataset is too small for training. Need at least 10 examples.")
        return None, {'error': 'Insufficient data'}
    
    # Drop non-feature columns
    X = df.drop(['commit_hash', 'test_file', 'test_status'], axis=1)
    y = df['test_status']
    
    # Handle categorical variables
    X = pd.get_dummies(X)
    
    # Fill missing values
    X = X.fillna(0)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection
    k = min(20, X.shape[1])  # Select up to 20 features
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    print(f"Selected features: {selected_features}")
    
    # Hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear']
    }
    
    grid_search = GridSearchCV(
        SVC(probability=True),
        param_grid,
        cv=3,  # Reduced fold count for test dataset
        scoring='f1',
        verbose=1
    )
    
    grid_search.fit(X_train_selected, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Train final model with best parameters
    best_model = grid_search.best_estimator_
    best_model.fit(X_train_selected, y_train)
    
    # Evaluate model
    y_pred = best_model.predict(X_test_selected)
    
    print("Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    
    # Save confusion matrix
    plt.savefig(f"{os.path.splitext(output_model_path)[0]}_cm.png")
    
    # Save model and preprocessing components
    joblib.dump({
        'model': best_model,
        'scaler': scaler,
        'selector': selector,
        'feature_names': X.columns.tolist(),
        'selected_features': selected_features
    }, output_model_path)
    
    return best_model, {
        'classification_report': report,
        'confusion_matrix': cm,
        'best_params': grid_search.best_params_,
        'selected_features': selected_features
    }

def predict_test_failures(model_path, code_changes):
    """
    Predict which tests are likely to fail based on code changes.
    
    Args:
        model_path: Path to the saved model
        code_changes: Dictionary of file paths and their changes
        
    Returns:
        Dictionary mapping test files to failure probability
    """
    # Load model
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    selector = model_data['selector']
    feature_names = model_data['feature_names']
    
    # Extract features from code changes
    features = extract_features_from_changes(code_changes, feature_names)
    
    # Create feature vector matching the training data schema
    feature_vector = pd.DataFrame([features])
    
    # Fill missing columns with zeros
    for col in feature_names:
        if col not in feature_vector.columns:
            feature_vector[col] = 0
    
    # Ensure all columns from training are present and in the same order
    feature_vector = feature_vector[feature_names]
    
    # Scale features
    X_scaled = scaler.transform(feature_vector)
    
    # Select features
    X_selected = selector.transform(X_scaled)
    
    # Predict probabilities
    probabilities = model.predict_proba(X_selected)[0]
    
    # Map back to test files
    test_files = code_changes.get('test_files', [])
    predictions = {}
    
    for test_file in test_files:
        # For simplicity, we use the same prediction for all test files
        # In a real implementation, you would compute features for each test file
        predictions[test_file] = {
            'failure_probability': float(probabilities[1]),
            'should_run': probabilities[1] > 0.3  # Threshold can be adjusted
        }
    
    return predictions

def extract_features_from_changes(code_changes, required_features):
    """
    Extract features from new code changes for prediction.
    
    Args:
        code_changes: Dictionary of file paths and their changes
        required_features: List of features needed by the model
        
    Returns:
        Dictionary of features
    """
    features = {}
    
    # Basic features
    total_files_changed = len(code_changes.get('files', []))
    features['total_files_changed'] = total_files_changed
    
    # Count test vs non-test files
    test_files = code_changes.get('test_files', [])
    features['test_files_changed'] = len(test_files)
    features['non_test_files_changed'] = total_files_changed - len(test_files)
    
    # Look at diffs
    diffs = code_changes.get('diffs', {})
    total_lines_added = 0
    total_lines_removed = 0
    
    for file_path, diff_text in diffs.items():
        added_lines = len(re.findall(r'^\+[^+]', diff_text, re.MULTILINE))
        removed_lines = len(re.findall(r'^-[^-]', diff_text, re.MULTILINE))
        total_lines_added += added_lines
        total_lines_removed += removed_lines
        
        # Check if it's a Python file
        if file_path.endswith('.py'):
            features['modified_functions_count'] = features.get('modified_functions_count', 0) + \
                len(re.findall(r'^\+\s*def\s+', diff_text, re.MULTILINE))
            features['modified_classes_count'] = features.get('modified_classes_count', 0) + \
                len(re.findall(r'^\+\s*class\s+', diff_text, re.MULTILINE))
    
    features['total_lines_added'] = total_lines_added
    features['total_lines_removed'] = total_lines_removed
    features['total_lines_changed'] = total_lines_added + total_lines_removed
    
    # Set default values for missing features
    for feature in required_features:
        if feature not in features:
            features[feature] = 0
    
    return features