import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os
from git import Repo

def extract_code_features(commit_diff, file_path):
    """
    Extract features from code changes in a specific file.
    
    Args:
        commit_diff: The git diff for a specific commit
        file_path: Path to the file being analyzed
    
    Returns:
        Dictionary of features extracted from the code changes
    """
    # Skip if file wasn't modified in this commit
    if file_path not in commit_diff:
        return {}
    
    diff_text = commit_diff[file_path]
    features = {}
    
    # Count number of lines added/removed
    added_lines = len(re.findall(r'^\+[^+]', diff_text, re.MULTILINE))
    removed_lines = len(re.findall(r'^-[^-]', diff_text, re.MULTILINE))
    features['lines_added'] = added_lines
    features['lines_removed'] = removed_lines
    features['lines_changed'] = added_lines + removed_lines
    
    # Identify code components modified
    if file_path.endswith('.py'):
        # Python-specific features
        features['modified_functions'] = len(re.findall(r'^\+\s*def\s+', diff_text, re.MULTILINE))
        features['modified_classes'] = len(re.findall(r'^\+\s*class\s+', diff_text, re.MULTILINE))
        features['import_changes'] = len(re.findall(r'^\+\s*import\s+|^\+\s*from\s+.*import', diff_text, re.MULTILINE))
    elif file_path.endswith('.js'):
        # JavaScript-specific features
        features['modified_functions'] = len(re.findall(r'^\+\s*function\s+|^\+.*=>\s*{', diff_text, re.MULTILINE))
        features['modified_classes'] = len(re.findall(r'^\+\s*class\s+', diff_text, re.MULTILINE))
    
    # Generate file location features
    parts = file_path.split('/')
    features['is_test'] = 1 if 'test' in file_path.lower() else 0
    features['directory_depth'] = len(parts) - 1
    features['file_extension'] = os.path.splitext(file_path)[1]
    
    return features

def extract_workflow_features(workflow_file):
    """
    Extract features from CI/CD workflow files.
    
    Args:
        workflow_file: Content of the workflow file
    
    Returns:
        Dictionary of workflow-related features
    """
    features = {}
    
    # Count workflow steps
    features['total_steps'] = len(re.findall(r'^\s*-\s+name:', workflow_file, re.MULTILINE))
    
    # Identify test frameworks used
    test_frameworks = {
        'pytest': 'pytest' in workflow_file,
        'jest': 'jest' in workflow_file,
        'unittest': 'unittest' in workflow_file,
        'mocha': 'mocha' in workflow_file
    }
    features.update(test_frameworks)
    
    # Count conditional execution patterns
    features['conditional_steps'] = len(re.findall(r'if:\s+', workflow_file, re.MULTILINE))
    
    return features

def create_test_dependency_graph(repo_path, test_files):
    """
    Create a dependency graph between test files and implementation files.
    
    Args:
        repo_path: Path to the git repository
        test_files: List of test files in the repository
        
    Returns:
        Dictionary mapping test files to implementation files they depend on
    """
    dependency_graph = {}
    
    for test_file in test_files:
        test_file_path = os.path.join(repo_path, test_file)
        if not os.path.exists(test_file_path):
            continue
            
        with open(test_file_path, 'r') as f:
            try:
                content = f.read()
            except UnicodeDecodeError:
                # Skip binary files
                continue
        
        # Extract imports (this is a simplified approach)
        imports = re.findall(r'import\s+(\w+)|from\s+(\w+)', content)
        imported_modules = set([imp[0] or imp[1] for imp in imports])
        
        # Map to implementation files (simplified)
        impl_files = []
        for module in imported_modules:
            # Search for potential implementation files
            for root, _, files in os.walk(repo_path):
                for file in files:
                    if file.startswith(module) and not file.startswith('test_'):
                        impl_files.append(os.path.join(root, file).replace(repo_path + '/', ''))
        
        dependency_graph[test_file] = impl_files
    
    return dependency_graph

def generate_training_dataset(repo_path, travis_builds):
    """
    Generate training dataset for the SVM model.
    
    Args:
        repo_path: Path to the git repository
        travis_builds: DataFrame containing Travis CI build data
        
    Returns:
        DataFrame ready for SVM training
    """
    repo = Repo(repo_path)
    training_data = []
    
    # Identify test files
    test_files = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.startswith('test_') or 'test' in root.lower():
                rel_path = os.path.join(root, file).replace(repo_path + '/', '')
                test_files.append(rel_path)
    
    # Create dependency graph
    dependency_graph = create_test_dependency_graph(repo_path, test_files)
    
    # Process each build
    for _, build in travis_builds.iterrows():
        commit_hash = build['commit_hash']
        try:
            commit = repo.commit(commit_hash)
        except:
            # Skip if commit not found
            continue
            
        parent_commit = commit.parents[0] if commit.parents else None
        
        if not parent_commit:
            continue
        
        # Get the diff between current commit and parent
        diff = repo.git.diff(parent_commit.hexsha, commit.hexsha)
        
        # Parse diff to get per-file changes
        file_diffs = {}
        current_file = None
        current_diff = []
        
        for line in diff.split('\n'):
            if line.startswith('diff --git'):
                if current_file:
                    file_diffs[current_file] = '\n'.join(current_diff)
                current_diff = []
                # Extract filename from diff header
                file_match = re.search(r'b/(.*?)$', line)
                if file_match:
                    current_file = file_match.group(1)
            elif current_file:
                current_diff.append(line)
        
        if current_file:
            file_diffs[current_file] = '\n'.join(current_diff)
        
        # Extract workflow features
        workflow_features = {}
        workflow_path = os.path.join(repo_path, '.github/workflows')
        if os.path.exists(workflow_path):
            workflow_files = [f for f in os.listdir(workflow_path) 
                            if f.endswith('.yml') or f.endswith('.yaml')]
            
            for wf_file in workflow_files:
                with open(os.path.join(workflow_path, wf_file), 'r') as f:
                    workflow_features.update(extract_workflow_features(f.read()))
        
        # Process each test file
        for test_file in test_files:
            # Check if test passed or failed in this build
            test_status = build['test_results'].get(test_file, 'unknown')
            if test_status == 'unknown':
                continue  # Skip if we don't have test results
            
            # Combine features
            features = {
                'commit_hash': commit_hash,
                'test_file': test_file,
                'test_status': 1 if test_status == 'pass' else 0
            }
            
            # Add workflow features
            features.update(workflow_features)
            
            # Add code change features
            code_features = extract_code_features(file_diffs, test_file)
            features.update(code_features)
            
            # Add dependency-related features
            dependent_files = dependency_graph.get(test_file, [])
            dependent_changed = False
            for dep_file in dependent_files:
                if dep_file in file_diffs:
                    dependent_changed = True
                    dep_features = extract_code_features(file_diffs, dep_file)
                    # Prefix to avoid collision
                    for key, value in dep_features.items():
                        features[f'dep_{key}'] = value
            
            features['dependent_changed'] = 1 if dependent_changed else 0
            
            # Add relationship features
            features['direct_change'] = 1 if test_file in file_diffs else 0
            features['dependency_change_ratio'] = sum(1 for f in dependent_files if f in file_diffs) / len(dependent_files) if dependent_files else 0
            
            # Store this example
            training_data.append(features)
    
    return pd.DataFrame(training_data)