import os
import sys
import json
import requests
import logging
from git import Repo
import pandas as pd
import re

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('perceptor')

def get_travis_build_data(repo_slug, api_token=None, build_limit=100):
    """
    Fetch build data from Travis CI API.
    
    Args:
        repo_slug: GitHub repository in format "owner/repo"
        api_token: Travis CI API token
        build_limit: Max number of builds to retrieve
        
    Returns:
        DataFrame of build data
    """
    # For the sample repository, we'll simulate Travis CI data
    # In a real implementation, this would call the Travis CI API
    
    logger.info("Using simulated Travis CI data for the sample repository")
    
    # Create simulated data
    builds = []
    repo_path = os.path.abspath(os.path.join(os.getcwd(), 'test-repo'))
    repo = Repo(repo_path)
    
    # Get commit history
    commits = list(repo.iter_commits())[:10]  # Limit to 10 most recent commits
    
    for i, commit in enumerate(commits):
        # Simulate test results
        test_files = []
        for root, _, files in os.walk(os.path.join(repo_path, 'tests')):
            for file in files:
                if file.startswith('test_'):
                    rel_path = os.path.join(root, file).replace(repo_path + '/', '')
                    test_files.append(rel_path)
        
        # Simulate pass/fail status - mostly passing with some failures
        test_results = {}
        for test_file in test_files:
            # Random success with 80% probability
            import random
            test_results[test_file] = 'pass' if random.random() < 0.8 else 'fail'
        
        builds.append({
            'build_id': i + 1000,
            'commit_hash': commit.hexsha,
            'branch': 'main',
            'created_at': commit.authored_datetime.isoformat(),
            'state': 'passed' if all(result == 'pass' for result in test_results.values()) else 'failed',
            'test_results': test_results
        })
    
    return pd.DataFrame(builds)

def get_changed_files(repo_path, commit_hash=None):
    """
    Get files changed in the latest commit or specified commit.
    
    Args:
        repo_path: Path to the git repository
        commit_hash: Optional specific commit to analyze
        
    Returns:
        Dictionary with changed files and their diffs
    """
    repo = Repo(repo_path)
    
    if commit_hash:
        try:
            commit = repo.commit(commit_hash)
        except:
            logger.error(f"Commit {commit_hash} not found in repository")
            return {'files': [], 'diffs': {}, 'test_files': []}
    else:
        commit = repo.head.commit
    
    parent_commit = commit.parents[0] if commit.parents else None
    
    if not parent_commit:
        logger.warning("No parent commit found, can't compute diff")
        return {'files': [], 'diffs': {}, 'test_files': []}
    
    # Get list of changed files
    changed_files = [item.a_path for item in commit.diff(parent_commit)]
    
    # Get diffs for each file
    diffs = {}
    for file_path in changed_files:
        try:
            file_diff = repo.git.diff(parent_commit.hexsha, commit.hexsha, '--', file_path)
            diffs[file_path] = file_diff
        except Exception as e:
            logger.warning(f"Error getting diff for {file_path}: {e}")
    
    # Identify test files
    test_files = [f for f in changed_files if f.startswith('tests/test_') or '/test_' in f]
    
    logger.info(f"Found {len(changed_files)} changed files, including {len(test_files)} test files")
    
    return {
        'files': changed_files,
        'diffs': diffs,
        'test_files': test_files
    }

def create_selective_test_config(predictions, output_path):
    """
    Create configuration file for selective test execution.
    
    Args:
        predictions: Dictionary of test predictions
        output_path: Path to save the configuration
        
    Returns:
        Path to the created configuration file
    """
    # Filter tests to run based on predictions
    tests_to_run = [
        test_file for test_file, prediction in predictions.items()
        if prediction.get('should_run', False)
    ]
    
    config = {
        'version': '1.0',
        'tests_to_run': tests_to_run,
        'skip_remaining': True,
        'prediction_timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save configuration
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created test configuration at {output_path}")
    logger.info(f"Selected {len(tests_to_run)} tests to run out of {len(predictions)} total tests")
    
    return output_path