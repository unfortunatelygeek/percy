import time
import random
import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
import json
import os
import datetime

# Import your Marksman system
# Assuming the code is in a file called marksman.py
from marksman import MarksmanSystem

class MarksmanTester:
    """Test harness for demonstrating Marksman system capabilities."""
    
    def __init__(self, cache_size_mb=500, test_dir="./test_data"):
        """Initialize the test harness with the Marksman system."""
        logging.info("Initializing Marksman test harness")
        
        # Create test directory if it doesn't exist
        os.makedirs(test_dir, exist_ok=True)
        self.test_dir = test_dir
        
        # Initialize the Marksman system
        self.marksman = MarksmanSystem(
            cache_size_mb=cache_size_mb,
            model_path=os.path.join(test_dir, "models"),
            monitoring_enabled=True
        )
        
        # Define test resources
        self.resources = self._generate_test_resources(50)
        
        # Define test actions
        self.actions = self._generate_test_actions()
        
        # Metrics for tracking test performance
        self.test_metrics = {
            'prediction_accuracy': [],
            'cache_hit_ratio': [],
            'resources_prefetched': [],
            'processing_times': []
        }
    
    def _generate_test_resources(self, num_resources: int) -> Dict[str, Dict]:
        """Generate simulated resources for testing."""
        resources = {}
        file_types = ['py', 'cpp', 'h', 'json', 'md', 'txt', 'html', 'css', 'js']
        projects = ['marksman', 'frontend', 'backend', 'utils', 'tests']
        
        for i in range(num_resources):
            project = random.choice(projects)
            file_type = random.choice(file_types)
            file_name = f"{project}_{i}.{file_type}"
            
            # Create resource content based on file type
            if file_type == 'py':
                content = f"# Python file for {project}\ndef function_{i}():\n    return 'This is test content {i}'\n"
            elif file_type in ['cpp', 'h']:
                content = f"// C++ file for {project}\nvoid function_{i}() {{\n    // This is test content {i}\n}}\n"
            else:
                content = f"Content for {file_name} in project {project}. Test content {i}."
            
            # Size in MB (realistic small file sizes)
            size = random.uniform(0.1, 5.0)
            
            resources[file_name] = {
                'content': content,
                'size': size,
                'project': project,
                'type': file_type
            }
        
        logging.info(f"Generated {len(resources)} test resources")
        return resources
    
    def _generate_test_actions(self) -> List[Dict]:
        """Generate simulated developer actions for testing."""
        action_types = [
            'open_file', 'edit_file', 'save_file', 'close_file', 
            'build_project', 'run_tests', 'debug_start', 'debug_stop'
        ]
        
        # Create sequences of related actions
        actions = []
        for _ in range(10):  # 10 work sessions
            project = random.choice(list(set(r['project'] for r in self.resources.values())))
            project_files = [f for f, r in self.resources.items() if r['project'] == project]
            
            # Simulate a typical workflow
            session_actions = []
            
            # Open a few files
            open_files = random.sample(project_files, min(3, len(project_files)))
            for file in open_files:
                session_actions.append({
                    'type': 'open_file',
                    'file': file,
                    'project': project,
                    'timestamp': time.time()
                })
            
            # Edit, save cycle
            for _ in range(random.randint(5, 15)):
                file = random.choice(open_files)
                session_actions.append({
                    'type': 'edit_file',
                    'file': file,
                    'project': project,
                    'timestamp': time.time()
                })
                if random.random() > 0.3:  # 70% chance to save after edit
                    session_actions.append({
                        'type': 'save_file',
                        'file': file,
                        'project': project,
                        'timestamp': time.time()
                    })
            
            # Build/test actions
            if random.random() > 0.5:  # 50% chance for build
                session_actions.append({
                    'type': 'build_project',
                    'project': project,
                    'timestamp': time.time()
                })
                if random.random() > 0.5:  # 50% chance for tests after build
                    session_actions.append({
                        'type': 'run_tests',
                        'project': project,
                        'timestamp': time.time()
                    })
            
            # Debug session
            if random.random() > 0.7:  # 30% chance for debug
                session_actions.append({
                    'type': 'debug_start',
                    'file': random.choice(open_files),
                    'project': project,
                    'timestamp': time.time()
                })
                session_actions.append({
                    'type': 'debug_stop',
                    'project': project,
                    'timestamp': time.time()
                })
            
            # Close files
            for file in open_files:
                session_actions.append({
                    'type': 'close_file',
                    'file': file,
                    'project': project,
                    'timestamp': time.time()
                })
            
            actions.extend(session_actions)
        
        logging.info(f"Generated {len(actions)} test actions")
        return actions
    
    def resource_fetcher(self, resource_id: str) -> tuple:
        """Simulated resource fetcher function."""
        # Simulate network delay
        time.sleep(random.uniform(0.05, 0.2))
        
        if resource_id in self.resources:
            resource = self.resources[resource_id]
            return resource['content'], resource['size']
        else:
            # Create a new random resource if not found
            content = f"Dynamically generated content for {resource_id}"
            size = random.uniform(0.1, 2.0)
            return content, size
    
    def simulate_resource_usage(self) -> Dict[str, float]:
        """Simulate current resource usage metrics."""
        return {
            'cpu': random.uniform(5.0, 80.0),
            'memory': random.uniform(20.0, 70.0),
            'network': random.uniform(1.0, 30.0),
            'disk': random.uniform(10.0, 50.0),
            'gpu': random.uniform(0.0, 5.0) if random.random() > 0.8 else 0.0
        }
    
    def run_sequential_test(self, num_actions=100) -> Dict:
        """Run a sequential test processing actions one by one."""
        logging.info(f"Starting sequential test with {num_actions} actions")
        
        results = {
            'actions_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'resources_prefetched': 0,
            'prediction_hits': 0,
            'execution_time': 0,
            'metrics_history': []
        }
        
        start_time = time.time()
        
        # Process a subset of actions
        for i, action in enumerate(self.actions[:num_actions]):
            # Process the action
            resources = self.simulate_resource_usage()
            start_process = time.time()
            action_result = self.marksman.process_action(action, resources)
            process_time = time.time() - start_process
            self.test_metrics['processing_times'].append(process_time)
            
            # Simulate fetching resources related to the action
            if 'file' in action:
                try:
                    self.marksman.fetch_resource(action['file'], self.resource_fetcher)
                    # Associate this resource with the action
                    self.marksman.associate_resource_with_action(action['file'], action)
                except Exception as e:
                    logging.error(f"Error fetching resource {action['file']}: {e}")
            
            # Periodically prefetch resources (every 5 actions)
            if i % 5 == 0:
                prefetch_stats = self.marksman.prefetch_resources(self.resource_fetcher)
                results['resources_prefetched'] += prefetch_stats.get('resources_fetched', 0)
                self.test_metrics['resources_prefetched'].append(prefetch_stats.get('resources_fetched', 0))
            
            # Record metrics
            if i % 10 == 0:
                metrics = self.marksman.get_metrics()
                results['metrics_history'].append(metrics)
                
                # Update test metrics
                if metrics.get('hits', 0) + metrics.get('misses', 0) > 0:
                    hit_ratio = metrics.get('hits', 0) / (metrics.get('hits', 0) + metrics.get('misses', 0))
                    self.test_metrics['cache_hit_ratio'].append(hit_ratio)
                
                self.test_metrics['prediction_accuracy'].append(metrics.get('prediction_accuracy', 0))
            
            results['actions_processed'] += 1
        
        results['execution_time'] = time.time() - start_time
        results['cache_hits'] = self.marksman.metrics['cache_hits']
        results['cache_misses'] = self.marksman.metrics['cache_misses']
        
        logging.info(f"Sequential test completed: {results['actions_processed']} actions processed")
        return results
    
    def run_workflow_simulation(self, num_workflows=3) -> Dict:
        """Simulate realistic workflows with specific patterns."""
        logging.info(f"Starting workflow simulation with {num_workflows} workflows")
        
        results = {
            'workflows_completed': 0,
            'actions_processed': 0,
            'resources_accessed': 0,
            'prediction_accuracy': []
        }
        
        # Define some workflow patterns
        workflow_patterns = [
            # Code-build-test pattern
            [
                {'type': 'open_file', 'repeat': 3},
                {'type': 'edit_file', 'repeat': 5},
                {'type': 'save_file', 'repeat': 5},
                {'type': 'build_project', 'repeat': 1},
                {'type': 'run_tests', 'repeat': 1},
                {'type': 'close_file', 'repeat': 3}
            ],
            # Debug pattern
            [
                {'type': 'open_file', 'repeat': 2},
                {'type': 'debug_start', 'repeat': 1},
                {'type': 'edit_file', 'repeat': 3},
                {'type': 'save_file', 'repeat': 3},
                {'type': 'debug_stop', 'repeat': 1},
                {'type': 'close_file', 'repeat': 2}
            ],
            # Review pattern
            [
                {'type': 'open_file', 'repeat': 5},
                {'type': 'edit_file', 'repeat': 2},
                {'type': 'save_file', 'repeat': 2},
                {'type': 'close_file', 'repeat': 5}
            ]
        ]
        
        for workflow_num in range(num_workflows):
            # Select a workflow pattern and project
            pattern = random.choice(workflow_patterns)
            project = random.choice(['marksman', 'frontend', 'backend', 'utils', 'tests'])
            project_files = [f for f, r in self.resources.items() if r['project'] == project]
            
            if not project_files:
                continue
                
            # Track files used in this workflow
            active_files = []
            
            logging.info(f"Starting workflow {workflow_num+1}: {project} project")
            
            # Process actions according to pattern
            for action_template in pattern:
                action_type = action_template['type']
                repeat = action_template['repeat']
                
                for _ in range(repeat):
                    if action_type == 'open_file':
                        # Open a new file
                        available_files = [f for f in project_files if f not in active_files]
                        if not available_files:
                            continue
                        file = random.choice(available_files)
                        active_files.append(file)
                        action = {'type': action_type, 'file': file, 'project': project}
                    elif action_type in ['edit_file', 'save_file']:
                        # Edit or save an active file
                        if not active_files:
                            continue
                        file = random.choice(active_files)
                        action = {'type': action_type, 'file': file, 'project': project}
                    elif action_type == 'close_file':
                        # Close an active file
                        if not active_files:
                            continue
                        file = random.choice(active_files)
                        active_files.remove(file)
                        action = {'type': action_type, 'file': file, 'project': project}
                    elif action_type == 'debug_start':
                        # Start debugging an active file
                        if not active_files:
                            continue
                        file = random.choice(active_files)
                        action = {'type': action_type, 'file': file, 'project': project}
                    else:
                        # Project-wide actions
                        action = {'type': action_type, 'project': project}
                    
                    # Process the action
                    resources = self.simulate_resource_usage()
                    self.marksman.process_action(action, resources)
                    results['actions_processed'] += 1
                    
                    # Fetch related resources
                    if 'file' in action:
                        try:
                            self.marksman.fetch_resource(action['file'], self.resource_fetcher)
                            results['resources_accessed'] += 1
                            self.marksman.associate_resource_with_action(action['file'], action)
                        except Exception as e:
                            logging.error(f"Error fetching resource {action['file']}: {e}")
                    
                    # Record metrics
                    metrics = self.marksman.get_metrics()
                    if metrics.get('prediction_accuracy'):
                        results['prediction_accuracy'].append(metrics.get('prediction_accuracy'))
            
            results['workflows_completed'] += 1
            
            # After each workflow, prefetch for next potential actions
            self.marksman.prefetch_resources(self.resource_fetcher)
            
            # Simulate a pause between workflows
            time.sleep(0.5)
        
        logging.info(f"Workflow simulation completed: {results['workflows_completed']} workflows")
        return results
    
    def test_anomaly_detection(self) -> Dict:
        """Test the anomaly detection capabilities."""
        logging.info("Starting anomaly detection test")
        
        results = {
            'normal_observations': 0,
            'anomalous_observations': 0,
            'detected_anomalies': 0,
            'resource_types': ['cpu', 'memory', 'network', 'disk', 'gpu']
        }
        
        # Add normal observations
        for _ in range(50):
            resources = self.simulate_resource_usage()
            for resource_type, value in resources.items():
                self.marksman.anomaly_detector.add_observation(resource_type, value)
                results['normal_observations'] += 1
        
        # Add anomalous observations
        for resource_type in results['resource_types']:
            # Simulate a spike (3-5x normal values)
            for _ in range(5):
                if resource_type == 'cpu':
                    value = random.uniform(90.0, 100.0)  # CPU spike
                elif resource_type == 'memory':
                    value = random.uniform(85.0, 95.0)  # Memory spike
                elif resource_type == 'network':
                    value = random.uniform(80.0, 100.0)  # Network spike
                elif resource_type == 'disk':
                    value = random.uniform(90.0, 100.0)  # Disk spike
                else:  # GPU
                    value = random.uniform(60.0, 100.0) if random.random() > 0.5 else 0.0
                
                is_anomaly = self.marksman.anomaly_detector.add_observation(resource_type, value)
                results['anomalous_observations'] += 1
                if is_anomaly:
                    results['detected_anomalies'] += 1
        
        # Get recent anomalies
        anomalies = self.marksman.get_anomalies(hours=1)
        results['anomalies'] = anomalies
        
        logging.info(f"Anomaly detection test completed: {results['detected_anomalies']} anomalies detected")
        return results
    
    def visualize_results(self, test_results):
        """Create visualizations of test results."""
        logging.info("Generating visualizations")
        
        # Create output directory
        os.makedirs(os.path.join(self.test_dir, "visualizations"), exist_ok=True)
        output_dir = os.path.join(self.test_dir, "visualizations")
        
        # Plot cache performance
        self._plot_cache_performance(output_dir)
        
        # Plot prediction accuracy
        self._plot_prediction_accuracy(output_dir)
        
        # Plot resource usage
        if 'metrics_history' in test_results:
            self._plot_resource_usage(test_results['metrics_history'], output_dir)
        
        # Generate summary report
        self._generate_report(test_results, output_dir)
        
        logging.info(f"Visualizations saved to {output_dir}")
    
    def _plot_cache_performance(self, output_dir):
        """Plot cache hit ratio over time."""
        if not self.test_metrics['cache_hit_ratio']:
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.test_metrics['cache_hit_ratio'], 'b-', linewidth=2)
        plt.title('Cache Hit Ratio Over Time')
        plt.xlabel('Measurement Interval')
        plt.ylabel('Hit Ratio')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cache_performance.png'))
        plt.close()
    
    def _plot_prediction_accuracy(self, output_dir):
        """Plot prediction accuracy over time."""
        if not self.test_metrics['prediction_accuracy']:
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.test_metrics['prediction_accuracy'], 'g-', linewidth=2)
        plt.title('Resource Prediction Accuracy')
        plt.xlabel('Measurement Interval')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prediction_accuracy.png'))
        plt.close()
    
    def _plot_resource_usage(self, metrics_history, output_dir):
        """Plot resource usage over time."""
        if not metrics_history:
            return
        
        # Extract resource usage
        resource_types = ['cpu', 'memory', 'network', 'disk', 'gpu']
        resource_data = {rt: [] for rt in resource_types}
        
        for metrics in metrics_history:
            for rt in resource_types:
                if rt in metrics:
                    resource_data[rt].append(metrics[rt])
                else:
                    resource_data[rt].append(0)
        
        # Plot
        plt.figure(figsize=(12, 8))
        for rt in resource_types:
            if resource_data[rt]:
                plt.plot(resource_data[rt], label=rt.capitalize(), linewidth=2)
        
        plt.title('Resource Usage Over Time')
        plt.xlabel('Measurement Interval')
        plt.ylabel('Usage (%)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'resource_usage.png'))
        plt.close()
    
    def _generate_report(self, test_results, output_dir):
        """Generate a summary report of test results."""
        final_metrics = self.marksman.get_metrics()
        
        report = [
            "# Marksman System Test Report",
            f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Test Summary",
            f"- Actions Processed: {test_results.get('actions_processed', 0)}",
            f"- Cache Hits: {final_metrics.get('hits', 0)}",
            f"- Cache Misses: {final_metrics.get('misses', 0)}",
            f"- Hit Ratio: {final_metrics.get('hit_ratio', 0):.2f}",
            f"- Resources Prefetched: {test_results.get('resources_prefetched', 0)}",
            f"- Execution Time: {test_results.get('execution_time', 0):.2f} seconds",
            "",
            "## Cache Performance",
            f"- Current Cache Size: {final_metrics.get('size_mb', 0):.2f} MB",
            f"- Maximum Cache Size: {final_metrics.get('max_size_mb', 0)} MB",
            f"- Usage: {final_metrics.get('usage_percent', 0):.2f}%",
            f"- Items in Cache: {final_metrics.get('items', 0)}",
            f"- Evictions: {final_metrics.get('evictions', 0)}",
            "",
            "## Prediction Performance",
            f"- Prediction Accuracy: {final_metrics.get('prediction_accuracy', 0):.2f}%",
            f"- Total States in Markov Model: {final_metrics.get('total_states', 0)}",
            "",
            "## Anomaly Detection",
            f"- Total Anomalies Detected: {final_metrics.get('total_anomalies', 0)}",
            f"- Recent Anomalies (24h): {final_metrics.get('recent_anomalies', 0)}",
            "",
            "## Visualizations",
            "- Cache Performance: [cache_performance.png](cache_performance.png)",
            "- Prediction Accuracy: [prediction_accuracy.png](prediction_accuracy.png)",
            "- Resource Usage: [resource_usage.png](resource_usage.png)"
        ]
        
        with open(os.path.join(output_dir, 'test_report.md'), 'w') as f:
            f.write('\n'.join(report))
    
    def run_complete_demo(self):
        """Run a complete demonstration of the Marksman system."""
        logging.info("Starting complete Marksman system demonstration")
        
        # Step 1: Sequential test
        sequential_results = self.run_sequential_test(num_actions=100)
        
        # Step 2: Workflow simulation
        workflow_results = self.run_workflow_simulation(num_workflows=5)
        
        # Step 3: Anomaly detection test
        anomaly_results = self.test_anomaly_detection()
        
        # Step 4: Create dashboard
        dashboard_id = self.marksman.create_dashboard()
        if dashboard_id:
            logging.info(f"Created dashboard with ID: {dashboard_id}")
        
        # Step 5: Visualize results
        combined_results = {
            **sequential_results,
            **workflow_results,
            'anomalies': anomaly_results['detected_anomalies']
        }
        self.visualize_results(combined_results)
        
        # Print final metrics
        final_metrics = self.marksman.get_metrics()
        logging.info("Demo completed with final metrics:")
        logging.info(json.dumps(final_metrics, indent=2))
        
        print("\n" + "="*80)
        print("MARKSMAN SYSTEM DEMONSTRATION COMPLETED")
        print("="*80)
        print(f"Results saved to: {os.path.join(self.test_dir, 'visualizations')}")
        print(f"- Cache Hit Ratio: {final_metrics.get('hit_ratio', 0):.2f}")
        print(f"- Prediction Accuracy: {final_metrics.get('prediction_accuracy', 0):.2f}%")
        print(f"- Anomalies Detected: {final_metrics.get('total_anomalies', 0)}")
        print("="*80)
        
        # Shutdown
        self.marksman.shutdown()
        return combined_results


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("marksman_test.log"),
            logging.StreamHandler()
        ]
    )
    
    # Run the complete demo
    tester = MarksmanTester(cache_size_mb=1000, test_dir="./marksman_test_results")
    tester.run_complete_demo()