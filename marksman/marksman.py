import numpy as np
import pandas as pd
import datetime
import json
import logging
import time
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict, Counter
import pickle
import os
from sklearn.cluster import KMeans
import prometheus_client as prom
from grafana_client import GrafanaApi

REGISTRY = prom.CollectorRegistry(auto_describe=True)

# The Cache class implements an enhanced caching system with a hybrid LRU-LFU (Least Recently Used - Least Frequently Used) 
# eviction strategy. It includes features like decay factors to reduce the impact of older accesses. 
# Key methods include get for retrieving data, put for adding data, and get_metrics for performance monitoring. 
# The cache is limited by a maximum size in MB, and it tracks metrics such as hits, misses, and evictions.
# See the paper: An Improved Cache Eviction Strategy: Combining Least Recently Used and Least Frequently Used
# Citation: An improved cache eviction strategy: combining least recently used and least frequently used policies.
# IEEE Conference Publication | IEEE Xplore. https://ieeexplore.ieee.org/document/10454976
# class Cache:
#     def __init__(self, max_size_mb: int = 1000, decay_factor: float = 0.9):
#         """
#         args:
#             max_size_mb: Maximum cache size in MB
#             decay_factor: Factor to decay frequency scores over time (0-1)
#         """
#         self.max_size_mb = max_size_mb
#         self.current_size = 0
#         self.data = {}  # resource_id -> (data, size_mb)
#         self.freq = defaultdict(int)  # Access frequency counter
#         self.access_time = {}  # Last access timestamp
#         self.decay_factor = decay_factor
#         self.metrics = {
#             'hits': 0,
#             'misses': 0,
#             'evictions': 0,
#         }
    
#     def get(self, key: str) -> Optional[Any]:
#         """
#         Get item from cache and update its frequency and timestamp.        
#         args:
#             key: Cache key to retrieve            
#         Returns:
#             Cached data or none if not found!
#         """
#         if key in self.data:
#             data, _ = self.data[key]
#             self.freq[key] += 1
#             self.access_time[key] = time.time()
#             self.metrics['hits'] += 1
#             return data
        
#         self.metrics['misses'] += 1
#         return None
    
#     def put(self, key: str, data: Any, size_mb: float) -> bool:
#         """
#         Add item to cache with specified size.        
#         args:
#             key: Cache key
#             data: Data to store
#             size_mb: Size of data in MB
#         Returns:
#             Boolean indicating if caching was successful
#         """
#         #if already cached, just update
#         if key in self.data:
#             _, old_size = self.data[key]
#             self.data[key] = (data, size_mb)
#             self.current_size = self.current_size - old_size + size_mb
#             self.freq[key] += 1
#             self.access_time[key] = time.time()
#             return True
        
#         #agar adding would exceed cache size, evict items
#         if self.current_size + size_mb > self.max_size_mb:
#             self._evict_until_fits(size_mb)
        
#         #if still too large after eviction, can't cache
#         if self.current_size + size_mb > self.max_size_mb:
#             return False
        
#         #add to cache
#         self.data[key] = (data, size_mb)
#         self.freq[key] = 1
#         self.access_time[key] = time.time()
#         self.current_size += size_mb
#         return True
    
#     def _evict_until_fits(self, required_space: float) -> None:
#         """
#         Evict items until required space is available. Uses hybrid LRU-LFU with time decay for eviction decisions.
        
#         args:
#             required_space: Space needed in MB
#         """
#         if not self.data:
#             return
        
#         now = time.time()
#         items_to_evict = []
        
#         #Calculate scores for all items (combining frequency and recency)
#         scores = {}
#         for key in self.data:
#             # Time factor (higher score for more recent access!)
#             time_factor = now - self.access_time[key]
#             # Apply decay to frequency based on time (older accesses count less)
#             decayed_freq = self.freq[key] * (self.decay_factor ** time_factor)
#             # Final score (normalize by size for efficiency)
#             _, size = self.data[key]
#             scores[key] = decayed_freq / (size * (1 + time_factor))
        
#         # Sort by score (lowest first ---> these will be evicted)
#         sorted_items = sorted(scores.items(), key=lambda x: x[1])
        
#         # Evict until we have enough space
#         space_freed = 0
#         for key, _ in sorted_items:
#             if space_freed >= required_space:
#                 break
            
#             _, size = self.data[key]
#             del self.data[key]
#             del self.freq[key]
#             del self.access_time[key]
            
#             space_freed += size
#             self.current_size -= size
#             self.metrics['evictions'] += 1
    
#     def clear(self) -> None:
#         """Clear the cache completely."""
#         self.data = {}
#         self.freq = defaultdict(int)
#         self.access_time = {}
#         self.current_size = 0
    
#     def get_metrics(self) -> Dict:
#         """Get cache performance metrics."""
#         total_requests = self.metrics['hits'] + self.metrics['misses']
#         hit_ratio = self.metrics['hits'] / total_requests if total_requests > 0 else 0
        
#         return {
#             'size_mb': self.current_size,
#             'max_size_mb': self.max_size_mb,
#             'usage_percent': (self.current_size / self.max_size_mb) * 100 if self.max_size_mb > 0 else 0,
#             'items': len(self.data),
#             'hits': self.metrics['hits'],
#             'misses': self.metrics['misses'],
#             'evictions': self.metrics['evictions'],
#             'hit_ratio': hit_ratio
#         }

class Cache:
    def __init__(self, max_size_mb=1000, decay_factor=0.9):
        self.max_size_mb = max_size_mb
        self.current_size = 0
        self.data = {}
        self.freq = defaultdict(int)
        self.access_time = {}
        self.decay_factor = decay_factor
        self.metrics = {'hits': 0, 'misses': 0, 'evictions': 0}
    
    def get(self, key):
        if key in self.data:
            data, _ = self.data[key]
            self.freq[key] += 1
            self.access_time[key] = time.time()
            self.metrics['hits'] += 1
            return data
        
        self.metrics['misses'] += 1
        return None
    
    def put(self, key, data, size_mb):
        if key in self.data:
            _, old_size = self.data[key]
            self.data[key] = (data, size_mb)
            self.current_size = self.current_size - old_size + size_mb
            self.freq[key] += 1
            self.access_time[key] = time.time()
            return True
        
        if self.current_size + size_mb > self.max_size_mb:
            self._evict_until_fits(size_mb)
        
        if self.current_size + size_mb > self.max_size_mb:
            return False
        
        self.data[key] = (data, size_mb)
        self.freq[key] = 1
        self.access_time[key] = time.time()
        self.current_size += size_mb
        return True
    
    def _evict_until_fits(self, required_space):
        if not self.data:
            return
        
        now = time.time()
        scores = {
            key: self.freq[key] / (self.data[key][1] * (1 + (now - self.access_time[key])))
            for key in self.data
        }
        
        sorted_items = sorted(scores.items(), key=lambda x: x[1])
        
        space_freed = 0
        for key, _ in sorted_items:
            if space_freed >= required_space:
                break
            
            _, size = self.data.pop(key)
            self.freq.pop(key)
            self.access_time.pop(key)
            space_freed += size
            self.current_size -= size
            self.metrics['evictions'] += 1
    
    def get_metrics(self):
        total_requests = self.metrics['hits'] + self.metrics['misses']
        hit_ratio = self.metrics['hits'] / total_requests if total_requests > 0 else 0
        return {
            'size_mb': self.current_size,
            'max_size_mb': self.max_size_mb,
            'usage_percent': (self.current_size / self.max_size_mb) * 100,
            'items': len(self.data),
            'hits': self.metrics['hits'],
            'misses': self.metrics['misses'],
            'evictions': self.metrics['evictions'],
            'hit_ratio': hit_ratio
        }

# AnomalyDetector is just that - an anomaly detector. It detects anomalies in resource usage and access patterns
class AnomalyDetector:    
    def __init__(self, window_size: int = 100, threshold: float = 2.0, decay_factor: float = 0.95):
        """
        Initialize the anomaly detector.        
        args:
            window_size: Number of observations to use for baseline
            threshold: Standard deviation multiplier for anomaly detection
            decay_factor: Weight factor for older observations
        """
        self.window_size = window_size
        self.threshold = threshold
        self.decay_factor = decay_factor
        self.observations = {
            'cpu': [],
            'memory': [],
            'network': [],
            'disk': [],
            'gpu': [],
            'response_time': []
        }
        self.anomalies = []
        
    def add_observation(self, metric_type: str, value: float, timestamp: Optional[datetime.datetime] = None) -> bool:
        """
        add a new resource usage observation and check for any anomaly.        
        args:
            metric_type: Type of metric (cpu, memory, etc.)
            value: Observed value
            timestamp: Observation timestamp (optional)            
        Returns:
            Boolean indicating if observation is anomalous
        """
        if metric_type not in self.observations:
            return False
        
        if timestamp is None:
            timestamp = datetime.datetime.now()
            
        # Add new observation
        self.observations[metric_type].append((value, timestamp))
        
        # keep only window_size most recent observations
        if len(self.observations[metric_type]) > self.window_size:
            self.observations[metric_type] = self.observations[metric_type][-self.window_size:]
        
        # check if new observation is anomalous
        is_anomaly = self._check_anomaly(metric_type, value, timestamp)
        
        if is_anomaly:
            self.anomalies.append({
                'metric': metric_type,
                'value': value,
                'timestamp': timestamp,
                'threshold': self.threshold
            })
            
        return is_anomaly
    
    def _check_anomaly(self, metric_type: str, value: float, timestamp: datetime.datetime) -> bool:
        """
        Check if an observation is anomalous using statistical methods.   
        This works similar to the Z-score method for anomaly detection
        (find mean and SD, then determine how many SDs away from the mean a new observation is), 
        combined with exponential time decay to weight recent observations more heavily.    
        args:
            metric_type: Type of metric
            value: Value to check
            timestamp: Observation timestamp
            
        Returns:
            Boolean indicating if observation is anomalous
        """
        if len(self.observations[metric_type]) < 10:  # Need minimum data for baseline
            return False
        
        # Apply time decay to weight recent observations more heavily
        now = datetime.datetime.now()
        values = []
        weights = []
        
        for val, ts in self.observations[metric_type]:
            time_diff = (now - ts).total_seconds() / 3600  # Hours difference
            weight = self.decay_factor ** time_diff
            values.append(val)
            weights.append(weights)
        
        # Calculate mean and standard deviation
        weighted_mean = np.average(values, weights=weights)
        weighted_variance = np.average((np.array(values) - weighted_mean) ** 2, weights=weights)
        weighted_std_dev = weighted_variance ** 0.5
        
        # Check if value exceeds threshold
        return abs(value - weighted_mean) > self.threshold * weighted_std_dev
    
    def get_recent_anomalies(self, hours: int = 24) -> List[Dict]:
        """
        Get anomalies detected within the last N hours.        
        args:
            hours: Time window in hours            
        Returns:
            List of anomaly records
        """
        now = datetime.datetime.now()
        cutoff = now - datetime.timedelta(hours=hours)
        
        recent = [a for a in self.anomalies if a['timestamp'] > cutoff]
        return recent
    
    def get_metrics(self) -> Dict:
        """Get anomaly detection metrics."""
        return {
            'total_anomalies': len(self.anomalies),
            'recent_anomalies': len(self.get_recent_anomalies(24)),
            'observations_count': {k: len(v) for k, v in self.observations.items()}
        }


class MonitoringSystem:
    """Integrates with Prometheus and Grafana for monitoring and visualization."""
    
    def __init__(self, grafana_url: str = 'http://localhost:3000', 
                 prometheus_url: str = 'http://localhost:9090', 
                 auth: Tuple[str, str] = ('admin', 'admin')):
        """
        Initialize the monitoring system.        
        args:
            grafana_url: URL to Grafana server
            prometheus_url: URL to Prometheus server
            auth: Authentication credentials (username, password)
        """
        self.grafana_url = grafana_url
        self.prometheus_url = prometheus_url
        
        try:
            self.grafana = GrafanaApi(auth=auth, host=grafana_url)
        except Exception as e:
            logging.warning(f"Could not connect to Grafana: {e}")
            self.grafana = None
            
        # Setup Prometheus metrics. registry=REGISTRY is now part of mys muscle memory.
        self.metrics = {
            'cache_hits': prom.Counter('marksman_cache_hits', 'Number of cache hits', registry=REGISTRY),
            'cache_misses': prom.Counter('marksman_cache_misses', 'Number of cache misses', registry=REGISTRY),
            'cache_evictions': prom.Counter('marksman_cache_evictions', 'Number of cache evictions', registry=REGISTRY),
            'anomalies_detected': prom.Counter('marksman_anomalies', 'Number of anomalies detected', registry=REGISTRY),
            'prediction_accuracy': prom.Gauge('marksman_prediction_accuracy', 'Prediction accuracy percentage', registry=REGISTRY),
            'optimization_time': prom.Histogram('marksman_optimization_time', 'Time taken for optimization', buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0], registry=REGISTRY), 
            'resource_usage': {
                'cpu': prom.Gauge('marksman_cpu_usage', 'CPU utilization percentage', registry=REGISTRY),
                'memory': prom.Gauge('marksman_memory_usage', 'Memory utilization percentage', registry=REGISTRY),
                'network': prom.Gauge('marksman_network_usage', 'Network utilization percentage', registry=REGISTRY),
                'disk': prom.Gauge('marksman_disk_usage', 'Disk utilization percentage', registry=REGISTRY),
                'gpu': prom.Gauge('marksman_gpu_usage', 'GPU utilization percentage', registry=REGISTRY)
            }
        }
        
        # Start Prometheus HTTP server if not already running. remove this? this works now
        try:
            prom.start_http_server(8000)
            logging.info("Started Prometheus metrics server on port 8000")
        except Exception as e:
            logging.warning(f"Could not start Prometheus server: {e}")
    
    def log_cache_hit(self) -> None:
        self.metrics['cache_hits'].inc()
    
    def log_cache_miss(self) -> None:
        self.metrics['cache_misses'].inc()
    
    def log_cache_eviction(self) -> None:
        self.metrics['cache_evictions'].inc()
    
    def log_anomaly_detected(self) -> None:
        self.metrics['anomalies_detected'].inc()
    
    def set_prediction_accuracy(self, accuracy: float) -> None:
        self.metrics['prediction_accuracy'].set(accuracy)
    
    def observe_optimization_time(self, seconds: float) -> None:
        self.metrics['optimization_time'].observe(seconds)
    
    def set_resource_usage(self, resource_type: str, value: float) -> None:
        if resource_type in self.metrics['resource_usage']:
            self.metrics['resource_usage'][resource_type].set(value)
    
    def create_dashboard(self, title: str = "Marksman Performance Dashboard") -> Optional[int]:
        if self.grafana is None:
            return None
            
        try:
            # Simple dashboard with panels for cache, predictions, and resources
            dashboard = {
                "dashboard": {
                    "id": None,
                    "title": title,
                    "tags": ["marksman", "optimization", "cache"],
                    "timezone": "browser",
                    "panels": [
                        # Cache performance panel
                        {
                            "title": "Cache Performance",
                            "type": "graph",
                            "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
                            "targets": [
                                {"expr": "rate(marksman_cache_hits[5m])", "legendFormat": "Hits"},
                                {"expr": "rate(marksman_cache_misses[5m])", "legendFormat": "Misses"}
                            ]
                        },
                        # Prediction accuracy panel
                        {
                            "title": "Prediction Accuracy",
                            "type": "gauge",
                            "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8},
                            "targets": [
                                {"expr": "marksman_prediction_accuracy", "legendFormat": "Accuracy"}
                            ],
                            "options": {
                                "fieldOptions": {
                                    "max": 100,
                                    "min": 0
                                }
                            }
                        },
                        # Resource usage panel
                        {
                            "title": "Resource Usage",
                            "type": "graph",
                            "gridPos": {"x": 0, "y": 8, "w": 24, "h": 8},
                            "targets": [
                                {"expr": "marksman_cpu_usage", "legendFormat": "CPU"},
                                {"expr": "marksman_memory_usage", "legendFormat": "Memory"},
                                {"expr": "marksman_network_usage", "legendFormat": "Network"},
                                {"expr": "marksman_disk_usage", "legendFormat": "Disk"}
                            ]
                        },
                        # Anomalies panel
                        {
                            "title": "Anomalies Detected",
                            "type": "stat",
                            "gridPos": {"x": 0, "y": 16, "w": 12, "h": 8},
                            "targets": [
                                {"expr": "sum(increase(marksman_anomalies[24h]))"}
                            ]
                        }
                    ],
                    "refresh": "10s"
                },
                "overwrite": True
            }
            
            result = self.grafana.dashboard.update_dashboard(dashboard)
            return result.get('id')
            
        except Exception as e:
            logging.error(f"Failed to create Grafana dashboard: {e}")
            return None
    
    # update everything now
    def update_metrics(self, cache_metrics: Dict, anomaly_metrics: Dict, resource_metrics: Dict) -> None:
        # Update cache metrics
        if 'hits' in cache_metrics:
            self.metrics['cache_hits']._value.set(cache_metrics['hits'])            
        if 'misses' in cache_metrics:
            self.metrics['cache_misses']._value.set(cache_metrics['misses'])            
        if 'evictions' in cache_metrics:
            self.metrics['cache_evictions']._value.set(cache_metrics['evictions'])        
        if 'total_anomalies' in anomaly_metrics:
            self.metrics['anomalies_detected']._value.set(anomaly_metrics['total_anomalies'])
        for resource_type, value in resource_metrics.items():
            if resource_type in self.metrics['resource_usage']:
                self.metrics['resource_usage'][resource_type].set(value)

# Cherry pie. Markov Model for predicting developer actions and resource needs.
class MarkovModel:   
    def __init__(self, n_states: int = 10, alpha: float = 0.1):
        """
        Initialize the Markov model for state transition prediction.        
        args:
            n_states: Number of states in the Markov model
            alpha: Learning rate for updating transition probabilities
        """
        self.n_states = n_states
        self.alpha = alpha
        self.transition_matrix = np.zeros((n_states, n_states))
        self.state_mapping = {}  # Maps actions to state ids
        self.reverse_mapping = {}  # Mps state ids to actions
        self.state_count = 0
        self.last_state = None
        self.last_transition_time = None
    
    def _get_state_id(self, action: Dict) -> int:
        """
        Get or create (depending on presence) a state ID for an action.        
        args:
            action: dict containing action details            
        Returns:
            state ID
        """
        action_key = json.dumps(action, sort_keys=True)
        
        if action_key in self.state_mapping:
            return self.state_mapping[action_key]
        
        if self.state_count < self.n_states:
            # Create a new state
            state_id = self.state_count
            self.state_count += 1
            self.state_mapping[action_key] = state_id
            self.reverse_mapping[state_id] = action
            return state_id
        else:
            # Reuse the least frequently used state
            state_frequencies = np.sum(self.transition_matrix, axis=0)
            least_used_state = np.argmin(state_frequencies)
            
            # Update mappings
            for k, v in list(self.state_mapping.items()):
                if v == least_used_state:
                    del self.state_mapping[k]            
            self.state_mapping[action_key] = least_used_state
            self.reverse_mapping[least_used_state] = action
            
            # Reset transition probabilities for this state
            self.transition_matrix[:, least_used_state] = 0
            self.transition_matrix[least_used_state, :] = 0
            
            return least_used_state
    
    def update(self, action: Dict) -> None:
        """
        Update the Markov model with a new observed action.
        args:
            action: dict containing action details
        """
        current_time = datetime.datetime.now()
        current_state = self._get_state_id(action)
        
        if self.last_state is not None:
            # Update transition matrix with learning rate alpha
            self.transition_matrix[self.last_state, current_state] += self.alpha
            
            # Normalize the row to ensure it remains a probability distribution and not a. well.
            row_sum = np.sum(self.transition_matrix[self.last_state, :])
            if row_sum > 0:
                self.transition_matrix[self.last_state, :] /= row_sum
        
        self.last_state = current_state
        self.last_transition_time = current_time
    
    def predict_next_state(self, top_n: int = 3) -> List[Dict]:
        """
        Predict the most likely next states/actions.
        args:
            top_n: Number of top predictions to return
        Returns:
            list of predicted actions
        """
        if self.last_state is None:
            return []
        
        # Get probabilities for next states
        probs = self.transition_matrix[self.last_state, :]
        
        # Get top N states
        top_states = np.argsort(probs)[-top_n:][::-1]
        
        # Only include states with non-zero prob
        top_states = [s for s in top_states if probs[s] > 0]
        
        # Return the corr. actions
        return [self.reverse_mapping[s] for s in top_states if s in self.reverse_mapping]
    
    def save(self, filepath: str) -> None:
        model_data = {
            'transition_matrix': self.transition_matrix,
            'state_mapping': self.state_mapping,
            'reverse_mapping': self.reverse_mapping,
            'state_count': self.state_count,
            'n_states': self.n_states,
            'alpha': self.alpha
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.transition_matrix = model_data['transition_matrix']
        self.state_mapping = model_data['state_mapping']
        self.reverse_mapping = model_data['reverse_mapping']
        self.state_count = model_data['state_count']
        self.n_states = model_data['n_states']
        self.alpha = model_data['alpha']


class ResourcePredictor:    
    def __init__(self, n_clusters: int = 5):
        """  
        args:
            n_clusters: Number of resource usage clusters
        """
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.state_to_resources = defaultdict(list)
        self.fitted = False
    
    def update(self, state_id: int, resources: Dict[str, float]) -> None:
        """
        Update the resource predictor with observed resource usage.
        args:
            state_id: State ID from the Markov model
            resources: dict of resource metrics (CPU, memory, network, etc.)
        """
        resource_vector = [resources.get(k, 0) for k in ['cpu', 'memory', 'network', 'disk', 'gpu']]
        self.state_to_resources[state_id].append(resource_vector)
        
        # Retrain the model if we have enough data(we will not for ow)
        all_resources = []
        for resources_list in self.state_to_resources.values():
            all_resources.extend(resources_list)
        
        if len(all_resources) > self.n_clusters * 10:  #ensure enough data for clustering
            self.kmeans.fit(all_resources)
            self.fitted = True
    
    def predict_resources(self, state_id: int) -> Dict[str, float]:
        """
        Predict resource needs for a given state.
        
        args:
            state_id: State ID from the Markov model
            
        Returns:
            dict of predicted resource needs
        """
        if state_id not in self.state_to_resources or not self.fitted:
            # Retrn default values if we don't have data
            return {'cpu': 1.0, 'memory': 1.0, 'network': 1.0, 'disk': 1.0, 'gpu': 0.0}
        
        # Get resources associated with this state
        resources = self.state_to_resources[state_id]
        
        if not resources:
            return {'cpu': 1.0, 'memory': 1.0, 'network': 1.0, 'disk': 1.0, 'gpu': 0.0}
        
        #find the center of the cluster this resource usage belongs to
        resource_vector = np.mean(resources, axis=0)
        cluster = self.kmeans.predict([resource_vector])[0]
        cluster_center = self.kmeans.cluster_centers_[cluster]
        
        # Convert back to dict
        resource_keys = ['cpu', 'memory', 'network', 'disk', 'gpu']
        return {k: max(0.1, v) for k, v in zip(resource_keys, cluster_center)}


class CacheManager:
    """Manage predictive caching of resources based on Markov predictions."""    
    def __init__(self, cache_size_mb: int = 1000, decay_factor: float = 0.9):
        """
        args:
            cache_size_mb: Maximum cache size in MB
            decay_factor: Frequency decay factor for the cache eviction algorithm
        """
        self.cache = Cache(max_size_mb=cache_size_mb, decay_factor=decay_factor)
        self.resource_to_states = defaultdict(set)  # Maps resources to states that use them
        self.state_to_resources = defaultdict(set)  # Maps states to resources they use
        self.hit_prediction = {
            'correct': 0,
            'incorrect': 0,
            'predictions': []
        }
    
    def associate_resource_with_state(self, resource_id: str, state_id: int) -> None:
        """
        Associate a resource with a state to improve prediction.        
        args:
            resource_id: id for the resource
            state_id: State ID from the Markov model
        """
        self.resource_to_states[resource_id].add(state_id)
        self.state_to_resources[state_id].add(resource_id)
    
    def cache_resource(self, resource_id: str, data, size_mb: float) -> bool:
        """
        Cache a resource (if space allows)!!
        args:
            resource_id: id for the resource
            data: The resource data to cache
            size_mb: Size of the resource in MB            
        returns:
            voolean indicating whether caching was successful
        """
        return self.cache.put(resource_id, data, size_mb)
    
    def get_resource(self, resource_id: str) -> Optional[Any]:
        """
        Retrieve a resource from cache.        
        args:
            resource_id: id for the resource
        Returns:
            Cached resource data or None if not found
        """
        return self.cache.get(resource_id)
    
    def prefetch_resources(self, predicted_state_ids: List[int], fetcher_function, priority_resources: Optional[Set[str]] = None) -> Dict:
        """
        Prefetch resources likely to be needed based on predicted states.        
        args:
            predicted_state_ids: List of predicted state IDs from Markov model
            fetcher_function: Function to fetch a resource given its ID
            priority_resources: Set of resource IDs to prioritise (optional)
            
        Returns:
            dict with prefetch statistics
        """
        resources_to_fetch = set()
        stats = {
            'resources_identified': 0,
            'resources_fetched': 0,
            'fetch_failures': 0,
            'already_cached': 0,
            'size_fetched_mb': 0.0,
            'priority_fetched': 0
        }
        
        # Collect resources associated with predicted states
        for state_id in predicted_state_ids:
            state_resources = self.state_to_resources[state_id]
            resources_to_fetch.update(state_resources)
            stats['resources_identified'] += len(state_resources)
        
        # Check which resources are already cached
        not_cached = [r for r in resources_to_fetch if self.cache.get(r) is None]
        stats['already_cached'] = stats['resources_identified'] - len(not_cached)
        
        # Prioritize resources if specified
        if priority_resources:
            high_priority = [r for r in not_cached if r in priority_resources]
            low_priority = [r for r in not_cached if r not in priority_resources]
            not_cached = high_priority + low_priority
            stats['priority_fetched'] = len(high_priority)
        
        # Fetch and cache resources
        for resource_id in not_cached:
            try:
                data, size = fetcher_function(resource_id)
                if self.cache_resource(resource_id, data, size):
                    stats['resources_fetched'] += 1
                    stats['size_fetched_mb'] += size
                    
                    # Record prediction for accuracy tracking
                    self.hit_prediction['predictions'].append({
                        'resource_id': resource_id,
                        'timestamp': datetime.datetime.now(),
                        'hit': False  # Will be set to True if resource is accessed
                    })
            except Exception as e:
                logging.error(f"Failed to prefetch resource {resource_id}: {e}")
                stats['fetch_failures'] += 1
        
        # Trim prediction history
        if len(self.hit_prediction['predictions']) > 1000:
            self.hit_prediction['predictions'] = self.hit_prediction['predictions'][-1000:]
        
        return stats
    
    def record_resource_access(self, resource_id: str) -> None:
        """
        Record actual resource access to measure prediction accuracy.
        
        args:
            resource_id: id for the accessed resource
        """
        # Find this resource in recent predictions
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=1)
        
        for pred in self.hit_prediction['predictions']:
            if pred['resource_id'] == resource_id and pred['timestamp'] > cutoff_time:
                if not pred.get('hit', False):
                    pred['hit'] = True
                    self.hit_prediction['correct'] += 1
                return
            break
        
        # If not found in predictions, count as incorrect prediction
        self.hit_prediction['incorrect'] += 1
    
    def get_prediction_accuracy(self) -> float:
        """
        Calculate the accuracy of cache predictions.
        
        Returns:
            Prediction accuracy as a percentage
        """
        total = self.hit_prediction['correct'] + self.hit_prediction['incorrect']
        if total == 0:
            return 0.0
        return (self.hit_prediction['correct'] / total) * 100
    
    def get_metrics(self) -> Dict:
        """Get cache manager performance metrics."""
        cache_metrics = self.cache.get_metrics()
        prediction_metrics = {
            'prediction_accuracy': self.get_prediction_accuracy(),
            'predictions_made': len(self.hit_prediction['predictions']),
            'correct_predictions': self.hit_prediction['correct'],
            'incorrect_predictions': self.hit_prediction['incorrect']
        }
        
        return {**cache_metrics, **prediction_metrics}


class MarksmanSystem:
    """Main system that integrates all components for predictive resource optimization."""
    
    def __init__(self, cache_size_mb: int = 2000, n_markov_states: int = 20, 
                 monitoring_enabled: bool = True, model_path: Optional[str] = None):
        """
        Initialize the Marksman system.
        
        args:
            cache_size_mb: Maximum cache size in MB
            n_markov_states: Number of states in the Markov model
            monitoring_enabled: Whether to enable Prometheus/Grafana monitoring
            model_path: Path to load/save models (None for no persistence)
        """
        # Initialize components
        self.markov_model = MarkovModel(n_states=n_markov_states)
        self.resource_predictor = ResourcePredictor()
        self.cache_manager = CacheManager(cache_size_mb=cache_size_mb)
        self.anomaly_detector = AnomalyDetector()
        self.model_path = model_path
        
        # Initialize monitoring if enabled
        self.monitoring = None
        if monitoring_enabled:
            self.monitoring = MonitoringSystem()
            
        # Load models if path provided
        if model_path and os.path.exists(model_path):
            self._load_models()
        
        # Track performance metrics
        self.metrics = {
            'actions_processed': 0,
            'resources_cached': 0,
            'predictions_made': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'prediction_accuracy': 0.0,
            'anomalies_detected': 0
        }
    
    def process_action(self, action: Dict, resources: Optional[Dict[str, float]] = None) -> Dict:
        """
        Process a developer action and update predictions.
        
        args:
            action: dict containing action details
            resources: Current resource usage (optional)
            
        Returns:
            dict with processing results and predictions
        """
        # Update metrics
        self.metrics['actions_processed'] += 1
        
        # Update Markov model
        self.markov_model.update(action)
        
        # Update resource predictor if resource usage provided
        if resources and self.markov_model.last_state is not None:
            self.resource_predictor.update(self.markov_model.last_state, resources)
            
            # Check for resource usage anomalies
            for resource_type, value in resources.items():
                if self.anomaly_detector.add_observation(resource_type, value):
                    self.metrics['anomalies_detected'] += 1
                    if self.monitoring:
                        self.monitoring.log_anomaly_detected()
        
        # Predict next states and resources
        predicted_states = self.markov_model.predict_next_state(top_n=3)
        self.metrics['predictions_made'] += 1
        
        # Prepare result
        result = {
            'predicted_next_actions': predicted_states,
            'current_state': self.markov_model.last_state,
            'resource_predictions': {}
        }        
        # Predict resources for each predicted state
        for i, state in enumerate(predicted_states):
            state_id = self.markov_model.state_mapping.get(json.dumps(state, sort_keys=True))
            if state_id is not None:
                result['resource_predictions'][f'state_{i}'] = self.resource_predictor.predict_resources(state_id)        
        # Update monitoring
        if self.monitoring:
            self.monitoring.set_prediction_accuracy(self.cache_manager.get_prediction_accuracy())
            if resources:
                for resource_type, value in resources.items():
                    self.monitoring.set_resource_usage(resource_type, value)
        
        return result
    
    def fetch_resource(self, resource_id: str, fetcher_function) -> Any:
        """
        Fetch a resource with caching.        
        args:
            resource_id: id for the resource
            fetcher_function: Function to fetch the resource if not cached
            
        Returns:
            The resource data
        """
        # Try to get from cache
        data = self.cache_manager.get_resource(resource_id)
        
        if data is not None:
            # Cache hit
            self.metrics['cache_hits'] += 1
            if self.monitoring:
                self.monitoring.log_cache_hit()
                
            # Record this access for predicn accuracy
            self.cache_manager.record_resource_access(resource_id)
            return data
        
        # Cache miss
        self.metrics['cache_misses'] += 1
        if self.monitoring:
            self.monitoring.log_cache_miss()
        
        #Fetch the resource
        try:
            data, size = fetcher_function(resource_id)            
            # Cache it for future use
            if self.cache_manager.cache_resource(resource_id, data, size):
                self.metrics['resources_cached'] += 1
                
            return data            
        except Exception as e:
            logging.error(f"Failed to fetch resource {resource_id}: {e}")
            raise
    
    def prefetch_resources(self, fetcher_function, priority_resources: Optional[Set[str]] = None) -> Dict:
        """
        Prefetch resources likely to be needed soon.
        args:
            fetcher_function: Function to fetch a resource given its ID
            priority_resources: Set of resource IDs to prioritise (optional)            
        Returns:
            Prefetch stats
        """
        # get predicted states
        predicted_states = [self.markov_model.last_state] if self.markov_model.last_state is not None else []
        predicted_next = self.markov_model.predict_next_state(top_n=5)
        
        for state in predicted_next:
            state_id = self.markov_model.state_mapping.get(json.dumps(state, sort_keys=True))
            if state_id is not None:
                predicted_states.append(state_id)
        
        # Prefetch resources for these states
        return self.cache_manager.prefetch_resources(predicted_states, fetcher_function, priority_resources)
    
    def associate_resource_with_action(self, resource_id: str, action: Dict) -> None:
        """
        Associate a resource with a specific action for improved prediction.        
        args:
            resource_id: id for the resource
            action: dict containing action details
        """
        state_id = self.markov_model._get_state_id(action)
        self.cache_manager.associate_resource_with_state(resource_id, state_id)
    
    def get_anomalies(self, hours: int = 24) -> List[Dict]:
        """
        Get recent anomalies detected in rsrc usage.        
        args:
            hours: Time window in hours
            
        Returns:
            List of anomaly records
        """
        return self.anomaly_detector.get_recent_anomalies(hours)
    
    def get_metrics(self) -> Dict:
        cache_metrics = self.cache_manager.get_metrics()
        anomaly_metrics = self.anomaly_detector.get_metrics()
        
        combined_metrics = {
            **self.metrics,
            **cache_metrics,
            **anomaly_metrics,
            'total_states': self.markov_model.state_count,
        }        
        
        if self.monitoring:
            resource_metrics = {k: v for k, v in combined_metrics.items() if k in ['cpu', 'memory', 'network', 'disk', 'gpu']}
            self.monitoring.update_metrics(cache_metrics, anomaly_metrics, resource_metrics)
        
        return combined_metrics
    
    def _save_models(self) -> None:
        if not self.model_path:
            return
            
        os.makedirs(self.model_path, exist_ok=True)
        
        # Save Markov model
        self.markov_model.save(os.path.join(self.model_path, 'markov_model.pkl'))
        
        # Save other model data
        model_data = {
            'resource_predictor': self.resource_predictor,
            'cache_manager': self.cache_manager,
            'anomaly_detector': self.anomaly_detector,
            'metrics': self.metrics
        }
        
        with open(os.path.join(self.model_path, 'marksman_models.pkl'), 'wb') as f:
            pickle.dump(model_data, f)
    
    def _load_models(self) -> None:
        if not self.model_path:
            return
            
        markov_path = os.path.join(self.model_path, 'markov_model.pkl')
        if os.path.exists(markov_path):
            self.markov_model.load(markov_path)            
        models_path = os.path.join(self.model_path, 'marksman_models.pkl')
        if os.path.exists(models_path):
            with open(models_path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.resource_predictor = model_data['resource_predictor']
            self.cache_manager = model_data['cache_manager']
            self.anomaly_detector = model_data['anomaly_detector']
            self.metrics = model_data['metrics']
    
    def create_dashboard(self) -> Optional[int]:
        if self.monitoring:
            return self.monitoring.create_dashboard("Marksman System Performance")
        return None
    
    def shutdown(self) -> None:
        self._save_models()
        logging.info(f"Marksman System shutting down. Final metrics: {self.get_metrics()}")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    marksman = MarksmanSystem(cache_size_mb=500, model_path="./models")
    action = {"type": "open_file", "file": "main.py", "project": "marksman"}
    resources = {"cpu": 10.5, "memory": 30.2, "network": 5.1, "disk": 12.0, "gpu": 0.0}
    
    result = marksman.process_action(action, resources)
    logging.info(f"Processed action with result: {result}")
    
    def fetch_mock(resource_id):
        # In real application, this would fetch actual resources
        data = f"Resource content for {resource_id}"
        size = len(data) / 1024.0  # Size in MB
        return data, size    
    resource_data = marksman.fetch_resource("main.py", fetch_mock)
    logging.info(f"Fetched resource: {resource_data}")
    
    # Example prefetch
    prefetch_stats = marksman.prefetch_resources(fetch_mock)
    logging.info(f"Prefetch stats: {prefetch_stats}")
    
    dashboard_id = marksman.create_dashboard()
    if dashboard_id:
        logging.info(f"Created dashboard with ID: {dashboard_id}")
    
    # Get metrics
    metrics = marksman.get_metrics()
    logging.info(f"System metrics: {metrics}")
    
    # Proper shutdown
    marksman.shutdown()