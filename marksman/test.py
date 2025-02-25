import pytest
from unittest.mock import MagicMock
from .marksman import MarksmanSystem, AdvancedCache
import prometheus_client
import time

def test_cache_put_get():
    cache = AdvancedCache(max_size_mb=100)
    cache.put("test_key", "test_data", 10)
    assert cache.get("test_key") == "test_data"

def test_cache_eviction():
    cache = AdvancedCache(max_size_mb=100)
    cache.put("key1", "data1", 60)
    cache.put("key2", "data2", 50)  # Should trigger eviction
    assert cache.get("key1") is None  # Should be evicted
    assert cache.get("key2") == "data2"

def test_action_processing_and_prediction():
    marksman = MarksmanSystem(cache_size_mb=100, monitoring_enabled=False)
    action = {"type": "open_file", "file": "test.py"}
    result = marksman.process_action(action)
    assert "predicted_next_actions" in result

def test_resource_fetching_and_caching():
    marksman = MarksmanSystem(cache_size_mb=100)
    mock_fetcher = lambda resource_id: (f"Data for {resource_id}", 1.0)
    
    data1 = marksman.fetch_resource("test.py", mock_fetcher)
    metrics1 = marksman.get_metrics()
    data2 = marksman.fetch_resource("test.py", mock_fetcher)
    metrics2 = marksman.get_metrics()
    
    assert metrics2["cache_hits"] == metrics1["cache_hits"] + 1

def test_end_to_end():
    prometheus_client.REGISTRY = prometheus_client.CollectorRegistry(auto_describe=True)
    marksman = MarksmanSystem(cache_size_mb=1000, model_path="./e2e_models")
    workflows = [[({"type": "open_file", "file": "main.cpp"}, {"cpu": 10, "memory": 200})]]
    mock_fetcher = lambda resource_id: (f"Content of {resource_id}", len(resource_id))
    
    for _ in range(10):
        for workflow in workflows:
            for action, resources in workflow:
                marksman.process_action(action, resources)
                if "file" in action:
                    marksman.fetch_resource(action["file"], mock_fetcher)
    
    action = {"type": "open_file", "file": "main.cpp"}
    result = marksman.process_action(action, {"cpu": 10, "memory": 200})
    assert "predicted_next_actions" in result