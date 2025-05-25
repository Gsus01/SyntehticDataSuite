import pytest
from src.core.model_selector import select_model

def test_select_model_auto():
    config = {"model": "auto"}
    assert select_model(config) == "dummy_time_series_model"

def test_select_model_manual_specific_model():
    model_name = "my_custom_model"
    config = {"model": model_name}
    assert select_model(config) == model_name

def test_select_model_manual_another_model():
    model_name = "another_model_123"
    config = {"model": model_name}
    assert select_model(config) == model_name

def test_select_model_empty_config():
    # Even if config is empty, it defaults to "auto" behavior
    config = {}
    assert select_model(config) == "dummy_time_series_model"

def test_select_model_config_none():
    # If model key is present but None, it should also default to "auto"
    config = {"model": None}
    assert select_model(config) == "dummy_time_series_model"
