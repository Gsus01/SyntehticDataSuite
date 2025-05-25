import pytest
import os
import shutil
from pathlib import Path
from src.flows.main import preprocess_data, check_if_model_exists

# Helper function to create dummy files for testing
def create_dummy_file(filepath, content="dummy content"):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)

@pytest.fixture(scope="function")
def setup_main_flow_test_environment(tmp_path):
    # Create temporary data and model registry directories
    raw_data_dir = tmp_path / "data"
    raw_data_dir.mkdir()
    model_reg_dir = tmp_path / "model_registry"
    model_reg_dir.mkdir()

    # Original paths for tasks
    original_raw_data_path = "./data/raw_input.csv" # As used in preprocess_data
    original_processed_data_path = "./data/processed_input.csv" # As used in preprocess_data
    original_model_registry_path = "./model_registry" # As used in check_if_model_exists

    # Create dummy raw input file in the expected location for preprocess_data
    # This task copies from ./data/raw_input.csv to ./data/processed_input.csv
    # So we need to ensure these paths are writable and exist for the test.
    # We'll use the actual relative paths for these specific tasks as they are hardcoded.

    # Ensure ./data directory exists for raw and processed files
    Path("./data").mkdir(parents=True, exist_ok=True)
    dummy_raw_content = "timestamp,value\n2023-01-01,10"
    create_dummy_file(original_raw_data_path, dummy_raw_content)
    
    yield {
        "tmp_path": tmp_path,
        "raw_data_path": original_raw_data_path,
        "processed_data_path": original_processed_data_path,
        "model_registry": original_model_registry_path,
        "dummy_raw_content": dummy_raw_content
    }

    # Teardown: Remove created files and directories
    if Path(original_raw_data_path).exists():
        os.remove(original_raw_data_path)
    if Path(original_processed_data_path).exists():
        os.remove(original_processed_data_path)
    # Clean up ./data directory if empty, handle potential errors if other files exist
    try:
        if not os.listdir("./data"):
            os.rmdir("./data")
    except OSError:
        pass # Ignore if not empty or other issues, focus on primary test artifacts

    # Clean up model registry for check_if_model_exists tests
    # This test creates models in ./model_registry/{model_name}/model.pkl
    if Path(original_model_registry_path).exists():
        shutil.rmtree(original_model_registry_path, ignore_errors=True)


def test_preprocess_data(setup_main_flow_test_environment):
    raw_path = setup_main_flow_test_environment["raw_data_path"]
    processed_path_expected = setup_main_flow_test_environment["processed_data_path"]
    dummy_content = setup_main_flow_test_environment["dummy_raw_content"]

    returned_path = preprocess_data(raw_path)
    assert returned_path == processed_path_expected
    assert Path(processed_path_expected).exists()
    with open(processed_path_expected, 'r') as f:
        content = f.read()
    assert content == dummy_content

def test_check_if_model_exists_when_it_does(setup_main_flow_test_environment):
    model_name = "test_model_exists"
    model_registry_base = setup_main_flow_test_environment["model_registry"] # ./model_registry
    
    # Create a dummy model file
    model_dir = Path(model_registry_base) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    dummy_model_path = model_dir / "model.pkl" # Corrected from "model.pkl" to "dummy_model.pkl" as per training task
    create_dummy_file(dummy_model_path, "dummy model content")
    
    # The function expects model.pkl, not dummy_model.pkl. Let's stick to model.pkl for this test.
    # Re-creating the dummy model with the name check_if_model_exists expects
    expected_model_path = model_dir / "model.pkl"
    create_dummy_file(expected_model_path, "dummy model content for check")


    returned_path = check_if_model_exists(model_name)
    assert returned_path == str(expected_model_path)
    assert Path(returned_path).exists()

def test_check_if_model_exists_when_it_does_not(setup_main_flow_test_environment):
    model_name = "non_existent_model"
    # Ensure model registry exists but model does not
    Path(setup_main_flow_test_environment["model_registry"]).mkdir(parents=True, exist_ok=True)

    returned_path = check_if_model_exists(model_name)
    assert returned_path is None
