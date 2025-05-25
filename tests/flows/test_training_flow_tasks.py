import pytest
import os
import shutil
from pathlib import Path
from src.flows.training import train_dummy_model

@pytest.fixture(scope="function")
def setup_training_test_environment(tmp_path):
    # Temporary directory for model registry for this test
    model_registry_root = tmp_path / "model_registry"
    model_registry_root.mkdir()

    # The task train_dummy_model creates models under ./model_registry/
    # We need to ensure this path is writable for the test by effectively
    # redirecting it or ensuring the test runs where ./model_registry is safe to write.
    # For this test, we'll let it write to the actual relative path ./model_registry
    # and clean up afterwards.

    original_model_registry_path = Path("./model_registry")

    # Ensure the ./model_registry directory exists or is cleaned if it's from a previous run
    if original_model_registry_path.exists():
        shutil.rmtree(original_model_registry_path)
    original_model_registry_path.mkdir(parents=True, exist_ok=True)

    # Dummy data path (though not strictly used by train_dummy_model's logic, it's an input)
    data_path = str(tmp_path / "dummy_data.csv")
    Path(data_path).touch() # Create an empty dummy file

    yield {
        "data_path": data_path,
        "model_registry_path": original_model_registry_path # Test will write here
    }

    # Teardown: remove the ./model_registry created by the test
    if original_model_registry_path.exists():
        shutil.rmtree(original_model_registry_path, ignore_errors=True)

def test_train_dummy_model(setup_training_test_environment):
    data_path = setup_training_test_environment["data_path"]
    model_registry_path = setup_training_test_environment["model_registry_path"]
    model_name = "test_dummy_model"

    returned_model_path = train_dummy_model(data_path, model_name)

    expected_model_dir = model_registry_path / model_name
    expected_model_artifact_path = expected_model_dir / "dummy_model.pkl"

    assert returned_model_path == str(expected_model_artifact_path)
    assert expected_model_dir.exists()
    assert expected_model_dir.is_dir()
    assert expected_model_artifact_path.exists()
    assert expected_model_artifact_path.is_file()

    with open(expected_model_artifact_path, 'r') as f:
        content = f.read()
    assert content == "This is a dummy model artifact."

def test_train_dummy_model_another_name(setup_training_test_environment):
    data_path = setup_training_test_environment["data_path"]
    model_registry_path = setup_training_test_environment["model_registry_path"]
    model_name = "another_test_model_456"

    returned_model_path = train_dummy_model(data_path, model_name)

    expected_model_dir = model_registry_path / model_name
    expected_model_artifact_path = expected_model_dir / "dummy_model.pkl"

    assert returned_model_path == str(expected_model_artifact_path)
    assert expected_model_dir.exists()
    assert expected_model_artifact_path.exists()
