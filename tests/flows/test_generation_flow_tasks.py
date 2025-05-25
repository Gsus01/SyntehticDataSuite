import pytest
import os
import shutil
import pandas as pd
from pathlib import Path
from src.flows.generation import generate_dummy_data

@pytest.fixture(scope="function")
def setup_generation_test_environment(tmp_path):
    # The task generate_dummy_data writes to ./data/synthetic_output.csv
    # We need to ensure ./data exists and is writable for the test.
    original_data_dir = Path("./data")
    original_output_file = original_data_dir / "synthetic_output.csv"

    # Ensure ./data directory exists
    original_data_dir.mkdir(parents=True, exist_ok=True)

    # Dummy model path (not used by current dummy generation logic but required by signature)
    dummy_model_path = str(tmp_path / "dummy_model.pkl")
    Path(dummy_model_path).touch()

    yield {
        "model_path": dummy_model_path,
        "output_file_path": original_output_file,
        "data_dir": original_data_dir
    }

    # Teardown: remove the ./data/synthetic_output.csv created by the test
    if original_output_file.exists():
        os.remove(original_output_file)
    # Clean up ./data directory if empty
    try:
        if not os.listdir(original_data_dir):
            os.rmdir(original_data_dir)
    except OSError:
        pass # Ignore if not empty or other issues


def test_generate_dummy_data(setup_generation_test_environment):
    model_path = setup_generation_test_environment["model_path"]
    output_file_path = setup_generation_test_environment["output_file_path"]
    n_samples = 150

    returned_output_path = generate_dummy_data(model_path, n_samples)

    assert returned_output_path == str(output_file_path)
    assert output_file_path.exists()
    assert output_file_path.is_file()

    # Verify content of the CSV
    df = pd.read_csv(output_file_path)
    assert len(df) == n_samples
    assert "timestamp" in df.columns
    assert "value" in df.columns
    
    # Check if timestamp is datetime like
    try:
        pd.to_datetime(df["timestamp"])
    except Exception:
        pytest.fail("Timestamp column is not in a valid datetime format.")

    # Check some value properties if needed, e.g.
    assert df['value'].iloc[0] == 0.0 
    # Example: value = index + index * 0.1. For index 0, value = 0. For index 1, value = 1 + 0.1 = 1.1
    if n_samples > 1:
        assert df['value'].iloc[1] == 1.1 


def test_generate_dummy_data_different_samples(setup_generation_test_environment):
    model_path = setup_generation_test_environment["model_path"]
    output_file_path = setup_generation_test_environment["output_file_path"]
    n_samples = 75

    returned_output_path = generate_dummy_data(model_path, n_samples)

    assert returned_output_path == str(output_file_path)
    assert output_file_path.exists()

    df = pd.read_csv(output_file_path)
    assert len(df) == n_samples
    assert "timestamp" in df.columns
    assert "value" in df.columns
