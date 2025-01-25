import os
import shutil
import tempfile
import warnings
from pathlib import Path

import pytest

from src.processing.script import preprocess
from src.training.script import train

# Define the path to the clean insurance data file
DATA_FILEPATH = Path(__file__).parent.parent / "data" / "clean" / "clean-insurance.csv"


@pytest.fixture(autouse=False)
def directory():
    """
    Pytest fixture to set up a temporary directory for testing.

    This fixture creates a temporary directory structure and copies the data
    into it. It then runs the preprocess and train functions from the
    respective scripts. After the test, it cleans up the temporary directory.

    Yields:
        directory (Path): Path object of the temporary directory.
    """

    # Create a temporary directory
    directory = tempfile.mkdtemp()
    input_directory = Path(directory) / "input"
    input_directory.mkdir(parents=True, exist_ok=True)

    # Check if the data file exists
    if not DATA_FILEPATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILEPATH}")

    # Copy the data file to the input directory
    shutil.copy2(DATA_FILEPATH, input_directory / "data.csv")

    directory = Path(directory)

    # Suppress deprecation warnings and run preprocess and train functions
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        preprocess(base_directory=directory)
        train(
            model_directory=directory / "model",
            train_path=directory / "train",
            validation_path=directory / "validation",
            pipeline_path=directory / "model",
            experiment=None,
            epochs=10,
        )

    # Yield the directory path for use in tests
    yield directory

    # Clean up the temporary directory
    shutil.rmtree(directory)


def test_train_bundles_model_assets(directory):
    """
    Test to verify that the model assets are bundled correctly.

    This test checks that the trained model assets are saved in the expected directory
    structure after the training process.

    Args:
        directory (Path): Path object of the temporary directory.
    """
    bundle = os.listdir(directory / "model")
    assert "001" in bundle

    assets = os.listdir(directory / "model" / "001")
    assert "saved_model.pb" in assets


def test_train_bundles_transformation_pipelines(directory):
    """
    Test to verify that the transformation pipelines are bundled correctly.

    This test checks that the transformation pipelines are saved in the expected
    directory structure after the training process.

    Args:
        directory (Path): Path object of the temporary directory.
    """
    bundle = os.listdir(directory / "model")
    assert "features.joblib" in bundle
