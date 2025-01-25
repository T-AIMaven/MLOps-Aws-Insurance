import os
import shutil
import tarfile
import tempfile
import warnings
from pathlib import Path

import pandas as pd
import pytest

from src.processing.script import preprocess

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

    # Suppress deprecation warnings and run the preprocess function
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        preprocess(base_directory=directory)

    # Yield the directory path for use in tests
    yield directory

    # Clean up the temporary directory
    shutil.rmtree(directory)


def test_preprocess_generates_data_splits(directory):
    """
    Test that the preprocess function generates the correct data splits.

    This test checks that the preprocess function creates the expected directories
    for train, validation, and test data splits.

    Args:
        directory (Path): Path object of the temporary directory.
    """

    output_directories = os.listdir(directory)

    assert "train" in output_directories
    assert "validation" in output_directories
    assert "test" in output_directories


def test_preprocess_generates_baselines(directory):
    """
    Test that the preprocess function generates the correct baselines.

    This test checks that the preprocess function creates the expected directories
    for train and test baselines.

    Args:
        directory (Path): Path object of the temporary directory.
    """

    output_directories = os.listdir(directory)

    assert "train-baseline" in output_directories
    assert "test-baseline" in output_directories


def test_preprocess_creates_models(directory):
    """
    Test that the preprocess function creates the model tar.gz file.

    This test checks that the preprocess function creates a tar.gz file containing
    the model features.

    Args:
        directory (Path): Path object of the temporary directory.
    """

    model_path = directory / "model"
    tar = tarfile.open(model_path / "model.tar.gz", "r:gz")

    assert "features.joblib" in tar.getnames()


def test_preprocess_directories(directory):
    """
    Test that the preprocess function creates the necessary directories and files.

    This test checks that the preprocess function creates non-empty CSV files
    for train, validation, test, train-baseline, and test-baseline data, and
    a model tar.gz file.

    Args:
        directory (Path): Path object of the temporary directory.
    """

    assert (directory / "train" / "train.csv").stat().st_size > 0
    assert (directory / "test" / "test.csv").stat().st_size > 0
    assert (directory / "validation" / "validation.csv").stat().st_size > 0
    assert (directory / "train-baseline" / "train-baseline.csv").stat().st_size > 0
    assert (directory / "test-baseline" / "test-baseline.csv").stat().st_size > 0
    assert (directory / "model" / "model.tar.gz").exists()


def test_splits_are_transformed(directory):
    """
    Test that the data splits are correctly transformed.

    This test checks that the data splits (train, validation, test) have been
    transformed correctly by verifying the number of features in the CSV files.

    Args:
        directory (Path): Path object of the temporary directory.
    """

    train = pd.read_csv(directory / "train" / "train.csv", header=None)
    validation = pd.read_csv(directory / "validation" / "validation.csv", header=None)
    test = pd.read_csv(directory / "test" / "test.csv", header=None)

    # Expected number of features after transformation
    number_of_features = 11

    # Verify the number of features plus the target variable column
    assert train.shape[1] == number_of_features + 1
    assert validation.shape[1] == number_of_features + 1
    assert test.shape[1] == number_of_features + 1


def test_train_baseline_is_not_transformed(directory):
    """
    Test that the train baseline data is not transformed.

    This test checks that the train baseline data has not been transformed by
    verifying the presence of original categorical values.

    Args:
        directory (Path): Path object of the temporary directory.
    """

    baseline = pd.read_csv(directory / "train-baseline" / "train-baseline.csv", header=None)
    sex = baseline.iloc[:, 1].unique()

    assert "male" in sex
    assert "female" in sex


def test_test_baseline_is_not_transformed(directory):
    """
    Test that the test baseline data is not transformed.

    This test checks that the test baseline data has not been transformed by
    verifying the presence of original categorical values.

    Args:
        directory (Path): Path object of the temporary directory.
    """

    baseline = pd.read_csv(directory / "test-baseline" / "test-baseline.csv", header=None)
    sex = baseline.iloc[:, 1].unique()

    assert "male" in sex
    assert "female" in sex


def test_train_baseline_includes_header(directory):
    """
    Test that the train baseline data includes a header.

    This test checks that the train baseline CSV file includes a header row.

    Args:
        directory (Path): Path object of the temporary directory.
    """

    baseline = pd.read_csv(directory / "train-baseline" / "train-baseline.csv")
    assert baseline.columns[0] == "age"


def test_test_baseline_does_not_include_header(directory):
    """
    Test that the test baseline data does not include a header.

    This test checks that the test baseline CSV file does not include a header row.

    Args:
        directory (Path): Path object of the temporary directory.
    """

    baseline = pd.read_csv(directory / "test-baseline" / "test-baseline.csv")
    assert baseline.columns[0] != "age"
