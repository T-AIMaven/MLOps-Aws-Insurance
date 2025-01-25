import json
import os
import shutil
import tarfile
import tempfile
import warnings
from pathlib import Path

import pytest

from src.evaluation.script import evaluate
from src.processing.script import preprocess
from src.training.script import train

# Define the path to the clean insurance data file
DATA_FILEPATH = Path(__file__).parent.parent / "data" / "clean" / "clean-insurance.csv"


@pytest.fixture(scope="function", autouse=False)
def directory():
    """
    Pytest fixture to set up and tear down a temporary directory for testing.

    This fixture creates a temporary directory, copies the clean insurance data file to the input directory,
    runs preprocessing, training, and evaluation scripts, and yields the evaluation directory path.
    It also handles cleanup by removing the temporary directory after tests are run.

    Yields:
        Path: The path to the evaluation directory containing the evaluation report.
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

        with tarfile.open(directory / "model.tar.gz", "w:gz") as tar:
            tar.add(directory / "model" / "001", arcname="001")

        evaluate(
            model_path=directory,
            test_path=directory / "test",
            output_path=directory / "evaluation",
        )

    yield directory / "evaluation"

    shutil.rmtree(directory)


def test_evaluate_generates_evaluation_report(directory):
    """
    Test to verify that the evaluation report (evaluation.json) is generated.

    Args:
        directory (Path): The path to the evaluation directory containing the evaluation report.
    """
    output = os.listdir(directory)
    assert "evaluation.json" in output


def test_evaluation_report_contains_metrics(directory):
    """
    Test to verify that the evaluation report contains the expected metrics.

    Args:
        directory (Path): The path to the evaluation directory containing the evaluation report.
    """
    with open(directory / "evaluation.json", "r", encoding="utf-8") as file:
        report = json.load(file)

    assert "metrics" in report
    assert "mean_squared_error" in report["metrics"]
    assert "r2_score" in report["metrics"]
    assert "mean_absolute_error" in report["metrics"]
