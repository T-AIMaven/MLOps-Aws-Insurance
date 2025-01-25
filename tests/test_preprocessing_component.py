import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.pipeline.preprocessing_component import (
    input_fn,
    model_fn,
    output_fn,
    predict_fn,
)
from src.processing.script import preprocess

# Define the path to the clean insurance data file
DATA_FILEPATH = Path(__file__).parent.parent / "data" / "clean" / "clean-insurance.csv"


@pytest.fixture(scope="function", autouse=False)
def directory():
    directory = tempfile.mkdtemp()
    input_directory = Path(directory) / "input"
    input_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DATA_FILEPATH, input_directory / "data.csv")

    directory = Path(directory)

    preprocess(base_directory=directory)

    with tarfile.open(directory / "model" / "model.tar.gz") as tar:
        tar.extractall(path=directory / "model")

    yield directory / "model"

    shutil.rmtree(directory)


def test_input_csv_drops_target_column_if_present():
    input_data = """
    18, male, 33.7, 1, no, southeast, 16884.924
    """

    df = input_fn(input_data, "text/csv")
    assert len(df.columns) == 6 and "species" not in df.columns


def test_input_json_drops_target_column_if_present():
    input_data = json.dumps(
        {"age": 18, "sex": "male", "bmi": 33.7, "children": 1, "smoker": "no", "region": "southeast", "charges": 16884.924}
    )

    df = input_fn(input_data, "application/json")
    assert len(df.columns) == 6 and "species" not in df.columns


def test_input_csv_works_without_target_column():
    input_data = """
    18, male, 33.7, 1, no, southeast
    """

    df = input_fn(input_data, "text/csv")
    assert len(df.columns) == 6


def test_input_json_works_without_target_column():
    input_data = json.dumps(
        {
            "age": 18,
            "sex": "male",
            "bmi": 33.7,
            "children": 1,
            "smoker": "no",
            "region": "southeast",
        }
    )

    df = input_fn(input_data, "application/json")
    assert len(df.columns) == 6


def test_output_raises_exception_if_prediction_is_none():
    with pytest.raises(Exception):
        output_fn(None, "application/json")


def test_output_returns_tensorflow_ready_input():
    prediction = np.array(
        [
            [-1.432172577, -0.039782606, -0.911097544, 0, 1, 0, 1, 0, 0, 1, 0],
            [1.66234317, -1.253723351, -0.911097544, 1, 0, 1, 0, 1, 0, 0, 0],
        ]
    )

    response = output_fn(prediction, "application/json")

    assert response[0] == {
        "instances": [
            [-1.432172577, -0.039782606, -0.911097544, 0, 1, 0, 1, 0, 0, 1, 0],
            [1.66234317, -1.253723351, -0.911097544, 1, 0, 1, 0, 1, 0, 0, 0],
        ]
    }

    assert response[1] == "application/json"


def test_predict_transforms_data(directory):
    input_data = """
    18, male, 33.7, 1, no, southeast
    """

    model = model_fn(directory.as_posix())
    df = input_fn(input_data, "text/csv")
    response = predict_fn(df, model)
    assert type(response) is np.ndarray


def test_predict_returns_none_if_invalid_input(directory):
    input_data = """
    18, male, 33.7, 1, no, Invalid
    """

    model = model_fn(directory.as_posix())
    df = input_fn(input_data, "text/csv")
    assert predict_fn(df, model) is None
