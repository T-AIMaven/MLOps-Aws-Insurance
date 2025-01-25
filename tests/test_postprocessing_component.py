import json

import pytest

from src.pipeline.postprocessing_component import input_fn, output_fn


def test_input_fn_raises_error_for_unsupported_content_type():
    input_data = json.dumps({"predictions": [100, 200]})
    with pytest.raises(ValueError):
        input_fn(input_data, "text/csv")


def test_input_fn_parses_json():
    input_data = json.dumps({"predictions": [100, 200]})
    parsed_data = input_fn(input_data, "application/json")
    assert parsed_data == [100, 200]


def test_output_does_not_return_array_if_single_prediction():
    prediction = [100]
    response = output_fn(prediction, "application/json")
    expected_response = (json.dumps({"prediction": 100}), "application/json")
    assert response == expected_response


def test_output_returns_array_if_multiple_predictions():
    prediction = [100, 200]
    response = output_fn(prediction, "application/json")
    expected_response = (json.dumps([{"prediction": 100}, {"prediction": 200}]), "application/json")
    assert response == expected_response


def test_output_returns_csv_format():
    prediction = [100, 200]
    response = output_fn(prediction, "text/csv")
    expected_response = ("100\n200\n", "text/csv")
    assert response == expected_response
