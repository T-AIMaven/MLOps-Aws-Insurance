# | filename: postprocessing_component.py
# | code-line-numbers: true

import json
import os

import joblib

try:
    from sagemaker_containers.beta.framework import worker
except ImportError:
    # We don't have access to the `worker` package when testing locally.
    # We'll set it to None so we can change the way functions create
    # a response.
    worker = None


def model_fn(model_dir):
    """
    Deserializes the model that will be used in this container.
    """

    return joblib.load(os.path.join(model_dir, "features.joblib"))


def input_fn(input_data, content_type):
    if content_type == "application/json":
        return json.loads(input_data)["predictions"]
    raise ValueError(f"{content_type} is not supported.")


def predict_fn(input_data, model):
    """
    Transforms the prediction into its corresponding category.
    """
    predictions = input_data
    return predictions


def output_fn(prediction, accept):
    if accept == "text/csv":
        # Convert predictions to a flat list of numbers
        if isinstance(prediction, list) and all(isinstance(p, list) for p in prediction):
            flat_predictions = [p[0] for p in prediction]
        else:
            flat_predictions = prediction

        csv_output = "\n".join(map(str, flat_predictions)) + "\n"
        return worker.Response(csv_output, mimetype=accept) if worker else (csv_output, accept)

    if accept == "application/json":
        response = [{"prediction": p} for p in prediction]

        # If there's only one prediction, we'll return it
        # as a single object.
        if len(response) == 1:
            response = response[0]

        return worker.Response(json.dumps(response), mimetype=accept) if worker else (json.dumps(response), accept)

    raise Exception(f"{accept} accept type is not supported.")
