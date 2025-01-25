# | filename: app.py
# | code-line-numbers: true

import tarfile
import tempfile
from pathlib import Path

import numpy as np
from flask import Flask, request
from tensorflow import keras

MODEL_PATH = Path(__file__).parent


class Model:
    model = None

    def load(self):
        """
        Extracts the model package and loads the model in memory
        if it hasn't been loaded yet.
        """
        if not Model.model:
            with tempfile.TemporaryDirectory() as directory:
                with tarfile.open(MODEL_PATH / "model.tar.gz") as tar:
                    tar.extractall(path=directory)
                Model.model = keras.models.load_model(Path(directory) / "001")

    def predict(self, data):
        """
        Generates predictions for the supplied data.
        """
        self.load()
        return Model.model.predict(data)


app = Flask(__name__)
model = Model()


@app.route("/predict/", methods=["POST"])
def predict():
    """
    Generates predictions for the supplied data.
    """
    data = request.data.decode("utf-8")

    # Split the data into rows
    rows = data.strip().split("\n")

    # Split each row by commas and convert to floats
    data = [list(map(float, row.split(","))) for row in rows]

    # Convert to NumPy array
    data = np.array(data)

    # Get Predictions
    predictions = model.predict(data)

    # Format the predictions into a list for better readability
    predictions_list = predictions.tolist()

    return {"predictions": predictions_list}
