# | filename: script.py
# | code-line-numbers: true

import json
import tarfile
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras


def evaluate(model_path, test_path, output_path):
    """
    Evaluate the model.
    """
    X_test = pd.read_csv(Path(test_path) / "test.csv")
    y_test = X_test[X_test.columns[-1]]
    X_test = X_test.drop(X_test.columns[-1], axis=1)

    # Extract the model package
    with tarfile.open(Path(model_path) / "model.tar.gz") as tar:
        tar.extractall(path=Path(model_path))

    model = keras.models.load_model(Path(model_path) / "001")

    # Evaluate the model
    predictions = model.predict(X_test).flatten()

    # Print evaluation metrics
    test_mse = mean_squared_error(y_test, predictions)
    print(f"Test MSE: {test_mse}")
    test_r2_score = r2_score(y_test, predictions)
    print(f"Test R2 Score: {test_r2_score}")
    test_mae = mean_absolute_error(y_test, predictions)
    print(f"Test MAE: {test_mae}")

    # Let's create an evaluation report using the model metrics.
    evaluation_report = {
        "metrics": {
            "mean_squared_error": {"value": test_mse},
            "r2_score": {"value": test_r2_score},
            "mean_absolute_error": {"value": test_mae},
        },
    }

    Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(Path(output_path) / "evaluation.json", "w") as f:
        f.write(json.dumps(evaluation_report))


if __name__ == "__main__":
    evaluate(
        model_path="/opt/ml/processing/model/",
        test_path="/opt/ml/processing/test/",
        output_path="/opt/ml/processing/evaluation/",
    )
