# | filename: script.py
# | code-line-numbers: true

import argparse
import json
import os
import tarfile
from pathlib import Path

import keras
import pandas as pd
from comet_ml import Experiment
from keras import Input
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from packaging import version
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train(
    model_directory,
    train_path,
    validation_path,
    pipeline_path,
    experiment,
    epochs=50,
    batch_size=32,
):
    """
    Train a neural network model using the provided training and validation datasets.

    Parameters:
    - model_directory (str): Directory to save the trained model.
    - train_path (str): Path to the training dataset.
    - validation_path (str): Path to the validation dataset.
    - pipeline_path (str): Path to the preprocessing pipelines.
    - experiment (Experiment): Comet experiment object for logging.
    - epochs (int): Number of epochs for training.
    - batch_size (int): Batch size for training.
    """
    print(f"Keras version: {keras.__version__}")

    # Load training data
    X_train = pd.read_csv(Path(train_path) / "train.csv")
    y_train = X_train[X_train.columns[-1]]
    X_train = X_train.drop(X_train.columns[-1], axis=1)

    # Load validation data
    X_validation = pd.read_csv(Path(validation_path) / "validation.csv")
    y_validation = X_validation[X_validation.columns[-1]]
    X_validation = X_validation.drop(X_validation.columns[-1], axis=1)

    # Define the model
    model = Sequential(
        [
            Input(shape=(X_train.shape[1],)),
            Dense(256, activation="relu"),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1, activation="linear"),
        ]
    )

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.01), loss="mean_absolute_error", metrics=["mean_absolute_error"])

    # Train the model
    model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )

    # Evaluate the model
    predictions = model.predict(X_validation).flatten()
    # Print evaluation metrics
    val_mse = mean_squared_error(y_validation, predictions)
    print(f"Validation MSE: {val_mse}")
    val_r2_score = r2_score(y_validation, predictions)
    print(f"R2 Score: {val_r2_score}")

    val_mae = mean_absolute_error(y_validation, predictions)
    print(f"Validation MAE: {val_mae}")

    # Save the model
    model_filepath = (
        Path(model_directory) / "001"
        if version.parse(keras.__version__) < version.parse("3")
        else Path(model_directory) / "insurance.keras"
    )

    model.save(model_filepath)

    # Save the transformation pipelines
    with tarfile.open(Path(pipeline_path) / "model.tar.gz", "r:gz") as tar:
        tar.extractall(model_directory)

    # Log metrics and parameters to Comet
    if experiment:
        experiment.log_parameters(
            {
                "epochs": epochs,
                "batch_size": batch_size,
            }
        )
        experiment.log_dataset_hash(X_train)
        experiment.log_metric("mean_squared_error", val_mse)
        experiment.log_metric("mean_absolute_error", val_mae)
        experiment.log_metric("r2_score", val_r2_score)
        experiment.log_model("insurance", model_filepath.as_posix())


if __name__ == "__main__":
    # Parse script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    args, _ = parser.parse_known_args()

    # Create a Comet experiment to log metrics and parameters
    comet_api_key = os.environ.get("COMET_API_KEY", None)
    comet_project_name = os.environ.get("COMET_PROJECT_NAME", None)

    experiment = (
        Experiment(
            project_name=comet_project_name,
            api_key=comet_api_key,
            auto_metric_logging=True,
            auto_param_logging=True,
            log_code=True,
        )
        if comet_api_key and comet_project_name
        else None
    )

    # Get SageMaker training job name for experiment naming
    training_env = json.loads(os.environ.get("SM_TRAINING_ENV", {}))
    job_name = training_env.get("job_name", None) if training_env else None

    if job_name and experiment:
        experiment.set_name(job_name)

    # Call the train function with parsed arguments and environment variables
    train(
        model_directory=os.environ["SM_MODEL_DIR"],
        train_path=os.environ["SM_CHANNEL_TRAIN"],
        validation_path=os.environ["SM_CHANNEL_VALIDATION"],
        pipeline_path=os.environ["SM_CHANNEL_PIPELINE"],
        experiment=experiment,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
