import os
from pathlib import Path

from dotenv import load_dotenv
from sagemaker.inputs import TrainingInput
from sagemaker.tensorflow import TensorFlow
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import LocalPipelineSession, PipelineSession
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TrainingStep

from aws_pipelines.preprocessing_pipeline import (
    create_processor,
    define_parameters,
    define_processing_step,
)

# Load environment variables if not provided
load_dotenv()


def create_tensorflow_estimator(
    config: dict = None, role: str = None, code_folder: str = None, comet_apy_key: str = None, comet_project_name: str = None
) -> TensorFlow:
    """
    Creates the TensorFlow estimator for the SageMaker training job.

    Parameters:
        config (dict): The configuration dictionary containing session, instance type, image, framework version, and py_version.
        role (str): The IAM role used by SageMaker.
        code_folder (str): The folder path where the training script is located.
        comet_apy_key (str): The Comet API key for logging.
        comet_project_name (str): The Comet project name for logging.

    Returns:
        sagemaker.tensorflow.TensorFlow: The TensorFlow estimator object.

    Raises:
        ValueError: If 'role' or 'config' is not provided.
        KeyError: If required keys are missing in the 'config' dictionary.
    """
    # Validate required keys in the config dictionary
    if role is None:
        raise ValueError("The 'role' parameter must be provided.")
    if config is None:
        raise ValueError("The 'config' parameter must be provided.")

    # Validate required keys in the config dictionary
    required_keys = ["session", "instance_type", "image"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Config dictionary must contain the key: {key}")

    # TensorFlow container we'll use.
    config["framework_version"] = "2.12"
    config["py_version"] = "py310"

    return TensorFlow(
        base_job_name="training",
        entry_point="script.py",
        source_dir=f"{(code_folder / 'training').as_posix()}",
        hyperparameters={
            "epochs": 50,
            "batch_size": 32,
        },
        environment={
            "COMET_API_KEY": comet_apy_key,
            "COMET_PROJECT_NAME": comet_project_name,
        },
        metric_definitions=[
            {"Name": "loss", "Regex": "loss: ([0-9\\.]+)"},
            {"Name": "mean_absolute_error", "Regex": "mean_absolute_error: ([0-9\\.]+)"},
            {"Name": "val_loss", "Regex": "val_loss: ([0-9\\.]+)"},
            {"Name": "val_mean_absolute_error", "Regex": "val_mean_absolute_error: ([0-9\\.]+)"},
        ],
        image_uri=config["image"],
        framework_version=config["framework_version"],
        py_version=config["py_version"],
        instance_type=config["instance_type"],
        instance_count=1,
        disable_profiler=True,
        debugger_hook_config=False,
        sagemaker_session=config["session"],
        role=role,
    )


def create_training_step(estimator: TensorFlow, preprocessing_step: ProcessingStep, cache_config: CacheConfig) -> TrainingStep:
    """
    Creates a SageMaker TrainingStep using the provided estimator.

    Parameters:
        estimator (sagemaker.tensorflow.TensorFlow): The TensorFlow estimator object.
        preprocessing_step (sagemaker.workflow.steps.ProcessingStep): The preprocessing step object.
        cache_config (sagemaker.workflow.steps.CacheConfig): The cache configuration for the pipeline step.

    Returns:
        sagemaker.workflow.steps.TrainingStep: The training step object.
    """
    return TrainingStep(
        name="train-model",
        step_args=estimator.fit(
            inputs={
                "train": TrainingInput(
                    s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                    content_type="text/csv",
                ),
                "validation": TrainingInput(
                    s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                    content_type="text/csv",
                ),
                "pipeline": TrainingInput(
                    s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs["model"].S3Output.S3Uri,
                    content_type="application/tar+gzip",
                ),
            },
        ),
        cache_config=cache_config,
    )


def create_training_pipeline(
    role: str = None,
    bucket: str = None,
    local_mode: bool = False,
    s3_location: str = None,
    code_folder: str = None,
    comet_api_key: str = None,
    comet_project_name: str = None,
) -> Pipeline:
    """
    Creates and upserts a SageMaker training pipeline.

    Parameters:
        role (str): The IAM role used by SageMaker.
        bucket (str): The S3 bucket used for storing data and artifacts.
        local_mode (bool): A flag indicating whether to run the pipeline in local mode. Defaults to False.
        s3_location (str): The base S3 location for storing pipeline outputs. If None, it will default to f"s3://{bucket}".
        code_folder (str): The folder path where the processing script is located. If None, it will default to "../src".
        comet_apy_key (str): The Comet API key for logging.
        comet_project_name (str): The Comet project name for logging.

    Returns:
        sagemaker.workflow.pipeline.Pipeline: The SageMaker pipeline object.
    """
    # Default values
    if s3_location is None:
        s3_location = f"s3://{bucket}"
    if code_folder is None:
        code_folder = "../src"

    # Configure pipeline session
    pipeline_session = PipelineSession(default_bucket=bucket) if not local_mode else None

    if local_mode:
        config = {
            "session": LocalPipelineSession(default_bucket=bucket),
            "instance_type": "local",
            "image": None,
        }
    else:
        config = {
            "session": pipeline_session,
            "instance_type": "ml.m5.xlarge",
            "image": None,
        }

    # Define pipeline configuration and caching
    pipeline_definition_config = PipelineDefinitionConfig(use_custom_job_prefix=True)
    cache_config = CacheConfig(enable_caching=True, expire_after="15d")

    # Define parameters
    dataset_location = define_parameters(s3_location)

    # Create processor
    processor = create_processor(role, config)

    # Define processing step
    preprocessing_step = define_processing_step(processor, dataset_location, code_folder, s3_location, cache_config)

    # Create TensorFlow estimator
    estimator = create_tensorflow_estimator(config, role, code_folder, comet_api_key, comet_project_name)

    # Create training step
    train_model_step = create_training_step(estimator, preprocessing_step, cache_config)

    # Define pipeline
    training_pipeline = Pipeline(
        name="training-pipeline",
        parameters=[dataset_location],
        steps=[
            preprocessing_step,
            train_model_step,
        ],
        pipeline_definition_config=pipeline_definition_config,
        sagemaker_session=config["session"],
    )

    # Upsert pipeline
    training_pipeline.upsert(role_arn=role)

    print("Pipeline upserted successfully.")

    return training_pipeline


if __name__ == "__main__":
    # Check if running in GitHub Actions
    running_in_github_actions = os.getenv("RUNNING_IN_GITHUB_ACTIONS", "false") == "true"

    # Define the local code folder path
    if running_in_github_actions:
        root_dir = Path(__file__).resolve().parent.parent
        code_folder = root_dir / "src"
    else:
        code_folder = Path("../src")

    try:
        role = os.environ["ROLE"]
        bucket = os.environ.get("BUCKET", None)
        local_mode = os.environ.get("LOCAL_MODE", "False") == "True"
        s3_location = f"s3://{bucket}"
        # code_folder = Path("../src")
        comet_api_key = os.environ["COMET_API_KEY"]
        comet_project_name = os.environ["COMET_PROJECT_NAME"]
        print(f"LOCAL_MODE: {local_mode}")

        if not bucket:
            raise ValueError("S3 bucket must be specified")

        training_pipeline = create_training_pipeline(
            role, bucket, local_mode, s3_location, code_folder, comet_api_key, comet_project_name
        )

        # Run pipeline
        training_pipeline.start()

    except KeyError as e:
        print(f"Environment variable not set: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except ImportError as e:
        print(f"Import error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
