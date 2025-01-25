import os
from pathlib import Path

from dotenv import load_dotenv
from sagemaker.inputs import TrainingInput
from sagemaker.parameter import IntegerParameter
from sagemaker.tensorflow import TensorFlow
from sagemaker.tuner import HyperparameterTuner
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import LocalPipelineSession, PipelineSession
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TuningStep

from aws_pipelines.preprocessing_pipeline import (
    create_processor,
    define_parameters,
    define_processing_step,
)
from aws_pipelines.training_pipeline import create_tensorflow_estimator

# Load environment variables if not provided
load_dotenv()


def create_tuning_step(estimator: TensorFlow, preprocessing_step: ProcessingStep, cache_config: CacheConfig) -> TuningStep:
    """
    Creates a SageMaker TuningStep using the provided estimator.

    Parameters:
        estimator (sagemaker.tensorflow.TensorFlow): The TensorFlow estimator object.
        preprocessing_step (sagemaker.workflow.steps.ProcessingStep): The preprocessing step object.
        cache_config (sagemaker.workflow.steps.CacheConfig): The cache configuration for the pipeline step.

    Returns:
        sagemaker.workflow.steps.TuningStep: The tuning step object.
    """

    tuner = HyperparameterTuner(
        estimator,
        objective_metric_name="val_mean_absolute_error",
        objective_type="Minimize",
        hyperparameter_ranges={
            "epochs": IntegerParameter(10, 50),
        },
        metric_definitions=[{"Name": "val_mean_absolute_error", "Regex": "val_mean_absolute_error: ([0-9\\.]+)"}],
        max_jobs=3,
        max_parallel_jobs=3,
    )

    return TuningStep(
        name="tune-model",
        step_args=tuner.fit(
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


def create_tuning_pipeline(
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
        comet_api_key (str): The Comet API key for logging.
        comet_project_name (str): The Comet project name for logging.

    Returns:
        sagemaker.workflow.pipeline.Pipeline: The SageMaker pipeline object.
    """

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

    # Create tuning step
    tune_model_step = create_tuning_step(estimator, preprocessing_step, cache_config)

    # Define pipeline
    tuning_pipeline = Pipeline(
        name="tuning-pipeline",
        parameters=[dataset_location],
        steps=[
            preprocessing_step,
            tune_model_step,
        ],
        pipeline_definition_config=pipeline_definition_config,
        sagemaker_session=config["session"],
    )

    # Upsert pipeline
    tuning_pipeline.upsert(role_arn=role)

    print("Pipeline upserted successfully.")

    return tuning_pipeline


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

        tuning_pipeline = create_tuning_pipeline(
            role, bucket, local_mode, s3_location, code_folder, comet_api_key, comet_project_name
        )

        # Run pipeline
        tuning_pipeline.start()

    except KeyError as e:
        print(f"Environment variable not set: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except ImportError as e:
        print(f"Import error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
