import os
from pathlib import Path
from typing import Union

from dotenv import load_dotenv
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.pipeline import PipelineModel
from sagemaker.tensorflow.model import TensorFlowModel
from sagemaker.workflow.functions import Join
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import LocalPipelineSession, PipelineSession
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.steps import CacheConfig, ProcessingStep

from aws_pipelines.evaluate_pipeline import create_evaluation_step
from aws_pipelines.preprocessing_pipeline import (
    create_processor,
    define_parameters,
    define_processing_step,
)
from aws_pipelines.training_pipeline import (
    create_tensorflow_estimator,
    create_training_step,
)
from aws_pipelines.tuning_pipeline import create_tuning_step

# Load environment variables if not provided
load_dotenv()


def create_registration_step(
    model: Union[PipelineModel, TensorFlowModel],
    evaluate_model_step: ProcessingStep,
    config: dict,
    model_package_group_name: str,
    approval_status: str = "Approved",
) -> ModelStep:
    """
    Create a Registration Step using the supplied parameters.

    Parameters:
        model (sagemaker.model.FrameworkModel): The model object to register.
        evaluate_model_step (ProcessingStep): The evaluation step.
        config (dict): Configuration dictionary.
        model_package_group_name (str): Name of the model package group.
        approval_status (str): Approval status for the model package.

    Returns:
        ModelStep: The registration step object.

    Raises:
        ValueError: If 'config' is not provided.
        KeyError: If required keys are missing in the 'config' dictionary.
    """
    # Validate required keys in the config dictionary
    if config is None:
        raise ValueError("The 'config' parameter must be provided.")

    # Validate required keys in the config dictionary
    required_keys = ["framework_version", "instance_type", "image"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Config dictionary must contain the key: {key}")

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    evaluate_model_step.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
                    "evaluation.json",
                ],
            ),
            content_type="application/json",
        ),
    )
    config["framework_version"] = "2.12"

    return ModelStep(
        name="register",
        step_args=model.register(
            model_package_group_name=model_package_group_name,
            approval_status=approval_status,
            model_metrics=model_metrics,
            content_types=["text/csv", "application/json"],
            response_types=["text/csv", "application/json"],
            inference_instances=[config["instance_type"]],
            transform_instances=[config["instance_type"]],
            framework_version=config["framework_version"],
        ),
    )


def create_register_pipeline(
    role: str,
    bucket: str,
    local_mode: bool = False,
    s3_location: str = None,
    code_folder: str = None,
    comet_api_key: str = None,
    comet_project_name: str = None,
    use_tuning_step: bool = True,
    model_package_group_name: str = None,
) -> Pipeline:
    """
    Create a registration pipeline.

    Parameters:
        role (str): IAM role for SageMaker.
        bucket (str): S3 bucket for storing pipeline artifacts.
        local_mode (bool): A flag indicating whether to run the pipeline in local mode. Defaults to False.
        s3_location (str): The base S3 location for storing pipeline outputs. If None, it will default to f"s3://{bucket}".
        code_folder (str): The folder path where the processing script is located. If None, it will default to "../src".
        comet_apy_key (str): The Comet API key for logging.
        comet_project_name (str): The Comet project name for logging.
        use_tuning_step (bool): Flag to indicate if the tuning step should be used.
        model_package_group_name (Optional[str]): Name of the model package group.

    Returns:
        Pipeline: The SageMaker pipeline object.
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

    # Create tuning step
    tune_model_step = create_tuning_step(estimator, preprocessing_step, cache_config)

    # Select model assets based on whether tuning is used
    if use_tuning_step:
        model_assets = tune_model_step.get_top_model_s3_uri(
            top_k=0,
            s3_bucket=config["session"].default_bucket(),
        )
    else:
        model_assets = train_model_step.properties.ModelArtifacts.S3ModelArtifacts

    # Create evaluation step
    evaluate_model_step = create_evaluation_step(role, preprocessing_step, cache_config, code_folder, model_assets, config)

    config["framework_version"] = "2.12"

    # Create TensorFlow model
    tensorflow_model = TensorFlowModel(
        model_data=model_assets,
        framework_version=config["framework_version"],
        sagemaker_session=config["session"],
        role=role,
    )

    # Create registration step
    register_model_step = create_registration_step(tensorflow_model, evaluate_model_step, config, model_package_group_name)

    # Create and return the pipeline
    register_model_pipeline = Pipeline(
        name="register-model-pipeline",
        parameters=[dataset_location],
        steps=[
            preprocessing_step,
            tune_model_step if use_tuning_step else train_model_step,
            evaluate_model_step,
            register_model_step,
        ],
        pipeline_definition_config=pipeline_definition_config,
        sagemaker_session=config["session"],
    )

    register_model_pipeline.upsert(role_arn=role)

    print("Pipeline upserted successfully.")

    return register_model_pipeline


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
        basic_model_package_group = os.environ["BASIC_MODEL_PACKAGE_GROUP"]
        use_tuning_step = os.environ.get("USE_TUNING_STEP", "True") == "True"

        print(f"LOCAL_MODE: {local_mode}")
        print(f"USE_TUNING_STEP: {use_tuning_step}")

        if not bucket:
            raise ValueError("S3 bucket must be specified")

        register_pipeline = create_register_pipeline(
            role,
            bucket,
            local_mode,
            s3_location,
            code_folder,
            comet_api_key,
            comet_project_name,
            use_tuning_step,
            basic_model_package_group,
        )

        # Start the pipeline execution (if required)
        register_pipeline.start()

    except KeyError as e:
        print(f"Environment variable not set: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except ImportError as e:
        print(f"Import error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
