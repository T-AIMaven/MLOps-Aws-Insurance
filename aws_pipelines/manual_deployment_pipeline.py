import os
from pathlib import Path

from dotenv import load_dotenv
from sagemaker.tensorflow.model import TensorFlowModel

# from sagemaker.workflow.parameters import ParameterFloat
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import LocalPipelineSession, PipelineSession
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.steps import CacheConfig

from aws_pipelines.cond_register_pipeline import (
    create_condition_step,
    create_fail_step,
    create_registration_step,
)
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


# THIS DEPLOYMENT ASSUMES THAT THE LAMBDA FUNCTION AND THE EVENTBRIGE HAVE BEEN ALREADY SET UP


def create_manual_lambda_deployment_pipeline(
    role: str = None,
    bucket: str = None,
    local_mode: bool = False,
    s3_location: str = None,
    code_folder: str = None,
    comet_api_key: str = None,
    comet_project_name: str = None,
    use_tuning_step: bool = True,
    model_package_group_name: str = None,
    mae_threshold: float = 3000.00,
    approval_status: str = "PendingManualApproval",
):
    """
    Create a registration pipeline with a conditional step to evaluate model performance.

    Parameters:
        role (str): IAM role for SageMaker.
        bucket (str): S3 bucket for storing pipeline artifacts.
        local_mode (bool): A flag indicating whether to run the pipeline in local mode. Defaults to False.
        s3_location (str): The base S3 location for storing pipeline outputs. If None, it will default to f"s3://{bucket}".
        code_folder (str): The folder path where the processing script is located. If None, it will default to "../src".
        comet_apy_key (str): The Comet API key for logging.
        comet_project_name (str): The Comet project name for logging.
        use_tuning_step (bool): Flag to indicate if the tuning step should be used.
        model_package_group_name (str): Name of the model package group.
        mae_threshold (float): The MAE threshold for model evaluation.

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
    register_model_step = create_registration_step(
        tensorflow_model, evaluate_model_step, config, model_package_group_name, approval_status=approval_status
    )

    fail_step = create_fail_step(mae_threshold)

    condition_step = create_condition_step(
        mae_threshold,
        evaluate_model_step,
        register_model_step,
        fail_step,
    )

    # Create and return the pipeline
    lambda_pipeline = Pipeline(
        name="lambda-pipeline",
        parameters=[dataset_location, mae_threshold],
        steps=[
            preprocessing_step,
            tune_model_step if use_tuning_step else train_model_step,
            evaluate_model_step,
            condition_step,
        ],
        pipeline_definition_config=pipeline_definition_config,
        sagemaker_session=config["session"],
    )

    lambda_pipeline.upsert(role_arn=role)

    print("Pipeline upserted successfully.")

    return lambda_pipeline


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
        pending_model_package_group = os.environ["PENDING_MODEL_PACKAGE_GROUP"]
        use_tuning_step = os.environ.get("USE_TUNING_STEP", "True") == "True"
        # THRESHOLD = os.environ["MAE_THRESHOLD"]
        # mae_threshold = ParameterFloat(name="mae_threshold", default_value=float(THRESHOLD))
        mae_threshold = float(os.environ["MAE_THRESHOLD"])
        approval_status = "PendingManualApproval"
        print(f"MAE_THRESHOLD: {mae_threshold}")
        print(f"LOCAL_MODE: {local_mode}")
        print(f"USE_TUNING_STEP: {use_tuning_step}")

        if not bucket:
            raise ValueError("S3 bucket must be specified")

        manual_lambda_deployment_pipeline = create_manual_lambda_deployment_pipeline(
            role,
            bucket,
            local_mode,
            s3_location,
            code_folder,
            comet_api_key,
            comet_project_name,
            use_tuning_step,
            pending_model_package_group,
            mae_threshold,
            approval_status,
        )

        # Start the pipeline execution (if required)
        manual_lambda_deployment_pipeline.start()

    except KeyError as e:
        print(f"Environment variable not set: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except ImportError as e:
        print(f"Import error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
