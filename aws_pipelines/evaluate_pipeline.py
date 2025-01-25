import os
from pathlib import Path
from typing import Dict, Union

from dotenv import load_dotenv
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.tensorflow import TensorFlowProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import LocalPipelineSession, PipelineSession
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import CacheConfig, ProcessingStep

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


def create_evaluation_step(
    role: str,
    preprocessing_step: ProcessingStep,
    cache_config: CacheConfig,
    code_folder: Union[str, Path],
    model_assets: str,
    config: Dict = None,
) -> ProcessingStep:
    """
    Creates a SageMaker ProcessingStep for model evaluation.

    Parameters:
        role (str): The IAM role used by SageMaker.
        preprocessing_step (ProcessingStep): The preprocessing step object.
        cache_config (CacheConfig): The cache configuration for the pipeline step.
        code_folder (Union[str, Path]): The folder path where the evaluation script is located.
        model_assets (str): The S3 URI of the model assets.
        config (dict): Configuration dictionary containing session, instance type, image, framework, and py_version.

    Returns:
        ProcessingStep: The processing step object for evaluation.
    """

    evaluation_report = PropertyFile(name="evaluation-report", output_name="evaluation", path="evaluation.json")

    config["framework_version"] = "2.12"
    config["py_version"] = "py310"

    evaluation_processor = TensorFlowProcessor(
        base_job_name="evaluation-processor",
        image_uri=config["image"],
        framework_version=config["framework_version"],
        py_version=config["py_version"],
        instance_type=config["instance_type"],
        instance_count=1,
        role=role,
        sagemaker_session=config["session"],
    )
    return ProcessingStep(
        name="evaluate-model",
        step_args=evaluation_processor.run(
            code=f"{(code_folder / 'evaluation' / 'script.py').as_posix()}",
            inputs=[
                # The first input is the test split that we generated on
                # the first step of the pipeline when we split and
                # transformed the data.
                ProcessingInput(
                    source=preprocessing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                    destination="/opt/ml/processing/test",
                ),
                # The second input is the model that we generated on
                # the Training or Tunning Step.
                ProcessingInput(
                    source=model_assets,
                    destination="/opt/ml/processing/model",
                ),
            ],
            outputs=[
                # The output is the evaluation report that we generated
                # in the evaluation script.
                ProcessingOutput(
                    output_name="evaluation",
                    source="/opt/ml/processing/evaluation",
                ),
            ],
        ),
        property_files=[evaluation_report],
        cache_config=cache_config,
    )


def create_evaluation_pipeline(
    role: str = None,
    bucket: str = None,
    local_mode: bool = False,
    s3_location: str = None,
    code_folder: str = None,
    comet_api_key: str = None,
    comet_project_name: str = None,
    use_tuning_step: bool = True,
):
    """
    Creates a SageMaker pipeline for evaluation.

    Parameters:
        role (str): IAM role for SageMaker.
        bucket (str): S3 bucket for storing pipeline artifacts.
        local_mode (bool): A flag indicating whether to run the pipeline in local mode. Defaults to False.
        s3_location (str): The base S3 location for storing pipeline outputs. If None, it will default to f"s3://{bucket}".
        code_folder (str): The folder path where the processing script is located. If None, it will default to "../src".
        comet_apy_key (str): The Comet API key for logging.
        comet_project_name (str): The Comet project name for logging.
        use_tuning_step (bool): Flag to indicate if the tuning step should be used.

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

    tune_model_step = create_tuning_step(estimator, preprocessing_step, cache_config)

    if use_tuning_step:
        model_assets = tune_model_step.get_top_model_s3_uri(
            top_k=0,
            s3_bucket=config["session"].default_bucket(),
        )
    else:
        model_assets = train_model_step.properties.ModelArtifacts.S3ModelArtifacts

    # Create evaluation step
    evaluate_model_step = create_evaluation_step(role, preprocessing_step, cache_config, code_folder, model_assets, config)

    evaluation_pipeline = Pipeline(
        name="evaluation-pipeline",
        parameters=[dataset_location],
        steps=[
            preprocessing_step,
            tune_model_step if use_tuning_step else train_model_step,
            evaluate_model_step,
        ],
        pipeline_definition_config=pipeline_definition_config,
        sagemaker_session=config["session"],
    )

    evaluation_pipeline.upsert(role_arn=role)

    print("Pipeline upserted successfully.")

    return evaluation_pipeline


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
        use_tuning_step = os.environ.get("USE_TUNING_STEP", "True") == "True"
        print(f"LOCAL_MODE: {local_mode}")
        print(f"USE_TUNING_STEP: {use_tuning_step}")

        if not bucket:
            raise ValueError("S3 bucket must be specified")

        evaluation_pipeline = create_evaluation_pipeline(
            role,
            bucket,
            local_mode,
            s3_location,
            code_folder,
            comet_api_key,
            comet_project_name,
            use_tuning_step,
        )

        # Start the pipeline execution (if required)
        evaluation_pipeline.start()

    except KeyError as e:
        print(f"Environment variable not set: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except ImportError as e:
        print(f"Import error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
