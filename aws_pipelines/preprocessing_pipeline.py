import os
from pathlib import Path

from dotenv import load_dotenv
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import LocalPipelineSession, PipelineSession
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.steps import CacheConfig, ProcessingStep

# Load environment variables if not provided
load_dotenv()


def define_parameters(s3_location: str = None) -> ParameterString:
    """
    Defines the parameters for the SageMaker preprocessing pipeline.

    Parameters:
        s3_location (str, optional): The base S3 location for storing pipeline outputs.
                                     If not provided, it must be set when calling the function.

    Returns:
        ParameterString: The dataset location parameter.

    Raises:
        ValueError: If 's3_location' is not provided.
    """
    if s3_location is None:
        raise ValueError("s3_location must be provided")

    return ParameterString(
        name="dataset_location",
        default_value=f"{s3_location}/data",
    )


def create_processor(role: str = None, config: dict = None) -> SKLearnProcessor:
    """
    Creates the SKLearnProcessor for the SageMaker preprocessing pipeline.

    Parameters:
        role (str): The IAM role used by SageMaker. This parameter is required.
        config (dict): The configuration dictionary containing session and instance type.
                       This parameter is required and must contain 'session' and 'instance_type' keys.

    Returns:
        SKLearnProcessor: The SKLearnProcessor object.

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
    required_keys = ["instance_type", "session"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Config dictionary must contain the key: {key}")

    return SKLearnProcessor(
        base_job_name="preprocess-data",
        framework_version="1.2-1",
        instance_type=config["instance_type"],
        instance_count=1,
        role=role,
        sagemaker_session=config["session"],
    )


def define_processing_step(
    processor: SKLearnProcessor = None,
    dataset_location: ParameterString = None,
    code_folder: str = None,
    s3_location: str = None,
    cache_config: CacheConfig = None,
) -> ProcessingStep:
    """
    Defines the processing step for the SageMaker preprocessing pipeline.

    Parameters:
        processor (SKLearnProcessor): The processor object.
        dataset_location (ParameterString): The dataset location parameter.
        code_folder (str): The folder path where the processing script is located.
        s3_location (str): The base S3 location for storing pipeline outputs.
        cache_config (CacheConfig): The cache configuration for the pipeline step.

    Returns:
        ProcessingStep: The processing step object.

    Raises:
        ValueError: If any required parameter is not provided.
    """
    if processor is None:
        raise ValueError("The 'processor' parameter must be provided.")
    if dataset_location is None:
        raise ValueError("The 'dataset_location' parameter must be provided.")
    if code_folder is None:
        raise ValueError("The 'code_folder' parameter must be provided.")
    if s3_location is None:
        raise ValueError("The 's3_location' parameter must be provided.")

    return ProcessingStep(
        name="preprocess-data",
        step_args=processor.run(
            code=f"{(code_folder / 'processing' / 'script.py').as_posix()}",
            # code=str(code_folder / 'processing' / 'script.py'),
            # code=(code_folder / 'processing' / 'script.py').as_posix(),
            inputs=[
                ProcessingInput(
                    source=dataset_location,
                    destination="/opt/ml/processing/input",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="train",
                    source="/opt/ml/processing/train",
                    destination=f"{s3_location}/preprocessing/train",
                ),
                ProcessingOutput(
                    output_name="validation",
                    source="/opt/ml/processing/validation",
                    destination=f"{s3_location}/preprocessing/validation",
                ),
                ProcessingOutput(
                    output_name="test",
                    source="/opt/ml/processing/test",
                    destination=f"{s3_location}/preprocessing/test",
                ),
                ProcessingOutput(
                    output_name="model",
                    source="/opt/ml/processing/model",
                    destination=f"{s3_location}/preprocessing/model",
                ),
                ProcessingOutput(
                    output_name="train-baseline",
                    source="/opt/ml/processing/train-baseline",
                    destination=f"{s3_location}/preprocessing/train-baseline",
                ),
                ProcessingOutput(
                    output_name="test-baseline",
                    source="/opt/ml/processing/test-baseline",
                    destination=f"{s3_location}/preprocessing/test-baseline",
                ),
            ],
        ),
        cache_config=cache_config,
    )


def create_preprocessing_pipeline(
    role: str = None,
    bucket: str = None,
    local_mode: bool = False,
    s3_location: str = None,
    code_folder: str = None,
) -> Pipeline:
    """
    Creates and upserts a SageMaker preprocessing pipeline.

    Parameters:
        role (str): The IAM role used by SageMaker.
        bucket (str): The S3 bucket used for storing data and artifacts.
        local_mode (bool): A flag indicating whether to run the pipeline in local mode. Defaults to False.
        s3_location (str): The base S3 location for storing pipeline outputs. If None, it will default to f"s3://{bucket}".
        code_folder (str): The folder path where the processing script is located. If None, it will default to "../src".

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

    # Define pipeline
    preprocessing_pipeline = Pipeline(
        name="preprocessing-pipeline",
        parameters=[dataset_location],
        steps=[
            preprocessing_step,
        ],
        pipeline_definition_config=pipeline_definition_config,
        sagemaker_session=config["session"],
    )

    # Upsert pipeline
    preprocessing_pipeline.upsert(role_arn=role)

    print("Pipeline upserted successfully.")

    return preprocessing_pipeline


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
        print(f"LOCAL_MODE: {local_mode}")

        if not bucket:
            raise ValueError("S3 bucket must be specified")

        preprocessing_pipeline = create_preprocessing_pipeline(role, bucket, local_mode, s3_location, code_folder)

        # Start the pipeline execution (if required)
        preprocessing_pipeline.start()

    except KeyError as e:
        print(f"Environment variable not set: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except ImportError as e:
        print(f"Import error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
