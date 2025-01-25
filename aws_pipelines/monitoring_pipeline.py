import os
import time

import boto3
import sagemaker
from dotenv import load_dotenv
from sagemaker import ModelPackage
from sagemaker.model_monitor import (
    CronExpressionGenerator,
    DataCaptureConfig,
    DefaultModelMonitor,
    EndpointInput,
    ModelQualityMonitor,
)
from sagemaker.workflow.pipeline_context import LocalPipelineSession, PipelineSession

# Load environment variables if not provided
load_dotenv()


def define_config(local_mode: bool = False, bucket: str = None) -> dict:
    """
    Define configuration for the pipeline session.

    Parameters:
        local_mode (bool): A flag indicating whether to run the pipeline in local mode.
        bucket (str): S3 bucket for storing pipeline artifacts.

    Returns:
        dict: Configuration dictionary containing session, instance type, and image.
    """
    # Configure pipeline session
    pipeline_session = PipelineSession(default_bucket=bucket) if not local_mode else None

    if local_mode:
        return {
            "session": LocalPipelineSession(default_bucket=bucket),
            "instance_type": "local",
            "image": None,
        }
    else:
        return {
            "session": pipeline_session,
            "instance_type": "ml.m5.xlarge",
            "image": None,
        }


def deploy_model(
    config: dict = None,
    model_package_group_name: str = None,
    endpoint: str = None,
    data_capture_destination: str = None,
    role: str = None,
):
    """
    Deploy the latest model registered in the Model Registry.

    Parameters:
        config (dict): Configuration dictionary.
        model_package_group_name (str): Name of the model package group.
        endpoint_name (str): The SageMaker endpoint name.
        data_capture_destination (str): The S3 location for data capture.
        role (str): IAM role for SageMaker.

    Returns:
        None
    """
    sagemaker_session = sagemaker.session.Session()
    sagemaker_client = boto3.client("sagemaker")

    response = sagemaker_client.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        MaxResults=1,
    )

    package = response["ModelPackageSummaryList"][0] if response["ModelPackageSummaryList"] else None

    if package:
        model_package = ModelPackage(
            model_package_arn=package["ModelPackageArn"],
            sagemaker_session=sagemaker_session,
            role=role,
        )

        endpoint_config_name = f"monitoring-model-insurance-endpoint-{int(time.time())}"

        model_package.deploy(
            endpoint_name=endpoint,
            initial_instance_count=1,
            instance_type=config["instance_type"],
            # We must enable Data Capture to monitor the model.
            data_capture_config=DataCaptureConfig(
                enable_capture=True,
                sampling_percentage=100,
                destination_s3_uri=data_capture_destination,
                capture_options=["REQUEST", "RESPONSE"],
                csv_content_types=["text/csv"],
                json_content_types=["application/json"],
            ),
            endpoint_config_name=endpoint_config_name,
        )

        # Wait for the endpoint to be in service
        waiter = sagemaker_client.get_waiter("endpoint_in_service")
        waiter.wait(EndpointName=endpoint)
        print(f"Endpoint {endpoint} is in service.")


def create_data_monitor(role: str = None, endpoint: str = None, data_quality_location: str = None, config: dict = None):
    """
    Create a data monitor for the endpoint.

    Parameters:
        role (str): IAM role for SageMaker.
        endpoint_name (str): The SageMaker endpoint name.
        data_quality_location (str): S3 location for data quality output.
        config (dict): Configuration dictionary.

    Returns:
        MonitoringSchedule: The monitoring schedule object.
    """
    data_monitor = DefaultModelMonitor(
        instance_type=config["instance_type"],
        instance_count=1,
        max_runtime_in_seconds=1800,
        volume_size_in_gb=20,
        role=role,
    )

    return data_monitor.create_monitoring_schedule(
        monitor_schedule_name="insurance-data-monitoring-schedule",
        endpoint_input=endpoint,
        statistics=f"{data_quality_location}/statistics.json",
        constraints=f"{data_quality_location}/constraints.json",
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        output_s3_uri=data_quality_location,
        enable_cloudwatch_metrics=True,
    )


def create_model_monitor(
    role: str = None,
    endpoint: str = None,
    model_quality_location: str = None,
    ground_truth_location: str = None,
    config: dict = None,
):
    """
    Create a model monitor for the endpoint.

    Parameters:
        role (str): IAM role for SageMaker.
        endpoint_name (str): The SageMaker endpoint name.
        model_quality_location (str): S3 location for model quality output.
        ground_truth_location (str): S3 location for ground truth data.
        config (dict): Configuration dictionary.

    Returns:
        MonitoringSchedule: The monitoring schedule object.
    """
    model_monitor = ModelQualityMonitor(
        instance_type=config["instance_type"],
        instance_count=1,
        max_runtime_in_seconds=1800,
        volume_size_in_gb=20,
        role=role,
    )

    return model_monitor.create_monitoring_schedule(
        monitor_schedule_name="inference-model-monitoring-schedule",
        endpoint_input=EndpointInput(
            endpoint_name=endpoint,
            inference_attribute="0",
            destination="/opt/ml/processing/input_data",
        ),
        problem_type="Regression",
        ground_truth_input=ground_truth_location,
        constraints=f"{model_quality_location}/constraints.json",
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        output_s3_uri=model_quality_location,
        enable_cloudwatch_metrics=True,
    )


def monitoring_schedule_creation(monitoring_schedule, max_wait=7200, poll_interval=60):
    """
    Wait until the monitoring schedule is created and active.

    Parameters:
        monitoring_schedule: The monitoring schedule object.
        max_wait (int): Maximum wait time in seconds. Defaults to 7200 seconds.
        poll_interval (int): Poll interval in seconds. Defaults to 60 seconds.

    Returns:
        bool: True if the monitoring schedule is created and active, False otherwise.
    """
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            status = monitoring_schedule.describe_schedule()["MonitoringScheduleStatus"]
            print(f"Current Monitoring Schedule Status: {status}")
            if status == "Scheduled":
                print("Monitoring schedule is active.")
                return True
        except Exception as e:
            print(f"Waiting for schedule creation. Error: {str(e)}")
        time.sleep(poll_interval)
    print("Timeout: Monitoring schedule was not created in the expected time.")
    return False


if __name__ == "__main__":
    try:
        role = os.environ["ROLE"]
        bucket = os.environ.get("BUCKET", None)
        local_mode = os.environ.get("LOCAL_MODE", "False") == "True"
        s3_location = f"s3://{bucket}"
        data_quality_location = f"{s3_location}/monitoring/data-quality-baseline"
        model_quality_location = f"{s3_location}/monitoring/model-quality-baseline"
        ground_truth_location = f"{s3_location}/monitoring/groundtruth"
        endpoint = os.environ["ENDPOINT"]
        pipeline_model_package_group = os.environ["PIPELINE_MODEL_PACKAGE_GROUP"]
        data_capture_destination = f"{s3_location}/monitoring/data-capture"

        if not bucket:
            raise ValueError("S3 bucket must be specified")

        config = define_config(local_mode, bucket)

        deploy_model(config, pipeline_model_package_group, endpoint, data_capture_destination, role)

        # Ensure the model is deployed and endpoint is in service
        print("Model deployed and endpoint is in service. Starting monitoring schedule creation...")

        data_monitor = create_data_monitor(role, endpoint, data_quality_location, config)
        model_monitor = create_model_monitor(role, endpoint, model_quality_location, ground_truth_location, config)

        # Wait until the monitoring schedules are created and active
        if monitoring_schedule_creation(data_monitor):
            data_monitor.start_monitoring_schedule()
        else:
            print("Failed to create data monitoring schedule.")

        if monitoring_schedule_creation(model_monitor):
            model_monitor.start_monitoring_schedule()
        else:
            print("Failed to create model monitoring schedule.")

    except ImportError as e:
        print(f"Error: {e}")
    except KeyError as e:
        print(f"Environment variable not set: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
