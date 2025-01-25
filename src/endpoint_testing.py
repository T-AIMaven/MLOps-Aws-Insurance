import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sagemaker.deserializers import JSONDeserializer
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer, JSONSerializer
from sagemaker.workflow.pipeline_context import LocalPipelineSession, PipelineSession

# Load environment variables if not provided
load_dotenv()

DATA_FILEPATH = Path(__file__).parent.parent / "data" / "clean" / "clean-insurance.csv"
ENDPOINT = os.environ["ENDPOINT"]
bucket = os.environ.get("BUCKET", None)
LOCAL_MODE = os.environ.get("LOCAL_MODE", "False") == "True"

# Configure pipeline session
pipeline_session = PipelineSession(default_bucket=bucket) if not LOCAL_MODE else None

if LOCAL_MODE:
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

predictor = Predictor(
    endpoint_name=ENDPOINT,
    serializer=CSVSerializer(),
    sagemaker_session=config["session"],
)

data = pd.read_csv(DATA_FILEPATH)
data = data.drop("charges", axis=1)

payload = data.iloc[:3].to_csv(header=False, index=False)
print(f"Payload:\n{payload}")

try:
    response = predictor.predict(payload, initial_args={"ContentType": "text/csv"})
    response = json.loads(response.decode("utf-8"))
    print(json.dumps(response, indent=2))
except Exception as e:
    print(e)


sample = {
    "age": 18,
    "sex": "male",
    "bmi": 33.7,
    "children": 1,
    "smoker": "no",
    "region": "southeast",
}

predictor = Predictor(
    endpoint_name=ENDPOINT,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
    sagemaker_session=config["session"],
)

try:
    response = predictor.predict(sample)
    print(response)
except Exception as e:
    print(e)
