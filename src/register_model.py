import mlflow
from mlflow.tracking import MlflowClient
import json
import os
import yaml

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
config_path = os.path.join(PATH, "params.yaml")
run_id_path = os.path.join(PATH, "reports", "run_id.json")
model_info_path = os.path.join(PATH, "reports", "model_info.json")
registry_info_path = os.path.join(PATH, "reports", "registry_info.json")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

MLFLOW_TRACKING_URI = config["global_variables"]["tracking_uri"]
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

with open(run_id_path, "r") as f:
    run_id_data = json.load(f)
parent_run_id = run_id_data.get("parent_run_id")
print(f"Fetching model for parent_run_id: {parent_run_id}")

with open(model_info_path, "r") as f:
    model_info = json.load(f)

MODEL_NAME = model_info["model_name"]
MODEL_URI = model_info["logged_model_uri"]
print(f"🔹 Model to register: {MODEL_NAME}")
print(f"🔹 Source URI: {MODEL_URI}")

try:
    result = mlflow.register_model(model_uri=MODEL_URI, name=MODEL_NAME)
    print(f"✅ Model registered: {result.name}, version: {result.version}")
except Exception as e:
    print(f"⚠️ Model may already be registered: {e}")
    result = client.get_latest_versions(MODEL_NAME, stages=["None"])[-1]

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=result.version,
    stage="Production",
    archive_existing_versions=True
)

print(f"🚀 Model version {result.version} promoted to Production.")

registry_info = {
    "model_name": MODEL_NAME,
    "model_version": result.version,
    "stage": "Production"
}

with open(registry_info_path, "w") as f:
    json.dump(registry_info, f, indent=4)

print(f"Registry info saved to {registry_info_path}")
