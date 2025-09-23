import mlflow
from mlflow.tracking import MlflowClient
import json
import os
import yaml

# Base project path (two levels up from src)
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# Paths to config and model info
config_path = os.path.join(PATH, "params.yaml")
model_info_path = os.path.join(PATH, "reports", "model_info.json")

# Load config
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# MLflow tracking URI
MLFLOW_TRACKING_URI = config["global_variables"]["tracking_uri"]
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load run + model info
with open(model_info_path, "r") as f:
    model_info = json.load(f)

parent_run_id = model_info.get("parent_run_id")
MODEL_NAME = model_info.get("model_name")
MODEL_URI = model_info.get("logged_model_uri")
MODEL_PATH = model_info.get("saved_model_path")

print(f"🔹 Evaluating metrics for model: {MODEL_NAME}")
print(f"🔹 Parent Run ID: {parent_run_id}")
print(f"🔹 MLflow Model URI: {MODEL_URI}")
print(f"🔹 Saved Model Path: {MODEL_PATH}")

# Fetch metrics from MLflow
client = MlflowClient()
run = client.get_run(parent_run_id)

accuracy = run.data.metrics.get("test_accuracy", 0)
precision = run.data.metrics.get("test_precision_weighted", 0)
recall = run.data.metrics.get("test_recall_weighted", 0)
f1_score = run.data.metrics.get("test_f1_weighted", 0)

# Threshold check
THRESHOLD = 0.96

if all([
    accuracy >= THRESHOLD,
    precision >= THRESHOLD,
    recall >= THRESHOLD,
    f1_score >= THRESHOLD
]):
    print("✅ PASS : Model metrics above threshold.")
else:
    print("❌ FAIL : Model metrics below threshold.")
    print(f"Metrics: accuracy={accuracy}, precision={precision}, recall={recall}, f1_score={f1_score}")
    raise AssertionError("Model did not meet the threshold requirements.")
