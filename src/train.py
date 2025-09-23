from xgboost import XGBClassifier
import pandas as pd
import os
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import matplotlib.pyplot as plt
import json
import mlflow
import yaml
from sklearn.model_selection import RandomizedSearchCV
from mlflow.models.signature import infer_signature
from mlflow import sklearn as mlflow_sklearn
import dagshub

# Base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
CONFIG_PATH = os.path.join(BASE_DIR, "params.yaml")

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Load config
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

tracking_uri = config["global_variables"]["tracking_uri"]
dagshub.init(
    repo_owner='JayJajoo', 
    repo_name='END_TO_END_MLOPS', 
    mlflow=True,
    token=os.environ.get("DAGSHUB_TOKEN")
)
mlflow.set_tracking_uri(tracking_uri)

# Dataset paths
train_set_path = os.path.join(DATA_DIR, "train.csv")
test_set_path = os.path.join(DATA_DIR, "test.csv")
model_path = os.path.join(MODEL_DIR, "xgb_model.pkl")

# Load datasets
train_set = pd.read_csv(train_set_path)
test_set = pd.read_csv(test_set_path)
x_train, y_train = train_set.iloc[:, :-1], train_set.iloc[:, -1]
x_test, y_test = test_set.iloc[:, :-1], test_set.iloc[:, -1]

# Training config
params_grid = config["training"]["hyper_params"]
random_state = config["training"]["random_state"]
MODEL_NAME = "IrisXGBoostClassifier"

xgb = XGBClassifier()
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=params_grid,
    n_iter=1,
    n_jobs=-1,
    random_state=random_state,
    scoring="accuracy"
)

with mlflow.start_run(run_name="mlops_xgb_parent_run_random_search") as parent_run:
    random_search.fit(x_train, y_train)

    # Log all candidate models
    for i, candidate_params in enumerate(random_search.cv_results_["params"]):
        with mlflow.start_run(
            run_name=f"mlops_xgb_candidate_model_{i}",
            experiment_id=parent_run.info.experiment_id,
            nested=True,
        ) as child_run:
            mlflow.log_params(candidate_params)
            mlflow.log_metric("cv_mean_accuracy", random_search.cv_results_["mean_test_score"][i])
            mlflow.log_metric("cv_std_accuracy", random_search.cv_results_["std_test_score"][i])

            candidate_model = XGBClassifier(
                **candidate_params,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=random_state
            )
            candidate_model.fit(x_train, y_train)

            signature = infer_signature(x_train, candidate_model.predict(x_train))
            mlflow_sklearn.log_model(
                sk_model=candidate_model,
                artifact_path=f"candidate_model_{i}",
                input_example=x_train.iloc[:5],
                signature=signature,
            )

    # Best model evaluation
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred, output_dict=True)
    weighted = clf_report["weighted avg"]

    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("test_precision_weighted", weighted["precision"])
    mlflow.log_metric("test_recall_weighted", weighted["recall"])
    mlflow.log_metric("test_f1_weighted", weighted["f1-score"])

    # Log and register best model
    signature = infer_signature(x_test, best_model.predict(x_test))
    mlflow_model_info = mlflow_sklearn.log_model(
        sk_model=best_model,
        artifact_path="xgboost_model",
        registered_model_name=MODEL_NAME,
        signature=signature,
        input_example=x_test.iloc[:5],
    )

    # Save pickle
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print("Model Saved!!")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Final Model")
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path, artifact_path="confusion_matrix")

    # Metrics JSON
    metrics_path = os.path.join(REPORTS_DIR, "final_metrics.json")
    metrics_data = {
        "test_accuracy": acc,
        "test_precision_weighted": weighted["precision"],
        "test_recall_weighted": weighted["recall"],
        "test_f1_weighted": weighted["f1-score"],
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    # Run + model info JSON
    model_info_path = os.path.join(REPORTS_DIR, "model_info.json")
    model_info = {
        "parent_run_id": parent_run.info.run_id,
        "model_name": MODEL_NAME,
        "logged_model_uri": mlflow_model_info.model_uri,
        "saved_model_path": model_path,
    }
    with open(model_info_path, "w") as f:
        json.dump(model_info, f, indent=4)

    print(f"\n✅ Final metrics saved to {metrics_path}")
    print(f"✅ Run + model info saved to {model_info_path}")
