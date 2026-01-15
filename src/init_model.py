from __future__ import annotations
import os
from src.data_io import ensure_baseline_csv
from src.train import train_with_pycaret
from src.register import register_run_model

def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    model_name = os.getenv("MLFLOW_MODEL_NAME", "iris_classifier")
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "pycaret_automl")

    baseline = ensure_baseline_csv("/opt/airflow/data")
    res = train_with_pycaret(
        train_csv=str(baseline),
        tracking_uri=tracking_uri,
        experiment_name=exp_name,
        drift_score=None,
        save_dir="/opt/airflow/logs"
    )
    reg = register_run_model(tracking_uri, model_name, res["export_run_id"], stage="Production")
    print("INIT PRODUCTION OK:", reg)

if __name__ == "__main__":
    main()
