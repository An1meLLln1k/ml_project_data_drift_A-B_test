from __future__ import annotations
import os
import pandas as pd
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

from src.data_io import ensure_baseline_csv, generate_current_csv, FEATURES
from src.drift import drift_report
from src.train import train_with_pycaret
from src.register import register_run_model

@dag(
    schedule="@daily",
    start_date=days_ago(1),
    catchup=False,
    tags=["ml", "drift", "automl", "ab"]
)
def ml_process_from_drift_to_ab():

    @task
    def make_data():
        baseline_path = ensure_baseline_csv("/opt/airflow/data")
        shift = float(os.getenv("DRIFT_SHIFT", "0.0"))
        current_path = generate_current_csv("/opt/airflow/data", shift=shift)
        return {"baseline": str(baseline_path), "current": str(current_path), "shift": shift}

    @task
    def calc_drift(paths: dict):
        df_ref = pd.read_csv(paths["baseline"])
        df_cur = pd.read_csv(paths["current"])
        bins = int(os.getenv("DRIFT_BINS", "10"))
        rep = drift_report(df_ref, df_cur, FEATURES, bins=bins)
        threshold = float(os.getenv("DRIFT_THRESHOLD", "0.2"))
        drift_score = rep["psi_max"]
        drift_detected = drift_score > threshold
        return {"drift_detected": drift_detected, "drift_score": drift_score, "report": rep, **paths}

    @task.branch
    def branch_on_drift(drift: dict):
        return "retrain_and_register" if drift["drift_detected"] else "no_retrain"

    @task(task_id="no_retrain")
    def no_retrain(_drift: dict):
        return "Drift not detected. No retraining."

    @task(task_id="retrain_and_register")
    def retrain_and_register(drift: dict):
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        model_name = os.getenv("MLFLOW_MODEL_NAME", "iris_classifier")
        exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "pycaret_automl")

        res = train_with_pycaret(
            train_csv=drift["current"],
            tracking_uri=tracking_uri,
            experiment_name=exp_name,
            drift_score=drift["drift_score"],
            save_dir="/opt/airflow/logs"
        )
        reg = register_run_model(tracking_uri, model_name, res["export_run_id"], stage="Staging")
        return {"train": res, "registry": reg, "drift": drift}

    paths = make_data()
    drift = calc_drift(paths)
    decision = branch_on_drift(drift)
    decision >> [no_retrain(drift), retrain_and_register(drift)]

ml_process_from_drift_to_ab()
