from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn
from pycaret.classification import setup, compare_models, finalize_model
from sklearn.metrics import accuracy_score


FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "target"


def train_with_pycaret(
    train_csv: str,
    tracking_uri: str,
    experiment_name: str,
    drift_score: float | None = None,
    save_dir: str = "/opt/airflow/logs",
) -> dict:
    # 1) MLflow: tracking + experiment
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    df = pd.read_csv(train_csv)

    # Артефакты: куда писать (важно, чтобы путь был writable в airflow контейнере)
    artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT", "/opt/airflow/mlruns")

    # 2) PyCaret: автологинг в MLflow (он сам стартует run)
    setup(
        data=df,
        target=TARGET,
        session_id=42,
        fold=3,
        log_experiment=True,
        experiment_name=experiment_name,
        verbose=False,
    )

    best = compare_models()
    final = finalize_model(best)

    # 3) Папка для локальных логов (не MLflow)
    save_dir_p = Path(save_dir)
    save_dir_p.mkdir(parents=True, exist_ok=True)

    # 4) Логируем доп. метрики/артефакт в ТЕКУЩИЙ run PyCaret
    # Если по какой-то причине run не активен, создадим один (без конфликта)
    run = mlflow.active_run()
    created_new_run = False
    if run is None:
        run = mlflow.start_run(run_name="best_model_export")
        created_new_run = True

    try:
        y_true = df[TARGET].values
        y_pred = final.predict(df[FEATURES])
        acc = float(accuracy_score(y_true, y_pred))

        mlflow.log_metric("train_acc_quick", acc)
        if drift_score is not None:
            mlflow.log_metric("drift_score", float(drift_score))
        mlflow.log_param("source_csv", train_csv)

        # В этот же run складываем модель как артефакт "model"
        mlflow.sklearn.log_model(final, artifact_path="model")

        run_id = run.info.run_id
    finally:
        # Закрываем run только если мы его сами открыли
        if created_new_run:
            mlflow.end_run()

    return {"export_run_id": run_id, "train_acc_quick": acc}
