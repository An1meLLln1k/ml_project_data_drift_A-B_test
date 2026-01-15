from __future__ import annotations
import time
import mlflow
from mlflow.tracking import MlflowClient

def register_run_model(
    tracking_uri: str,
    model_name: str,
    run_id: str,
    stage: str = "Staging"
) -> dict:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)

    # ждём пока MLflow создаст версию
    for _ in range(30):
        info = client.get_model_version(name=model_name, version=mv.version)
        if info.status == "READY":
            break
        time.sleep(1)

    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage=stage,
        archive_existing_versions=False
    )

    return {"model_name": model_name, "version": int(mv.version), "stage": stage}
