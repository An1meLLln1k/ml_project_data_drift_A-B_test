from __future__ import annotations
import os, json, csv, hashlib
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from flask import Flask, request, jsonify
from mlflow.tracking import MlflowClient

app = Flask(__name__)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "iris_classifier")
LOG_PATH = os.getenv("LOG_PATH", "/app/logs/ab_log.csv")
traffic_b_percent = int(os.getenv("TRAFFIC_B_PERCENT", "30"))

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)

MODEL_CACHE = {}  # key: stage -> model

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def _bucket(user_id: str) -> int:
    h = hashlib.md5(user_id.encode("utf-8")).hexdigest()
    return int(h, 16) % 100


def _pick_variant(user_id: str) -> str:
    b = _bucket(user_id)
    return "B" if b < traffic_b_percent else "A"


def _model_uri(stage: str) -> str:
    # stage: "Production" / "Staging"
    return f"models:/{MODEL_NAME}/{stage}"


def _load_model(stage: str):
    if stage in MODEL_CACHE:
        return MODEL_CACHE[stage]
    m = mlflow.pyfunc.load_model(_model_uri(stage))
    MODEL_CACHE[stage] = m
    return m


def _get_stage_version(stage: str):
    # Может вернуть None, если стадий нет или не назначены
    try:
        vers = client.get_latest_versions(MODEL_NAME, stages=[stage])
        if not vers:
            return None
        return int(vers[0].version)
    except Exception:
        return None


def _ensure_log_header(path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        return
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts", "user_id", "variant", "stage", "model_version", "features_json", "y_true", "y_pred"])


def _append_log(row: list):
    _ensure_log_header(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def _extract_features(payload: dict) -> dict:
    """
    Поддерживаем 2 формата:
    1) {"user_id":"1", "features": {"sepal_length":..., ...}, "y_true": ...}
    2) {"user_id":"1", "sepal_length":..., "sepal_width":..., ... , "y_true": ...}
    """
    features = payload.get("features")
    if isinstance(features, dict):
        src = features
    else:
        src = payload  # плоский формат

    # собираем только нужные фичи
    x = {}
    missing = []
    for k in FEATURES:
        if k not in src:
            missing.append(k)
            continue
        try:
            x[k] = float(src[k])
        except Exception:
            raise ValueError(f"Feature '{k}' must be numeric, got: {src[k]!r}")
    if missing:
        raise ValueError(f"Missing features: {missing}. Expected: {FEATURES}")
    return x


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "tracking_uri": TRACKING_URI,
        "model": MODEL_NAME,
        "traffic_b_percent": traffic_b_percent
    })


@app.post("/config")
def config():
    global traffic_b_percent
    payload = request.get_json(force=True) or {}
    traffic_b_percent = int(payload.get("traffic_b_percent", traffic_b_percent))
    MODEL_CACHE.clear()
    return jsonify({"traffic_b_percent": traffic_b_percent})


@app.post("/predict")
def predict():
    payload = request.get_json(force=True) or {}

    user_id = str(payload.get("user_id", "0"))
    y_true = payload.get("y_true", None)

    # A/B выбор
    variant = _pick_variant(user_id)
    stage = "Production" if variant == "A" else "Staging"

    # Фичи
    try:
        x = _extract_features(payload)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Модель (если staging нет, падаем на production)
    try:
        model = _load_model(stage)
        model_version = _get_stage_version(stage)
    except Exception:
        stage = "Production"
        model = _load_model(stage)
        model_version = _get_stage_version(stage)

    df = pd.DataFrame([x])

    # Предикт
    y_pred = model.predict(df)[0]

    # аккуратно приводим тип
    if hasattr(y_pred, "item"):
        y_pred_out = y_pred.item()
    else:
        y_pred_out = y_pred

    _append_log([
        datetime.utcnow().isoformat(),
        user_id,
        variant,
        stage,
        model_version,
        json.dumps(x, ensure_ascii=False),
        y_true,
        y_pred_out,
    ])

    return jsonify({
        "variant": variant,
        "stage": stage,
        "model_version": model_version,
        "prediction": y_pred_out
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
