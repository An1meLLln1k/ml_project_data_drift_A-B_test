from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "target"

def ensure_baseline_csv(data_dir: str = "/opt/airflow/data") -> Path:
    data_dir_p = Path(data_dir)
    data_dir_p.mkdir(parents=True, exist_ok=True)
    baseline_path = data_dir_p / "baseline.csv"

    if baseline_path.exists():
        return baseline_path

    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]
    # Приводим к “человеческим” именам
    df = df.rename(columns={
        "sepal_length": "sepal_length",
        "sepal_width": "sepal_width",
        "petal_length": "petal_length",
        "petal_width": "petal_width",
        "target": "target"
    })

    train_df, _ = train_test_split(df, test_size=0.35, random_state=42, stratify=df["target"])
    train_df.to_csv(baseline_path, index=False)
    return baseline_path

def generate_current_csv(
    data_dir: str = "/opt/airflow/data",
    shift: float = 0.0,
    seed: int = 43
) -> Path:
    """
    current = тестовая часть iris, но с искусственным сдвигом распределения.
    """
    data_dir_p = Path(data_dir)
    data_dir_p.mkdir(parents=True, exist_ok=True)
    current_path = data_dir_p / "current.csv"

    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]
    df = df.rename(columns={
        "sepal_length": "sepal_length",
        "sepal_width": "sepal_width",
        "petal_length": "petal_length",
        "petal_width": "petal_width",
        "target": "target"
    })

    _, test_df = train_test_split(df, test_size=0.35, random_state=42, stratify=df["target"])
    test_df = test_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Сдвиг делаем на паре фич, чтобы PSI сработал
    if shift != 0.0:
        test_df["petal_length"] = test_df["petal_length"] + shift
        test_df["petal_width"]  = test_df["petal_width"] + (shift * 0.4)

    test_df.to_csv(current_path, index=False)
    return current_path
