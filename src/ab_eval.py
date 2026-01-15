from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from scipy.stats import chi2_contingency, norm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def ztest_prop(x1, n1, x2, n2):
    p1 = x1 / n1
    p2 = x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1/n1 + 1/n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    pval = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(pval)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--tracking-uri", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--promote", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.log)
    df = df.dropna(subset=["y_true"])  # без y_true метрики честно не посчитать
    if df.empty:
        print("Нет y_true в логах. Для лабы отправляй y_true в /predict.")
        return

    out = []
    for variant in ["A", "B"]:
        part = df[df["variant"] == variant]
        y_true = part["y_true"].astype(int).values
        y_pred = part["y_pred"].astype(int).values

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        out.append((variant, len(part), acc, prec, rec, f1, cm))

    a = [r for r in out if r[0] == "A"][0]
    b = [r for r in out if r[0] == "B"][0]

    print("\n=== METRICS ===")
    for r in out:
        print(f"{r[0]}: n={r[1]} acc={r[2]:.4f} prec={r[3]:.4f} rec={r[4]:.4f} f1={r[5]:.4f}")
        print("CM:\n", r[6], "\n")

    # Статзначимость для accuracy как доли успехов
    x1, n1 = int(round(a[2] * a[1])), a[1]
    x2, n2 = int(round(b[2] * b[1])), b[1]

    chi2, p_chi, _, _ = chi2_contingency([[x1, n1-x1], [x2, n2-x2]])
    z, p_z = ztest_prop(x1, n1, x2, n2)

    print("=== SIGNIFICANCE ===")
    print(f"chi2={chi2:.4f} p_chi={p_chi:.6f}")
    print(f"z={z:.4f} p_z={p_z:.6f}")

    b_better = b[2] > a[2]
    significant = p_chi < 0.05

    decision = "PROMOTE_B" if (b_better and significant) else "KEEP_A"
    print("=== DECISION ===")
    print(decision)

    if args.promote and decision == "PROMOTE_B":
        mlflow.set_tracking_uri(args.tracking_uri)
        client = MlflowClient(tracking_uri=args.tracking_uri)
        vers = client.get_latest_versions(args.model_name, stages=["Staging"])
        if not vers:
            print("Нет Staging версии в Registry.")
            return
        v = vers[0].version
        client.transition_model_version_stage(args.model_name, v, stage="Production", archive_existing_versions=False)
        print(f"Promoted {args.model_name} v{v} -> Production")

if __name__ == "__main__":
    main()
