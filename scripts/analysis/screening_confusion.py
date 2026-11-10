"""
Confusion matrices of the untuned severity classifiers (the screening of train_model.py).

train_model.py reports the scores of the untuned models but does not save their predictions. This script repeats
the same grouped cross-validation on the training runs, with the same models, weights and folds, for both feature
sets, and saves the confusion matrices. It first checks that each accuracy equals the one train_model.py reported.
No model is saved and the holdout is not used.

Usage:  venv/bin/python scripts/analysis/screening_confusion.py   -> results/reliability/screening_confusion/
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import train_model as T  # noqa: E402

PROC = ROOT / "dataset" / "processed"
OUT = ROOT / "results" / "reliability" / "screening_confusion"
SETS = (("Base", "training_dataset.csv", "baseline"), ("Engineered", "training_dataset_fe.csv", "fe"))


def main() -> None:
    warnings.filterwarnings("ignore")
    holdout = set((PROC / "holdout_ids.txt").read_text().split())
    rows = []
    for label, csv, stage in SETS:
        reported = pd.read_csv(ROOT / "results" / stage / "classification_report.csv").set_index("model").accuracy
        d = pd.read_csv(PROC / csv)
        d = d[~d.experiment_id.isin(holdout)].reset_index(drop=True)
        numeric = [c for c in T.BASE_NUMERIC_FEATURES + T.FE_NUMERIC_FEATURES if c in d.columns]
        X, groups = d[numeric + T.CATEGORICAL_FEATURES], T.setting_groups(d)
        le = LabelEncoder()
        le.classes_ = np.array(T.SEVERITY_ORDER)
        y = le.transform(d.severity)
        for name, estimator in T._classifiers().items():
            # as in train_model.main: XGBoost gets the balanced weights as sample weights
            _, _, _, pred, true = T._evaluate_classifier(name, estimator, X, y, groups, le, numeric,
                                                         balanced_weights=(name == "XGBoost"))
            cm = confusion_matrix(true, pred, labels=range(len(le.classes_)))
            accuracy = np.trace(cm) / cm.sum()
            if abs(accuracy - reported[name]) > 5e-4:
                raise SystemExit(f"{name} ({label}): accuracy {accuracy:.4f} differs from train_model's {reported[name]:.4f}")
            for cls, row in zip(le.classes_, cm):
                rows.append(dict(features=label, model=name, true=str(cls),
                                 **{f"pred_{c}": int(v) for c, v in zip(le.classes_, row)}))
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "confusion.csv", index=False)
    print("Reproduction check: every accuracy equals the one reported by train_model.py. OK")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
