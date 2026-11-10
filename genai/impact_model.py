"""
The trained impact-forecasting models (models/fe_tuned): the ML screen of the gated system, and the
forecast that the ml_llm and gated systems give the agent in its prompt.

For recorded runs the features already exist in dataset/processed/ for snapshots at
T = 15, 30, 45 and 60 s after the change, computed only from data up to T (preprocess.py,
feature_engineer.py), so they are exactly what the models were trained on. Label columns
in those files are never returned.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from train_model import (  # noqa: E402
    BASE_NUMERIC_FEATURES, CATEGORICAL_FEATURES, FE_NUMERIC_FEATURES, SEVERITY_ORDER,
)

SNAPSHOTS = (15, 30, 45, 60)


class ImpactModel:
    def __init__(self, models_dir: Path = REPO_ROOT / "models" / "fe_tuned",
                 processed_dir: Path = REPO_ROOT / "dataset" / "processed"):
        self.clf = joblib.load(self._one(models_dir, "clf_*_tuned.joblib"))    # base features
        self.reg = joblib.load(self._one(models_dir, "reg_*_tuned.joblib"))    # base + FE features
        base = pd.read_csv(processed_dir / "training_dataset.csv")
        fe = pd.read_csv(processed_dir / "training_dataset_fe.csv")
        # same feature selection as tune_model.py (time_into_fault_s is itself a feature)
        self.clf_features = [c for c in BASE_NUMERIC_FEATURES if c in base.columns] + CATEGORICAL_FEATURES
        self.reg_features = [c for c in BASE_NUMERIC_FEATURES + FE_NUMERIC_FEATURES if c in fe.columns] \
            + CATEGORICAL_FEATURES
        key = ["experiment_id", "time_into_fault_s"]
        self.base = base.set_index(key, drop=False)
        self.fe = fe.set_index(key, drop=False)

    @staticmethod
    def _one(models_dir: Path, pattern: str) -> Path:
        """The single model file matching `pattern` (run_pipeline.sh puts exactly one of each in models/fe_tuned)."""
        found = sorted(models_dir.glob(pattern))
        if len(found) != 1:
            raise FileNotFoundError(f"expected one {pattern} in {models_dir}, found {[f.name for f in found]}")
        return found[0]

    def predict(self, experiment_id: str, seconds_after_change: float) -> dict:
        """Forecast for the latest snapshot at or before `seconds_after_change`."""
        T = max((t for t in SNAPSHOTS if t <= seconds_after_change), default=None)
        if T is None:
            return {"available": False, "note": f"first forecast is available {SNAPSHOTS[0]} s after the change"}
        key = (experiment_id, T)
        X_clf = self.base.loc[[key], self.clf_features]
        X_reg = self.fe.loc[[key], self.reg_features]
        probs = self.clf.predict_proba(X_clf)[0]
        return {
            "available": True,
            "snapshot_s": T,
            "predicted_final_severity": SEVERITY_ORDER[int(probs.argmax())],
            "severity_probabilities": {c: round(float(p), 3) for c, p in zip(SEVERITY_ORDER, probs)},
            # the regressor can go slightly below zero; a count of services cannot
            "predicted_blast_radius": round(max(0.0, float(self.reg.predict(X_reg)[0])), 2),
            "reliability_notes": (
                "Severity 'High' forecasts are the most reliable; 'Medium' labels are noisy in the training "
                "data. The blast-radius estimate is low-confidence: the training label counts noisy latency "
                "changes, so treat it as a rough indication only."),
        }
