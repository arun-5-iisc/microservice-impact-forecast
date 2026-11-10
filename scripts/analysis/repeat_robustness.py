#!/usr/bin/env python
"""
Do the 20 settings that were run twice skew the ML results?

Compares three versions of the dataset, with the pipeline's tuned hyperparameters (no re-tuning) and holdout:
  full         all 244 runs
  keep_first   one run per setting (the later run of each repeated setting dropped)
  keep_second  one run per setting (the earlier run dropped)

Sections: A pipeline-style CV and holdout numbers per version; B every version scored on the same holdout rows;
C repeated CV with the same folds for every version; D how similar the two runs of a repeated setting are.
Usage: python scripts/analysis/repeat_robustness.py [output folder]   (default results/reliability/robustness/)
"""
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]          # repository root
OUT = (Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "reliability" / "robustness").resolve()
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "scripts"))
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor                            # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error  # noqa: E402
from sklearn.model_selection import GroupKFold              # noqa: E402
from sklearn.pipeline import Pipeline                       # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402
from xgboost import XGBClassifier                           # noqa: E402

from train_model import (                                   # noqa: E402
    BASE_NUMERIC_FEATURES, CATEGORICAL_FEATURES, FE_NUMERIC_FEATURES, SEVERITY_ORDER,
    _build_preprocessor, baseline_cv_regression, baseline_cv_report,
    early_blast_radius, full_rule_severity, setting_groups,
)
logging.getLogger().setLevel(logging.WARNING)

N_FOLDS = 5
SEEDS = list(range(int(os.environ.get("SEEDS", 10))))
XGB = json.loads((REPO / "results/baseline_clf_xgb/best_params.json").read_text())["xgb_clf"]
REG = json.loads((REPO / "results/fe_reg_xgb/best_params.json").read_text())["xgb_reg"]   # the chosen regressor
HOLDOUT = set((REPO / "dataset/processed/holdout_ids.txt").read_text().split())
LBL = {c: i for i, c in enumerate(SEVERITY_ORDER)}
SEV = np.array(SEVERITY_ORDER)
VARIANTS = ["full", "keep_first", "keep_second"]


def load(name: str) -> pd.DataFrame:
    """Load a processed CSV in its original row order (row order affects the models' sampling)."""
    df = pd.read_csv(REPO / "dataset/processed" / name)
    df["setting"] = setting_groups(df)
    runs = df.drop_duplicates("experiment_id")[["experiment_id", "setting"]].copy()
    runs["num"] = runs["experiment_id"].str[4:].astype(int)
    runs = runs.sort_values("num")
    runs["run_no"] = runs.groupby("setting").cumcount() + 1
    runs["n_runs"] = runs.groupby("setting")["experiment_id"].transform("size")
    df["run_no"] = df["experiment_id"].map(runs.set_index("experiment_id")["run_no"])
    df["n_runs"] = df["experiment_id"].map(runs.set_index("experiment_id")["n_runs"])
    df["holdout"] = df["experiment_id"].isin(HOLDOUT)
    assert df["n_runs"].max() == 2, "expected at most 2 runs per setting"
    return df


def in_variant(df: pd.DataFrame, v: str) -> np.ndarray:
    if v == "full":
        return np.ones(len(df), bool)
    if v == "keep_first":
        return (df["run_no"] == 1).values
    return ((df["n_runs"] == 1) | (df["run_no"] == 2)).values


def features(df):
    num = [c for c in BASE_NUMERIC_FEATURES + FE_NUMERIC_FEATURES if c in df.columns]
    return num, num + CATEGORICAL_FEATURES


def fit_clf(num, X, y, weighted):
    pipe = Pipeline([("pre", _build_preprocessor(num)), ("model", XGBClassifier(**XGB))])
    kw = {"model__sample_weight": compute_sample_weight("balanced", y)} if weighted else {}
    return pipe.fit(X, y, **kw)


def fit_reg(num, X, y):
    return Pipeline([("pre", _build_preprocessor(num)), ("model", XGBRegressor(**REG))]).fit(X, y)


def clf_scores(true, pred, rule, t):
    """Pooled scores for one set of rows (labels as strings)."""
    changed = rule != true
    t15 = t == 15
    return {
        "rows": len(true),
        "acc": accuracy_score(true, pred),
        "macro_f1": f1_score(true, pred, labels=SEVERITY_ORDER, average="macro", zero_division=0),
        "medium_f1": f1_score(true, pred, labels=["Medium"], average="macro", zero_division=0),
        "changed_rows": int(changed.sum()),
        "changed_acc": (pred[changed] == true[changed]).mean() if changed.any() else np.nan,
        "t15_acc": accuracy_score(true[t15], pred[t15]),
        "rule_acc": accuracy_score(true, rule),
        "rule_macro_f1": f1_score(true, rule, labels=SEVERITY_ORDER, average="macro", zero_division=0),
        "rule_t15_acc": accuracy_score(true[t15], rule[t15]),
    }


def reg_scores(true, pred, early, t):
    t15 = t == 15
    return {
        "mae": mean_absolute_error(true, pred),
        "t15_mae": mean_absolute_error(true[t15], pred[t15]),
        "base_mae": mean_absolute_error(true, early),
        "base_t15_mae": mean_absolute_error(true[t15], early[t15]),
    }


def fmt(df, cols=None):
    return (df if cols is None else df[cols]).to_string(index=False, float_format=lambda x: f"{x:.3f}")


base = load("training_dataset.csv")   # classifier features (pipeline: base features)
fe = load("training_dataset_fe.csv")  # regressor features (pipeline: FE features)
num_c, feat_c = features(base)
num_r, feat_r = features(fe)
report = []


def out(s=""):
    print(s, flush=True)
    report.append(s)


rep_settings = base.loc[base["n_runs"] == 2, "setting"].unique()
out(f"Runs: {base['experiment_id'].nunique()}  settings: {base['setting'].nunique()}  "
    f"repeated settings: {len(rep_settings)}  "
    f"(training {base.loc[(base.n_runs == 2) & ~base.holdout, 'setting'].nunique()}, "
    f"holdout {base.loc[(base.n_runs == 2) & base.holdout, 'setting'].nunique()})")
out(f"Holdout runs: {base.loc[base.holdout, 'experiment_id'].nunique()}")

# D. How similar are the two runs of a repeated setting?
pair = base[base["n_runs"] == 2].pivot_table(index=["setting", "time_into_fault_s"], columns="run_no",
                                             values="severity", aggfunc="first")
pair_r = fe[fe["n_runs"] == 2].pivot_table(index=["setting", "time_into_fault_s"], columns="run_no",
                                           values="blast_radius_final", aggfunc="first")
per_setting = base[base["n_runs"] == 2].drop_duplicates("experiment_id")[["setting", "fault_type", "run_no", "severity"]]
out("\n=== D. The 20 repeated settings ===")
out(f"Same severity label in both runs: {(pair[1] == pair[2]).mean():.0%} of rows "
    f"({(pair[1] == pair[2]).groupby('setting').all().sum()}/{len(rep_settings)} settings fully identical)")
out(f"Blast radius (final) differs between runs by {np.abs(pair_r[1] - pair_r[2]).mean():.2f} services on average "
    f"(identical in {(pair_r[1] == pair_r[2]).mean():.0%} of rows)")
mix = pd.DataFrame({
    "repeated settings": per_setting[per_setting.run_no == 1]["fault_type"].value_counts(normalize=True),
    "all settings": base.drop_duplicates("setting")["fault_type"].value_counts(normalize=True),
}).fillna(0)
out("Fault-type mix (share of settings):\n" + mix.to_string(float_format=lambda x: f"{x:.0%}"))
sev = pd.DataFrame({
    "repeated settings (rows)": base.loc[base.n_runs == 2, "severity"].value_counts(normalize=True),
    "all (rows)": base["severity"].value_counts(normalize=True),
}).reindex(SEVERITY_ORDER)
out("Severity mix:\n" + sev.to_string(float_format=lambda x: f"{x:.0%}"))

# A. Pipeline-style numbers per version
out("\n=== A. Pipeline-style evaluation per dataset version (same as run_pipeline.sh) ===")
rows_a = []
for v in VARIANTS:
    t0 = time.time()
    mc, mr = in_variant(base, v), in_variant(fe, v)
    tr, te = base[mc & ~base.holdout].reset_index(drop=True), base[mc & base.holdout].reset_index(drop=True)
    trr, ter = fe[mr & ~fe.holdout].reset_index(drop=True), fe[mr & fe.holdout].reset_index(drop=True)

    # CV classifier: unweighted per fold, as _evaluate_classifier does; macro-F1 = mean over folds
    y, g = tr["severity"].map(LBL).values, tr["setting"].values
    oof, f1s = np.empty(len(tr), int), []
    for a, b in GroupKFold(N_FOLDS).split(tr, y, g):
        oof[b] = fit_clf(num_c, tr[feat_c].iloc[a], y[a], weighted=False).predict(tr[feat_c].iloc[b])
        f1s.append(f1_score(y[b], oof[b], labels=[0, 1, 2], average="macro", zero_division=0))
    true, rule, t = tr["severity"].values, full_rule_severity(tr), tr["time_into_fault_s"].values
    s = clf_scores(true, SEV[oof], rule, t)
    s["macro_f1"] = np.mean(f1s)
    brep = baseline_cv_report(true, rule, g)
    s["rule_macro_f1"] = brep["macro avg"]["f1-score"]

    # CV regressor: MAE = mean over folds, as _evaluate_regressor does
    yr, gr = trr["blast_radius_final"].values.astype(float), trr["setting"].values
    maes = [mean_absolute_error(yr[b], fit_reg(num_r, trr[feat_r].iloc[a], yr[a]).predict(trr[feat_r].iloc[b]))
            for a, b in GroupKFold(N_FOLDS).split(trr, yr, gr)]
    s["mae"] = np.mean(maes)
    s["base_mae"] = baseline_cv_regression(yr, early_blast_radius(trr), gr)["mae_mean"]
    rows_a.append({"version": v, "split": "cv", "runs": tr["experiment_id"].nunique(), **s})

    # Holdout: weighted XGB classifier refit on all training rows, as the saved model; regressor refit
    clf = fit_clf(num_c, tr[feat_c], y, weighted=True)
    reg = fit_reg(num_r, trr[feat_r], yr)
    true_h, rule_h, t_h = te["severity"].values, full_rule_severity(te), te["time_into_fault_s"].values
    s = clf_scores(true_h, SEV[clf.predict(te[feat_c]).astype(int)], rule_h, t_h)
    s.update(reg_scores(ter["blast_radius_final"].values.astype(float), reg.predict(ter[feat_r]),
                        early_blast_radius(ter), ter["time_into_fault_s"].values))
    rows_a.append({"version": v, "split": "holdout", "runs": te["experiment_id"].nunique(), **s})
    print(f"  [{v}] {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

A = pd.DataFrame(rows_a)
A.to_csv(OUT / "A_pipeline_style.csv", index=False)
cols = ["version", "split", "runs", "acc", "rule_acc", "macro_f1", "rule_macro_f1", "medium_f1",
        "changed_acc", "t15_acc", "rule_t15_acc", "mae", "base_mae"]
out(fmt(A, cols))
out("(cv macro_f1 / mae are fold means like the pipeline; everything else is pooled)")

# B. Holdout cross-evaluation
out("\n=== B. Holdout: every version's model scored on the same holdout rows ===")
hb, hr = base[base.holdout].reset_index(drop=True), fe[fe.holdout].reset_index(drop=True)
eval_sets = {
    "all 50 runs": np.ones(len(hb), bool),
    "first runs (45)": in_variant(hb, "keep_first"),
    "second runs (45)": in_variant(hb, "keep_second"),
    "single-run settings (40)": (hb["n_runs"] == 1).values,
}
eval_sets_r = {
    "all 50 runs": np.ones(len(hr), bool),
    "first runs (45)": in_variant(hr, "keep_first"),
    "second runs (45)": in_variant(hr, "keep_second"),
    "single-run settings (40)": (hr["n_runs"] == 1).values,
}
rows_b = []
for v in VARIANTS:
    mc, mr = in_variant(base, v), in_variant(fe, v)
    tr, trr = base[mc & ~base.holdout], fe[mr & ~fe.holdout]
    clf = fit_clf(num_c, tr[feat_c], tr["severity"].map(LBL).values, weighted=True)
    reg = fit_reg(num_r, trr[feat_r], trr["blast_radius_final"].values.astype(float))
    pc = SEV[clf.predict(hb[feat_c]).astype(int)]
    pr = reg.predict(hr[feat_r])
    for name, m in eval_sets.items():
        mr_ = eval_sets_r[name]
        s = clf_scores(hb["severity"].values[m], pc[m], full_rule_severity(hb)[m], hb["time_into_fault_s"].values[m])
        s.update(reg_scores(hr["blast_radius_final"].values.astype(float)[mr_], pr[mr_],
                            early_blast_radius(hr)[mr_], hr["time_into_fault_s"].values[mr_]))
        rows_b.append({"trained_on": v, "scored_on": name, **s})
B = pd.DataFrame(rows_b)
B.to_csv(OUT / "B_holdout_cross.csv", index=False)
out(fmt(B, ["trained_on", "scored_on", "acc", "rule_acc", "macro_f1", "rule_macro_f1",
            "changed_acc", "t15_acc", "mae", "base_mae"]))

# C. Repeated CV, identical setting-level folds for every version
out(f"\n=== C. Repeated CV ({len(SEEDS)} shuffles), same folds for every version ===")
trb = base[~base.holdout].reset_index(drop=True)
trf = fe[~fe.holdout].reset_index(drop=True)
settings = np.array(sorted(trb["setting"].unique()))
yb, yf = trb["severity"].map(LBL).values, trf["blast_radius_final"].values.astype(float)
rule_b, early_f = full_rule_severity(trb), early_blast_radius(trf)
score_on = {
    "own rows": None,
    "single-run settings": ("n_runs", 1),
}
rows_c = []
t0 = time.time()
for seed in SEEDS:
    perm = np.random.default_rng(seed).permutation(settings)
    fold_of = {s: i % N_FOLDS for i, s in enumerate(perm)}
    fb, ff = trb["setting"].map(fold_of).values, trf["setting"].map(fold_of).values
    for v in VARIANTS:
        vb, vf = in_variant(trb, v), in_variant(trf, v)
        oof_u = np.full(len(trb), -1)
        oof_w = np.full(len(trb), -1)
        oof_r = np.full(len(trf), np.nan)
        for k in range(N_FOLDS):
            a, b = np.where(vb & (fb != k))[0], np.where(vb & (fb == k))[0]
            Xa, Xb = trb[feat_c].iloc[a], trb[feat_c].iloc[b]
            oof_u[b] = fit_clf(num_c, Xa, yb[a], weighted=False).predict(Xb)
            oof_w[b] = fit_clf(num_c, Xa, yb[a], weighted=True).predict(Xb)
            a, b = np.where(vf & (ff != k))[0], np.where(vf & (ff == k))[0]
            oof_r[b] = fit_reg(num_r, trf[feat_r].iloc[a], yf[a]).predict(trf[feat_r].iloc[b])
        for name, spec in score_on.items():
            mb = vb if spec is None else (trb[spec[0]] == spec[1]).values
            mf = vf if spec is None else (trf[spec[0]] == spec[1]).values
            t_b, t_f = trb["time_into_fault_s"].values[mb], trf["time_into_fault_s"].values[mf]
            su = clf_scores(trb["severity"].values[mb], SEV[oof_u[mb]], rule_b[mb], t_b)
            sw = clf_scores(trb["severity"].values[mb], SEV[oof_w[mb]], rule_b[mb], t_b)
            sr = reg_scores(yf[mf], oof_r[mf], early_f[mf], t_f)
            rows_c.append({"seed": seed, "version": v, "scored_on": name, "rows": su["rows"],
                           "acc": su["acc"], "macro_f1": su["macro_f1"], "medium_f1": su["medium_f1"],
                           "changed_acc": su["changed_acc"], "t15_acc": su["t15_acc"],
                           "acc_w": sw["acc"], "macro_f1_w": sw["macro_f1"],
                           "rule_acc": su["rule_acc"], "rule_macro_f1": su["rule_macro_f1"],
                           "mae": sr["mae"], "t15_mae": sr["t15_mae"], "base_mae": sr["base_mae"]})
    print(f"  seed {seed} done ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)

C = pd.DataFrame(rows_c)
C.to_csv(OUT / "C_repeated_cv_raw.csv", index=False)
metrics = ["acc", "macro_f1", "medium_f1", "changed_acc", "t15_acc", "acc_w", "macro_f1_w", "mae", "t15_mae",
           "rule_acc", "rule_macro_f1", "base_mae"]
summ = C.groupby(["scored_on", "version"], sort=False)[metrics].agg(["mean", "std"])
summ.columns = [f"{m}_{s}" for m, s in summ.columns]
summ.to_csv(OUT / "C_repeated_cv_summary.csv")
show = pd.DataFrame({m: summ[f"{m}_mean"].map("{:.3f}".format) + " ± " + summ[f"{m}_std"].map("{:.3f}".format)
                     for m in ["acc", "macro_f1", "medium_f1", "changed_acc", "t15_acc", "mae"]})
out(show.to_string())
out("(acc_w/macro_f1_w = XGB trained with class weights, like the saved model; see CSV)")
out(fmt(summ[[f"{m}_mean" for m in ["acc_w", "macro_f1_w", "rule_acc", "rule_macro_f1", "base_mae"]]].reset_index()))

# Paired differences per seed: full minus each one-run version, on the common rows
out("\nPaired difference per shuffle, scored on single-run settings (same rows for every version):")
common = C[C["scored_on"] == "single-run settings"].set_index(["seed", "version"])
diff_rows = []
for other in ["keep_first", "keep_second"]:
    for m in ["acc", "macro_f1", "acc_w", "macro_f1_w", "changed_acc", "t15_acc", "mae"]:
        d = common.xs("full", level="version")[m] - common.xs(other, level="version")[m]
        diff_rows.append({"full minus": other, "metric": m, "mean_diff": d.mean(), "sd": d.std(),
                          "full_better_in": f"{int(((d < 0) if m == 'mae' else (d > 0)).sum())}/{len(d)}"})
D = pd.DataFrame(diff_rows)
D.to_csv(OUT / "C_paired_differences.csv", index=False)
out(fmt(D))
d12 = common.xs("keep_first", level="version")[["acc", "macro_f1", "mae"]] - \
      common.xs("keep_second", level="version")[["acc", "macro_f1", "mae"]]
out(f"Run-to-run noise (keep_first minus keep_second): acc {d12['acc'].mean():+.3f} ± {d12['acc'].std():.3f}, "
    f"macro-F1 {d12['macro_f1'].mean():+.3f} ± {d12['macro_f1'].std():.3f}, MAE {d12['mae'].mean():+.3f} ± {d12['mae'].std():.3f}")

(OUT / "report.txt").write_text("\n".join(report) + "\n")
print(f"\nSaved to {OUT}", file=sys.stderr)
