"""Phase 8 — the null, the challengers, and the ablation ladder.

One row per (origin, unit); features dated <= T; label = birth mass in (T, T+h].

**The null boundary is the paper's central argument**, so it is enforced here as
data rather than as intent. A feature belongs to the null if it is computable
from the unit's own count series alone; to the challenger if it needs graph
topology, another unit, or people. Borderline goes to the null, which makes the
null strictly harder to beat.

    null   : exposure (offset), paper velocity (1y, 2y), YoY-quarter
             acceleration, confirmed-birth persistence
    +sem   : edge velocity, density, density change, pace of collaboration
    +social: social breadth, author influx
    +embed : embedding spread, embedding influx

The ladder is nested: each rung adds a family to the one before, so the marginal
log-score gain per rung is that family's information value.

Missingness is imputed at the TUNE median with its indicator retained as a
feature, never at zero. A region with no T-1 counterpart is not a region with
zero growth, and 52% of rows are in that position.

Standardization is fit on tune origins only and applied unchanged to test, or
the test distribution leaks into the transform.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.manifest import Manifest, load_config  # noqa: E402

MDIR = Path("data/models")

NULL_FEATS = ["paper_velocity_1y", "paper_velocity_2y", "yoy_quarter_accel",
              "confirmed_births_mass"]
SEM_FEATS = ["edge_velocity", "density", "density_change", "pace_collaboration"]
SOCIAL_FEATS = ["social_breadth", "author_influx"]
EMB_FEATS = ["embedding_spread", "embedding_influx"]
INDICATORS = ["graph_delta_missing", "embedding_missing", "yoy_quarter_accel_missing"]

LADDER = [
    ("null", NULL_FEATS),
    ("+semantic_relational", NULL_FEATS + SEM_FEATS),
    ("+social", NULL_FEATS + SEM_FEATS + SOCIAL_FEATS),
    ("full", NULL_FEATS + SEM_FEATS + SOCIAL_FEATS + EMB_FEATS),
]


def build_table(cfg: dict, k: int, horizon: int) -> pd.DataFrame:
    feats = pq.read_table("data/graphs/features/region_features.parquet").to_pandas()
    feats = feats[feats.k == k].copy()
    hist = pq.read_table("data/graphs/features/birth_history.parquet").to_pandas()
    hist = hist[hist.k == k][["origin", "unit_id", "confirmed_births_mass",
                              "confirmed_births_count"]]
    tgt = pq.read_table("data/registry/attachment/targets.parquet").to_pandas()
    tgt = tgt[(tgt.k == k) & (tgt.horizon == horizon)][["origin", "unit_id", "target_mass"]]

    df = feats.merge(hist, on=["origin", "unit_id"], how="left")
    df = df.merge(tgt, on=["origin", "unit_id"], how="left")
    # A unit with no births has target 0. Absent from the targets table means
    # zero births, not missing data -- the one place a zero fill is correct.
    df["target_mass"] = df["target_mass"].fillna(0.0)
    df["confirmed_births_mass"] = df["confirmed_births_mass"].fillna(0.0)
    # Only origins whose horizon is resolvable against the registry.
    df = df[df.origin + horizon <= cfg["registry"]["complete_through"]]
    df = df[df.exposure > 0]
    return df


def _prep(df: pd.DataFrame, feats: list[str], tune_mask: np.ndarray):
    X = df[feats + INDICATORS].copy()
    for c in feats:
        med = X.loc[tune_mask, c].median()
        if not np.isfinite(med):
            med = 0.0
        X[c] = X[c].fillna(med)
        # Winsorize at the tune 99th percentile, per the spec: computed within
        # the fitting set, never pooled across origins.
        hi = X.loc[tune_mask, c].quantile(0.99)
        if np.isfinite(hi):
            X[c] = X[c].clip(upper=hi)
    for c in INDICATORS:
        X[c] = X[c].astype(float)
    mu = X.loc[tune_mask].mean()
    sd = X.loc[tune_mask].std().replace(0, 1.0)
    return ((X - mu) / sd).astype(float)


def poisson_logscore(y: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """k ln(lam) - lam. The ln(k!) term cancels in every model-vs-null gain."""
    lam = np.clip(lam, 1e-9, None)
    return y * np.log(lam) - lam


def run(cfg: dict, k: int, horizon: int) -> dict:
    df = build_table(cfg, k, horizon).reset_index(drop=True)
    tune_lo, tune_hi = cfg["evaluation"]["origins_tune"]
    tune = ((df.origin >= tune_lo) & (df.origin <= tune_hi)).values
    test = ~tune
    if tune.sum() < 50 or test.sum() < 50:
        return {"k": k, "horizon": horizon, "skipped": "insufficient rows",
                "tune_rows": int(tune.sum()), "test_rows": int(test.sum())}

    y = df.target_mass.values.astype(float)
    off = np.log(df.exposure.values.astype(float))
    preds, out = {}, []

    for name, feats in LADDER:
        X = _prep(df, feats, tune)
        Xc = sm.add_constant(X, has_constant="add")
        glm = sm.GLM(y[tune], Xc[tune], family=sm.families.Poisson(),
                     offset=off[tune]).fit(maxiter=200)
        lam = np.asarray(glm.predict(Xc, offset=off), dtype=float)
        preds[name] = lam
        out.append({"model": f"glm/{name}", "n_features": len(feats)})

    gbm_lam = None
    X = _prep(df, LADDER[-1][1], tune)
    # 276 tune rows will not support 300 unregularized boosting iterations: the
    # first fit reached a test prediction ratio of 0.41, i.e. it under-predicted
    # total births by 59%. Early stopping on a held-out slice of TUNE (never
    # test) plus a leaf-count floor is the minimum hygiene at this sample size.
    gbm = HistGradientBoostingRegressor(
        loss="poisson", max_iter=500, learning_rate=0.03, max_depth=3,
        min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.25, n_iter_no_change=20,
        random_state=cfg["runtime"]["random_seed"])
    # The offset enters a GBM as an extra feature; there is no offset argument.
    Xg = X.copy()
    Xg["log_exposure"] = off
    gbm.fit(Xg[tune], y[tune])
    gbm_lam = np.clip(gbm.predict(Xg), 1e-9, None)
    preds["gbm_full"] = gbm_lam

    base = poisson_logscore(y, preds["null"])
    res = []
    for name, lam in preds.items():
        s = poisson_logscore(y, lam)
        gain = s - base
        res.append({
            "model": name,
            "tune_gain_total": float(gain[tune].sum()),
            "test_gain_total": float(gain[test].sum()),
            "test_gain_per_row": float(gain[test].mean()),
            "test_rows": int(test.sum()),
            "years_won": int(sum(
                1 for o in sorted(df.origin[test].unique())
                if gain[(df.origin == o).values & test].sum() > 0)),
            "years_total": int(df.origin[test].nunique()),
        })

    MDIR.mkdir(parents=True, exist_ok=True)
    pdf = df[["origin", "k", "unit_id", "size", "exposure", "target_mass"]].copy()
    pdf["horizon"] = horizon
    pdf["split"] = np.where(tune, "tune", "test")
    for name, lam in preds.items():
        pdf[f"lambda_{name}"] = lam
    path = MDIR / f"predictions_k{k}_h{horizon}.parquet"
    pdf.to_parquet(path, compression="zstd", index=False)

    # Calibration is a headline metric in the plan, so it is recorded per model.
    calib = {name: float(lam[test].sum() / max(1e-9, y[test].sum()))
             for name, lam in preds.items()}
    gains_med = {}
    for name, lam in preds.items():
        g = poisson_logscore(y, lam) - base
        gains_med[name] = {"median_per_row": float(np.median(g[test])),
                           "share_rows_positive": float((g[test] > 0).mean())}
    stats = {"k": k, "horizon": horizon, "rows": len(df),
             "test_calibration_ratio": calib, "test_gain_shape": gains_med,
             "tune_rows": int(tune.sum()), "test_rows": int(test.sum()),
             "tune_origins": sorted(int(x) for x in df.origin[tune].unique()),
             "test_origins": sorted(int(x) for x in df.origin[test].unique()),
             "target_mass_total": float(y.sum()),
             "zero_target_share": float((y == 0).mean()),
             "ladder": res}
    Manifest.build(str(path), phase="8",
                   inputs=["data/graphs/features/region_features.parquet",
                           "data/graphs/features/birth_history.parquet",
                           "data/registry/attachment/targets.parquet"],
                   cfg=cfg, params={"ladder": [n for n, _ in LADDER]},
                   stats=stats).write(path)
    return stats


def build_all(cfg: dict | None = None) -> list[dict]:
    cfg = cfg or load_config()
    out = []
    for k in cfg["regions"]["cpm_k"]:
        for h in cfg["evaluation"]["horizons"]:
            s = run(cfg, k, h)
            out.append(s)
            if "skipped" in s:
                print(f"  k={k} h={h}: skipped ({s['skipped']})", flush=True)
                continue
            print(f"\n  k={k} h={h}: {s['tune_rows']} tune / {s['test_rows']} test rows, "
                  f"{s['zero_target_share']:.0%} zero targets", flush=True)
            for r in s["ladder"]:
                print(f"      {r['model']:24} test gain {r['test_gain_total']:>+10.1f} nats "
                      f"({r['test_gain_per_row']:>+7.4f}/row)  years won "
                      f"{r['years_won']}/{r['years_total']}", flush=True)
    return out


if __name__ == "__main__":
    import json
    r = build_all()
    print("\n" + json.dumps([{k: v for k, v in s.items() if k != "ladder"} for s in r],
                            indent=2, default=str))
