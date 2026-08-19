"""Layer 2 under a proper three-way split.

    train      2016-2018   fit coefficients
    validation 2019-2021   every model decision is made here
    test       2022-2023   scored once, reported, never iterated on

The point of the validation set is that model selection stops being guesswork
against test. Hyperparameters and the ablation rung are chosen by validation
recall@K; the test origins see exactly one scoring pass with the selected
configuration.

Bootstrap note: with two test origins a year-block bootstrap has two blocks and
is degenerate, so the interval here resamples REGIONS (clustered, births-weighted)
and the year-level dependence is not captured. That is a weaker interval than the
plan specifies and is labelled as such.
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.manifest import Manifest, load_config  # noqa: E402
from src.models.layer2_eval import RUNGS, average_precision, build  # noqa: E402
from src.models.localization import BASE_FEATS  # noqa: E402

MDIR = Path("data/models")

GRID = list(product([2, 3], [50, 100], [1.0, 10.0]))     # depth, leaf, l2


def _fit(allf, cols, mask, depth, leaf, l2, seed):
    X = allf[cols].astype(float)
    X = X.fillna(X[mask].median())
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.03, max_depth=depth, min_samples_leaf=leaf,
        l2_regularization=l2, early_stopping=True, validation_fraction=0.25,
        n_iter_no_change=15, random_state=seed)
    clf.fit(X[mask], allf.is_parent.values[mask])
    return clf.predict_proba(X)[:, 1]


def _score(allf, truths, col, K, origins):
    """births-weighted recall@K, hit rate and AP over the given origins."""
    rec, hit, ap, w = [], [], [], []
    for T in origins:
        sub = allf[allf.origin == T]
        for rid, grp in sub.groupby("region_id"):
            b = truths[T].get(rid)
            if not b or len(grp) <= K:
                continue
            s = dict(zip(grp.node, grp[col]))
            order = sorted(s, key=lambda v: -s[v])
            topk = set(order[:K])
            r = [len(t & topk) / len(t) for _, t in b]
            h = [float(bool(t & topk)) for _, t in b]
            a = [average_precision(order, t) for _, t in b]
            rec.append(np.mean(r)); hit.append(np.mean(h)); ap.append(np.nanmean(a))
            w.append(len(b))
    if not w:
        return None
    w = np.array(w, dtype=float)
    return {"recall": float(np.average(rec, weights=w)),
            "hit_rate": float(np.average(hit, weights=w)),
            "avg_precision": float(np.average(ap, weights=w)),
            "regions": len(w), "births": int(w.sum()),
            "per_region_recall": np.array(rec), "weights": w}


def run(cfg: dict, k: int, horizon: int) -> dict:
    lc = cfg["localization"]
    tr_lo, tr_hi = lc["origins_train"]
    va_lo, va_hi = lc["origins_validation"]
    te = list(range(lc["origins_test"][0], lc["origins_test"][1] + 1))
    va = list(range(va_lo, va_hi + 1))
    seed = cfg["runtime"]["random_seed"]

    allf, truths = build(cfg, k, horizon)
    train = ((allf.origin >= tr_lo) & (allf.origin <= tr_hi)).values

    # ---- model selection: validation only -------------------------------
    sel = []
    feats: list[str] = []
    rung_cols = {}
    for name, add in RUNGS:
        feats = feats + add
        rung_cols[name] = feats + [f"{c}__z" for c in feats]
    for rung, cols in rung_cols.items():
        for depth, leaf, l2 in GRID:
            col = f"__tmp"
            allf[col] = _fit(allf, cols, train, depth, leaf, l2, seed)
            for K in lc["K"]:
                s = _score(allf, truths, col, K, va)
                if s:
                    sel.append({"rung": rung, "depth": depth, "leaf": leaf, "l2": l2,
                                "K": K, "val_recall": s["recall"],
                                "val_hit": s["hit_rate"]})
    seldf = pd.DataFrame(sel)
    K_fixed = lc["primary_K"]
    best = seldf[seldf.K == K_fixed].sort_values("val_recall", ascending=False).iloc[0]

    # ---- one scoring pass on test with the selected configuration --------
    cols = rung_cols[best.rung]
    allf["s_sel"] = _fit(allf, cols, train, int(best.depth), int(best.leaf),
                         float(best.l2), seed)
    rng = np.random.default_rng(seed)
    allf["s_random"] = rng.random(len(allf))
    allf["s_degree"] = allf.degree_in_region.values.astype(float)

    out = {}
    for split, origins in (("validation", va), ("test", te)):
        out[split] = {c: _score(allf, truths, c, K_fixed, origins)
                      for c in ("s_sel", "s_degree", "s_random")}

    # ---- region-clustered bootstrap on test ------------------------------
    sel_t, ran_t = out["test"]["s_sel"], out["test"]["s_random"]
    n = lc["bootstrap_n"]
    lifts = []
    idx = np.arange(len(sel_t["weights"]))
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        a = np.average(sel_t["per_region_recall"][b], weights=sel_t["weights"][b])
        c = np.average(ran_t["per_region_recall"][b], weights=ran_t["weights"][b])
        lifts.append(a / max(1e-9, c))
    lifts = np.sort(lifts)

    for split in out:
        for c in out[split]:
            out[split][c] = {kk: vv for kk, vv in out[split][c].items()
                             if kk not in ("per_region_recall", "weights")}

    stats = {
        "k": k, "horizon": horizon, "K": K_fixed,
        "split": {"train": [tr_lo, tr_hi], "validation": va, "test": te},
        "selected": {"rung": best.rung, "depth": int(best.depth),
                     "leaf": int(best.leaf), "l2": float(best.l2),
                     "val_recall": float(best.val_recall)},
        "results": out,
        "test_lift_vs_random": {
            "mean": float(np.mean(lifts)),
            "ci95": [float(lifts[int(0.025 * n)]), float(lifts[int(0.975 * n)])],
            "bootstrap": "region-clustered; year-level dependence NOT captured "
                         "(only 2 test origins)"},
        "selection_grid_rows": len(seldf),
    }
    MDIR.mkdir(parents=True, exist_ok=True)
    path = MDIR / f"layer2_split_k{k}_h{horizon}.parquet"
    seldf.to_parquet(path, compression="zstd", index=False)
    Manifest.build(str(path), phase="layer2",
                   inputs=["data/registry/attachment/attachments.parquet"],
                   cfg=cfg, params={"grid": [list(g) for g in GRID],
                                    "rungs": [n for n, _ in RUNGS]},
                   stats=stats).write(path)
    return stats


if __name__ == "__main__":
    import json
    cfg = load_config()
    k = cfg["localization"]["primary_k"]
    for h in cfg["evaluation"]["horizons"]:
        s = run(cfg, k, h)
        print(f"\n=== k={k} h={h}, K={s['K']} ===")
        print(f"  split: train {s['split']['train']}  validation {s['split']['validation']}"
              f"  test {s['split']['test']}")
        print(f"  selected on VALIDATION: rung={s['selected']['rung']} depth={s['selected']['depth']} "
              f"leaf={s['selected']['leaf']} l2={s['selected']['l2']} "
              f"(val recall {s['selected']['val_recall']:.3f}, {s['selection_grid_rows']} configs tried)")
        for split in ("validation", "test"):
            r = s["results"][split]
            print(f"  {split}:")
            for c in ("s_random", "s_degree", "s_sel"):
                v = r[c]
                if v:
                    print(f"    {c:10} recall {v['recall']:.3f}  hit {v['hit_rate']:.3f}  "
                          f"AP {v['avg_precision']:.3f}  ({v['births']:,} births)")
        L = s["test_lift_vs_random"]
        print(f"  TEST lift vs random: {L['mean']:.2f}x  95% CI [{L['ci95'][0]:.2f}, {L['ci95'][1]:.2f}]")
