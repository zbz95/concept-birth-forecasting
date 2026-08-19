"""Layer 2 evaluation — the mechanism ladder, CIs, and stability.

Now the primary task. Three things the headline table cannot answer on its own:

  1. **Which mechanism carries the signal?** The plan's whole argument is an
     ablation ladder. Its localization form: does a node's own count series
     already explain where births land, or do you need topology, people, or
     semantics? Same four rungs, at node level.

  2. **Is the lift real or a few lucky regions?** The plan requires a
     block-bootstrap CI resampling whole region-lineages and whole years,
     because rows inside a region-year are not independent -- one large region
     contributing 800 births would otherwise dominate a naive interval.

  3. **Does it hold across time?** Per-origin recall and the plan's regime splits
     (pre/post-2017, pre/post-2020).

Baselines stay deterministic top-N rather than the plan's weighted-random, which
makes them harder to beat than specified. Degree is the one that matters: it is
the trivial structural signal a null should own.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.manifest import Manifest, load_config  # noqa: E402
from src.models.localization import (BASE_FEATS, node_features,  # noqa: E402
                                     truth_sets)
from src.regions.features import _region_members  # noqa: E402

MDIR = Path("data/models")

# The ablation ladder, in node-feature terms. Borderline goes to the null.
RUNGS = [
    ("own_series", ["n_T", "velocity_1y", "velocity_2y", "coinage_age", "recent_birth"]),
    ("+topology", ["degree_in_region", "degree_global", "rel_size"]),
    ("+people", ["pace_in_region", "author_influx_node"]),
    ("+semantics", ["emb_dist_to_centroid"]),
]


def add_node_author_influx(con, cfg: dict, T: int, f: pd.DataFrame) -> pd.DataFrame:
    """Share of a node's authors at T who are new to it since T-2."""
    rows = con.execute(f"""
        SELECT c.concept, p.v1_year, a.author FROM (
          SELECT CASE WHEN m.effective_date <= make_date(e.year,12,31)
                      THEN m.cluster ELSE e.term END AS concept, e.paper_id, e.year
          FROM read_parquet('data/graphs/event_store.parquet') e
          JOIN read_parquet('data/interim/merge_map.parquet') m ON m.cand = e.term
          WHERE e.year BETWEEN {T-2} AND {T}) c
        JOIN read_parquet('data/interim/papers.parquet') p ON p.id = c.paper_id
        JOIN (SELECT id, unnest(authors_parsed) AS author
              FROM read_parquet('data/interim/papers.parquet')) a ON a.id = c.paper_id
        """).fetchall()
    cur: dict[str, set] = defaultdict(set)
    past: dict[str, set] = defaultdict(set)
    for concept, y, author in rows:
        (cur if y == T else past)[concept].add(author)
    f["author_influx_node"] = [
        (len(cur[c] - past[c]) / len(cur[c])) if cur.get(c) else 0.0 for c in f.node]
    return f


def average_precision(order: list, truth: set) -> float:
    hits = 0
    tot = 0.0
    for i, v in enumerate(order, 1):
        if v in truth:
            hits += 1
            tot += hits / i
    return tot / len(truth) if truth else np.nan


def build(cfg: dict, k: int, horizon: int) -> pd.DataFrame:
    lo = cfg["evaluation"]["origins_tune"][0] + 2
    complete = cfg["registry"]["complete_through"]
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{cfg['runtime']['mem_budget_gb'] - 4}GB'")
    frames, truths = {}, {}
    for T in range(lo, complete - horizon + 1):
        f = node_features(con, cfg, T, k)
        if f.empty:
            continue
        f = add_node_author_influx(con, cfg, T, f)
        t, b = truth_sets(cfg, T, k, horizon)
        f["is_parent"] = [int(t.get(r, {}).get(n, 0) > 0)
                          for r, n in zip(f.region_id, f.node)]
        frames[T], truths[T] = f, b
    con.close()
    allf = pd.concat(frames.values(), ignore_index=True)
    g = allf.groupby(["origin", "region_id"])
    for c in BASE_FEATS + ["author_influx_node"]:
        mu = g[c].transform("mean")
        sd = g[c].transform("std").replace(0, np.nan)
        allf[f"{c}__z"] = ((allf[c] - mu) / sd).fillna(0.0)
    return allf, truths


def score_ladder(cfg: dict, allf: pd.DataFrame, truths: dict, k: int,
                 horizon: int) -> pd.DataFrame:
    tune_lo, tune_hi = cfg["evaluation"]["origins_tune"]
    tune = ((allf.origin >= tune_lo) & (allf.origin <= tune_hi)).values
    Ks = cfg["localization"]["K"]
    rng = np.random.default_rng(cfg["runtime"]["random_seed"])

    feats: list[str] = []
    for name, add in RUNGS:
        feats = feats + add
        cols = feats + [f"{c}__z" for c in feats]
        X = allf[cols].astype(float)
        X = X.fillna(X[tune].median())
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.03, max_depth=2, min_samples_leaf=100,
            l2_regularization=10.0, early_stopping=True, validation_fraction=0.25,
            n_iter_no_change=15, random_state=cfg["runtime"]["random_seed"])
        clf.fit(X[tune], allf.is_parent.values[tune])
        allf[f"s_{name}"] = clf.predict_proba(X)[:, 1]

    allf["s_random"] = rng.random(len(allf))
    allf["s_degree"] = allf.degree_in_region.values.astype(float)
    allf["s_velocity"] = allf.velocity_1y.values.astype(float)
    scorers = [f"s_{n}" for n, _ in RUNGS] + ["s_random", "s_degree", "s_velocity"]

    out = []
    for T, sub in allf.groupby("origin"):
        split = "tune" if tune_lo <= T <= tune_hi else "test"
        for rid, grp in sub.groupby("region_id"):
            b = truths[T].get(rid)
            if not b:
                continue
            for sc in scorers:
                s = dict(zip(grp.node, grp[sc]))
                order = sorted(s, key=lambda v: -s[v])
                rank = {v: i + 1 for i, v in enumerate(order)}
                for K in Ks:
                    if len(grp) <= K:
                        continue
                    topk = set(order[:K])
                    rec, hit, ap, mr = [], 0, [], []
                    for concept, truth in b:
                        inter = truth & topk
                        rec.append(len(inter) / len(truth))
                        hit += int(bool(inter))
                        ap.append(average_precision(order, truth))
                        mr.append(float(np.mean([rank[v] for v in truth if v in rank])))
                    out.append({
                        "origin": T, "k": k, "horizon": horizon, "split": split,
                        "region_id": rid, "region_size": len(grp), "K": K,
                        "scorer": sc, "n_births": len(b),
                        "recall_at_k": float(np.mean(rec)),
                        "hit_rate": hit / len(b),
                        "avg_precision": float(np.nanmean(ap)),
                        "mean_true_rank": float(np.mean(mr)),
                        "norm_rank": float(np.mean(mr)) / len(grp),
                    })
    return pd.DataFrame(out)


def bootstrap_ci(res: pd.DataFrame, scorer: str, ref: str, K: int,
                 n: int, seed: int) -> tuple:
    """Block bootstrap: resample whole region-lineages and whole years.

    Rows inside a region-year are not independent; one 900-member region
    contributing thousands of births would otherwise drive a naive interval.
    """
    d = res[(res.split == "test") & (res.K == K)]
    a = d[d.scorer == scorer].set_index(["origin", "region_id"])
    b = d[d.scorer == ref].set_index(["origin", "region_id"])
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    # Lineage proxy: the region id's numeric suffix is not stable across origins,
    # so blocks are (year, region) pairs grouped by year -- resampling years
    # carries every region within them together.
    years = sorted(d.origin.unique())
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        yy = rng.choice(years, size=len(years), replace=True)
        sa = np.concatenate([a.xs(y, level=0).recall_at_k.values for y in yy])
        wa = np.concatenate([a.xs(y, level=0).n_births.values for y in yy])
        sb = np.concatenate([b.xs(y, level=0).recall_at_k.values for y in yy])
        wb = np.concatenate([b.xs(y, level=0).n_births.values for y in yy])
        stats.append(np.average(sa, weights=wa) / max(1e-9, np.average(sb, weights=wb)))
    s = np.sort(stats)
    return float(np.mean(s)), float(s[int(0.025 * n)]), float(s[int(0.975 * n)])


def run(cfg: dict, k: int, horizon: int) -> dict:
    allf, truths = build(cfg, k, horizon)
    res = score_ladder(cfg, allf, truths, k, horizon)
    MDIR.mkdir(parents=True, exist_ok=True)
    path = MDIR / f"layer2_ladder_k{k}_h{horizon}.parquet"
    res.to_parquet(path, compression="zstd", index=False)

    K = cfg["localization"]["primary_K"]
    t = res[(res.split == "test") & (res.K == K)]
    summ = []
    for sc, g in t.groupby("scorer"):
        w = g.n_births.values
        summ.append({"scorer": sc,
                     "recall_at_k": float(np.average(g.recall_at_k, weights=w)),
                     "hit_rate": float(np.average(g.hit_rate, weights=w)),
                     "avg_precision": float(np.average(g.avg_precision, weights=w)),
                     "norm_rank": float(np.average(g.norm_rank, weights=w)),
                     "births": int(w.sum())})
    ci = {}
    n_boot = cfg["localization"]["bootstrap_n"]
    for sc in [f"s_{n}" for n, _ in RUNGS] + ["s_degree"]:
        ci[sc] = bootstrap_ci(res, sc, "s_random", K, n_boot,
                              cfg["runtime"]["random_seed"])
    stats = {"k": k, "horizon": horizon, "K": K, "summary": summ,
             "lift_vs_random_ci": {s: {"mean": round(v[0], 3), "lo": round(v[1], 3),
                                       "hi": round(v[2], 3)} for s, v in ci.items()}}
    Manifest.build(str(path), phase="layer2",
                   inputs=["data/registry/attachment/attachments.parquet"],
                   cfg=cfg, params={"rungs": [n for n, _ in RUNGS],
                                    "bootstrap_n": n_boot}, stats=stats).write(path)
    return stats


if __name__ == "__main__":
    import json
    cfg = load_config()
    k = cfg["localization"]["primary_k"]
    out = []
    for h in cfg["evaluation"]["horizons"]:
        s = run(cfg, k, h)
        out.append(s)
        print(f"\n=== k={k} h={h}, K={s['K']} (test) ===")
        for r in sorted(s["summary"], key=lambda x: -x["recall_at_k"]):
            print(f"  {r['scorer']:16} recall {r['recall_at_k']:.3f}  hit {r['hit_rate']:.3f}  "
                  f"AP {r['avg_precision']:.3f}  norm-rank {r['norm_rank']:.3f}")
        print("  lift vs random, block-bootstrap 95% CI:")
        for sc, c in s["lift_vs_random_ci"].items():
            print(f"    {sc:16} {c['mean']:.2f}x  [{c['lo']:.2f}, {c['hi']:.2f}]")
