"""Layer 2 — parent localization: which nodes inside a region will a birth touch?

The task, in the PI's framing: a region has ~100 nodes; pick ~10 of them and
claim the next concept born here will be related to those. Then check.

This is the plan's Layer 2, and the plan already names it the headline
deliverable ("ranked node lists s(v) and parent-set suggestions at T; region/pair
rate tables as the statistical backbone beneath"). It is better posed than the
rate model in three ways:

  * Evaluation is per realized birth, so there are thousands of scored events
    rather than a few hundred unit-years.
  * It is a RANKING, so the calibration failure that sank the Phase 8 challengers
    cannot arise -- there is no rate to over- or under-state.
  * The baselines are unambiguous and specified: degree- and velocity-weighted
    random parent sets, per the plan. Lift is measured against those, never
    against chance-uniform alone, because node degree is exactly the trivial
    signal a null should already own.

Scoring:
  s_velocity  the plan's v1 -- node velocity, the trivial-but-real signal
  s_degree    node degree in graph_T, the "popular nodes get everything" null
  s_model     the plan's v2 -- a fitted node model over <=T features

Truth for a birth b attached to region R is `profile(b) ∩ M(R)`: the members of
R that the birth's own first papers actually sat among. Profiles are outcome
data, used here for evaluation only (leakage-checklist item 9).
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
from src.regions.embedder import load as load_emb  # noqa: E402
from src.regions.features import _edges_at, _paper_concepts, _region_members  # noqa: E402

MDIR = Path("data/models")

BASE_FEATS = ["n_T", "velocity_1y", "velocity_2y", "degree_in_region",
              "degree_global", "coinage_age", "recent_birth", "pace_in_region",
              "emb_dist_to_centroid", "rel_size"]
# The task is "which node inside THIS region", not "which node anywhere", so
# every feature also enters as a within-region z-score. Without these the model
# learns that big regions in later origins have more parents -- a global
# pattern that does not transfer, and the reason the first fit reached tune
# recall 0.380 against test 0.179.
REL_FEATS = [f"{c}__z" for c in BASE_FEATS]
NODE_FEATS = BASE_FEATS + REL_FEATS


def node_features(con, cfg: dict, T: int, k: int) -> pd.DataFrame:
    """One row per (region, member node), all features dated <= T."""
    members = _region_members(T, k)
    if not members:
        return pd.DataFrame()

    yr = dict(con.execute(f"""
        SELECT concept, sum(CASE WHEN year = {T} THEN n_papers ELSE 0 END) AS n_T
        FROM read_parquet('data/registry/concept_year.parquet')
        WHERE year <= {T} GROUP BY 1""").fetchall())
    yr1 = dict(con.execute(f"""
        SELECT concept, sum(CASE WHEN year = {T-1} THEN n_papers ELSE 0 END)
        FROM read_parquet('data/registry/concept_year.parquet')
        WHERE year <= {T} GROUP BY 1""").fetchall())
    yr2 = dict(con.execute(f"""
        SELECT concept, sum(CASE WHEN year = {T-2} THEN n_papers ELSE 0 END)
        FROM read_parquet('data/registry/concept_year.parquet')
        WHERE year <= {T} GROUP BY 1""").fetchall())
    coin = dict(con.execute("""
        SELECT concept, coinage_year FROM read_parquet('data/registry/births.parquet')
        """).fetchall())
    cry = dict(con.execute("""
        SELECT concept, crystallization_year FROM read_parquet('data/registry/births.parquet')
        WHERE crystallization_year IS NOT NULL""").fetchall())

    edges = _edges_at(con, T, cfg)
    deg: dict[str, int] = defaultdict(int)
    adj: dict[str, set] = defaultdict(set)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
        adj[u].add(v)
        adj[v].add(u)

    pcw = _paper_concepts(con, T, T - 2, T)
    cp: dict[str, set] = defaultdict(set)
    for pid, cs in pcw.items():
        for c in cs:
            cp[c].add(pid)

    emb = load_emb(T) if Path(f"data/graphs/embeddings/emb_{T}.npz").exists() else {}
    m_conf = cfg["registry"]["m"]

    rows = []
    for rid, mem in members.items():
        mset = set(mem)
        vs = [emb[c] for c in mem if c in emb]
        cen = None
        if len(vs) >= 3:
            cen = np.vstack(vs).mean(axis=0)
            cen /= max(1e-9, np.linalg.norm(cen))
        for v in mem:
            nT, n1, n2 = yr.get(v, 0), yr1.get(v, 0), yr2.get(v, 0)
            inside = len(adj[v] & mset)
            co = sum(len(cp[v] & cp[o]) for o in (adj[v] & mset))
            rows.append({
                "origin": T, "k": k, "region_id": rid, "node": v,
                "region_size": len(mem),
                "n_T": nT,
                "velocity_1y": nT / max(1, n1),
                "velocity_2y": nT / max(1.0, (n1 + n2) / 2),
                "degree_in_region": inside,
                "degree_global": deg[v],
                "rel_size": inside / max(1, len(mem) - 1),
                "coinage_age": T - coin.get(v, T),
                # A node that itself crystallized recently -- and confirmably so,
                # at t <= T-(m-1) -- is a different kind of neighbour.
                "recent_birth": int(cry.get(v, -1) in (T - m_conf, T - m_conf + 1)
                                    and cry.get(v, 9999) <= T - (m_conf - 1)),
                "pace_in_region": co / max(1, inside) / 3.0,
                "emb_dist_to_centroid": (float(1 - emb[v] @ cen)
                                         if cen is not None and v in emb else np.nan),
            })
    return pd.DataFrame(rows)


def truth_sets(cfg: dict, T: int, k: int, horizon: int) -> dict:
    """region_id -> {node: n_births whose profile touched it}."""
    con = duckdb.connect()
    att = con.execute(f"""
        SELECT concept, unit_id FROM read_parquet('data/registry/attachment/attachments.parquet')
        WHERE origin={T} AND k={k} AND horizon={horizon} AND NOT orphan""").fetchall()
    con.close()
    profiles = {r["concept"]: set(r["profile"])
                for r in pq.read_table("data/graphs/features/birth_profiles.parquet").to_pylist()}
    members = _region_members(T, k)
    out: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    births: dict[str, list] = defaultdict(list)
    for concept, rid in att:
        prof = profiles.get(concept)
        if not prof or rid not in members:
            continue
        hit = prof & set(members[rid])
        if not hit:
            continue
        births[rid].append((concept, hit))
        for h in hit:
            out[rid][h] += 1
    return out, births


def evaluate(scores: dict, births: list, K: int) -> dict:
    """Recall@K and mean rank of the true parents, for one region."""
    order = sorted(scores, key=lambda v: -scores[v])
    topk = set(order[:K])
    rank = {v: i + 1 for i, v in enumerate(order)}
    recalls, ranks, hits = [], [], 0
    for concept, truth in births:
        inter = truth & topk
        recalls.append(len(inter) / len(truth))
        hits += int(bool(inter))
        ranks.append(float(np.mean([rank[v] for v in truth if v in rank])))
    return {"recall_at_k": float(np.mean(recalls)) if recalls else np.nan,
            "hit_rate": hits / len(births) if births else np.nan,
            "mean_true_rank": float(np.mean(ranks)) if ranks else np.nan,
            "n_births": len(births)}


def run(cfg: dict, k: int, horizon: int, Ks=(5, 10, 20)) -> dict:
    lo = cfg["evaluation"]["origins_tune"][0] + 2
    complete = cfg["registry"]["complete_through"]
    tune_lo, tune_hi = cfg["evaluation"]["origins_tune"]
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{cfg['runtime']['mem_budget_gb'] - 4}GB'")

    frames, truths = {}, {}
    for T in range(lo, complete - horizon + 1):
        f = node_features(con, cfg, T, k)
        if f.empty:
            continue
        t, b = truth_sets(cfg, T, k, horizon)
        f["is_parent"] = [int(t.get(r, {}).get(n, 0) > 0)
                          for r, n in zip(f.region_id, f.node)]
        frames[T], truths[T] = f, b
    con.close()
    if not frames:
        return {"k": k, "horizon": horizon, "skipped": "no data"}

    allf = pd.concat(frames.values(), ignore_index=True)
    g = allf.groupby(["origin", "region_id"])
    for c in BASE_FEATS:
        mu = g[c].transform("mean")
        sd = g[c].transform("std").replace(0, np.nan)
        allf[f"{c}__z"] = ((allf[c] - mu) / sd).fillna(0.0)
    tune = ((allf.origin >= tune_lo) & (allf.origin <= tune_hi)).values
    X = allf[NODE_FEATS].astype(float)
    med = X[tune].median()
    X = X.fillna(med)
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.03, max_depth=2, min_samples_leaf=100,
        l2_regularization=10.0, early_stopping=True, validation_fraction=0.25,
        n_iter_no_change=15, random_state=cfg["runtime"]["random_seed"])
    clf.fit(X[tune], allf.is_parent.values[tune])
    allf["s_model"] = clf.predict_proba(X)[:, 1]

    rng = np.random.default_rng(cfg["runtime"]["random_seed"])
    results = []
    for T, f in frames.items():
        split = "tune" if tune_lo <= T <= tune_hi else "test"
        sub = allf[allf.origin == T]
        for rid, grp in sub.groupby("region_id"):
            b = truths[T].get(rid)
            if not b:
                continue
            sc = {
                "s_model": dict(zip(grp.node, grp.s_model)),
                "s_velocity": dict(zip(grp.node, grp.velocity_1y)),
                "s_degree": dict(zip(grp.node, grp.degree_in_region)),
                "s_random": dict(zip(grp.node, rng.random(len(grp)))),
            }
            for K in Ks:
                if len(grp) <= K:
                    continue          # a region smaller than K is not a prediction
                for name, s in sc.items():
                    r = evaluate(s, b, K)
                    r.update({"origin": T, "k": k, "horizon": horizon, "split": split,
                              "region_id": rid, "region_size": len(grp),
                              "K": K, "scorer": name})
                    results.append(r)

    res = pd.DataFrame(results)
    MDIR.mkdir(parents=True, exist_ok=True)
    path = MDIR / f"localization_k{k}_h{horizon}.parquet"
    res.to_parquet(path, compression="zstd", index=False)

    summ = []
    for (split, K, scorer), g in res.groupby(["split", "K", "scorer"]):
        w = g.n_births.values
        summ.append({"split": split, "K": int(K), "scorer": scorer,
                     "regions": len(g), "births": int(w.sum()),
                     "recall_at_k": float(np.average(g.recall_at_k, weights=w)),
                     "hit_rate": float(np.average(g.hit_rate, weights=w)),
                     "mean_true_rank": float(np.average(g.mean_true_rank, weights=w))})
    stats = {"k": k, "horizon": horizon, "rows": len(res), "summary": summ}
    Manifest.build(str(path), phase="8-layer2",
                   inputs=["data/registry/attachment/attachments.parquet",
                           "data/graphs/features/birth_profiles.parquet"],
                   cfg=cfg, params={"Ks": list(Ks), "features": NODE_FEATS},
                   stats=stats).write(path)
    return stats


if __name__ == "__main__":
    import json
    cfg = load_config()
    out = []
    for k in cfg["regions"]["cpm_k"]:
        for h in cfg["evaluation"]["horizons"]:
            s = run(cfg, k, h)
            out.append(s)
            if "skipped" in s:
                continue
            print(f"\n=== k={k} h={h} ===")
            for r in sorted(s["summary"], key=lambda x: (x["split"], x["K"], x["scorer"])):
                if r["split"] != "test":
                    continue
                print(f"  K={r['K']:>2} {r['scorer']:12} recall@K {r['recall_at_k']:.3f}  "
                      f"hit-rate {r['hit_rate']:.3f}  mean true rank {r['mean_true_rank']:.1f}  "
                      f"({r['births']:,} births, {r['regions']} regions)")
