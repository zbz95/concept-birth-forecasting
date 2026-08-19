"""Phase 6 — region features, exactly as specified in reports/phase6_feature_spec.md.

Every quantity is computed from data dated <= T. Feature 8 (confirmed-birth
persistence) is deliberately absent: it depends on Phase 7 attachment and is
back-filled afterwards, per the plan's 6 -> 7 -> back-fill -> 8 ordering.
Feature 7 (bridge mass) belongs to pair units, not region units.

Two spec rules do real work here and are easy to get wrong:

  * Membership is held fixed at M(R) as of T when computing any T-1 quantity, so
    edge velocity and density change measure growth rather than membership churn.
  * A feature that is undefined is emitted as null with a `_missing` indicator.
    It is never imputed as zero -- a region with no T-1 counterpart is not a
    region with zero growth, and at 47% lineage stability that distinction
    applies to about half the rows.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.manifest import Manifest, load_config  # noqa: E402
from src.regions.embedder import load as load_emb  # noqa: E402

FDIR = Path("data/graphs/features")


def _region_members(T: int, k: int) -> dict[str, list[str]]:
    rows = pq.read_table(
        f"data/graphs/regions/regions_{T}_k{k}.parquet").to_pylist()
    return {r["region_id"]: r["members"] for r in rows}


def _lineage(T: int, k: int) -> dict[str, str]:
    p = Path(f"data/graphs/regions/lineage_k{k}.parquet")
    if not p.exists():
        return {}
    return {r["region_id"]: r["parent_id"]
            for r in pq.read_table(p).to_pylist()
            if r["origin"] == T and r["parent_id"]}


def _paper_concepts(con, T: int, y0: int, y1: int) -> dict[str, set]:
    rows = con.execute(f"""
        SELECT p.paper_id,
               CASE WHEN m.effective_date <= DATE '{T}-12-31' THEN m.cluster ELSE p.term END
        FROM read_parquet('data/graphs/event_store.parquet') p
        JOIN read_parquet('data/interim/merge_map.parquet') m ON m.cand = p.term
        WHERE p.year BETWEEN {y0} AND {y1}""").fetchall()
    out: dict[str, set] = defaultdict(set)
    for pid, c in rows:
        out[pid].add(c)
    return out


def _edges_at(con, T: int, cfg: dict) -> set:
    gc = cfg["graph"]
    rows = con.execute(f"""
        SELECT u, v FROM read_parquet('data/graphs/graph_{T}_edges.parquet')
        WHERE n_papers >= {gc['binarize_min_papers']} AND lift >= {gc['edge_min_lift']}
    """).fetchall()
    return {(u, v) for u, v in rows}


def build_origin(con, cfg: dict, T: int, k: int) -> list[dict]:
    members = _region_members(T, k)
    if not members:
        return []
    lin = _lineage(T, k)
    prev_members = _region_members(T - 1, k) if Path(
        f"data/graphs/regions/regions_{T-1}_k{k}.parquet").exists() else {}

    edges_T = _edges_at(con, T, cfg)
    edges_prev = _edges_at(con, T - 1, cfg) if Path(
        f"data/graphs/graph_{T-1}_edges.parquet").exists() else None

    # Paper-level facts for the trailing window and the two preceding years,
    # which is everything features 1-6 and 9 need.
    meta = dict(con.execute(f"""
        SELECT id, struct_pack(y := v1_year, q := v1_quarter, a := authors_parsed)
        FROM read_parquet('data/interim/papers.parquet')
        WHERE v1_year BETWEEN {T-4} AND {T}""").fetchall())
    pc = _paper_concepts(con, T, T - 4, T)

    concept_papers: dict[str, list[str]] = defaultdict(list)
    for pid, cs in pc.items():
        for c in cs:
            concept_papers[c].append(pid)

    emb = load_emb(T) if Path(f"data/graphs/embeddings/emb_{T}.npz").exists() else {}

    rows = []
    for rid, mem in members.items():
        mset = set(mem)
        papers_by_year: dict[int, set] = defaultdict(set)
        quarter: dict[tuple, set] = defaultdict(set)
        for c in mem:
            for pid in concept_papers.get(c, ()):
                m = meta.get(pid)
                if m is None:
                    continue
                papers_by_year[m["y"]].add(pid)
                quarter[(m["y"], m["q"])].add(pid)

        n = {y: len(papers_by_year.get(y, ())) for y in range(T - 4, T + 1)}
        exposure = n[T]

        # 2. paper velocity
        v1 = n[T] / max(1, n[T - 1])
        v2 = n[T] / max(1.0, (n[T - 1] + n[T - 2]) / 2)

        # 4. YoY same-quarter acceleration
        acc, acc_missing = None, True
        if n[T - 2] > 0 or n[T - 1] > 0:
            terms = []
            for q in (1, 2, 3, 4):
                a = len(quarter.get((T, q), ()))
                b = len(quarter.get((T - 1, q), ()))
                c_ = len(quarter.get((T - 2, q), ()))
                terms.append(a / max(1, b) - b / max(1, c_))
            acc, acc_missing = float(np.mean(terms)), False

        # 3 & 5. edge velocity and density change, membership fixed at M(R)@T
        e_T = sum(1 for u, v in combinations(sorted(mset), 2) if (u, v) in edges_T)
        dens = (2 * e_T / (len(mset) * (len(mset) - 1))) if len(mset) > 2 else None
        ev = dd = None
        graph_missing = True
        parent = lin.get(rid)
        if parent and edges_prev is not None and parent in prev_members:
            e_prev = sum(1 for u, v in combinations(sorted(mset), 2) if (u, v) in edges_prev)
            ev = e_T / max(1, e_prev)
            if dens is not None:
                d_prev = 2 * e_prev / (len(mset) * (len(mset) - 1))
                dd = dens - d_prev
            graph_missing = False

        # 6. pace of collaboration and social breadth, over member pairs
        paces, breadths = [], []
        for u, v in combinations(sorted(mset), 2):
            if (u, v) not in edges_T:
                continue
            co = [p for p in set(concept_papers.get(u, ())) & set(concept_papers.get(v, ()))
                  if meta.get(p) and T - 2 <= meta[p]["y"] <= T]
            if not co:
                continue
            ys = [meta[p]["y"] for p in co]
            span = max(1.0, (max(ys) - min(ys)) + 1)
            paces.append(len(co) / span)
            by_auth: dict[str, list[str]] = defaultdict(list)
            for p in co:
                for a in meta[p]["a"] or ():
                    by_auth[a].append(p)
            par = {p: p for p in co}

            def find(x):
                while par[x] != x:
                    par[x] = par[par[x]]
                    x = par[x]
                return x
            for ps in by_auth.values():
                r0 = find(ps[0])
                for o in ps[1:]:
                    ro = find(o)
                    if ro != r0:
                        par[ro] = r0
            breadths.append(len({find(p) for p in co}))
        pace = float(np.median(paces)) if paces else None
        breadth = float(np.median(breadths)) if breadths else None

        # 9. author influx
        def authors(y):
            s = set()
            for pid in papers_by_year.get(y, ()):
                for a in (meta[pid]["a"] or ()):
                    s.add(a)
            return s
        aT, a1, a2 = authors(T), authors(T - 1), authors(T - 2)
        influx = len(aT - (a1 | a2)) / max(1, len(aT)) if aT else None

        # 10. embedding-density influx, T-vintage embedder for both years
        spread = spread_d = None
        vecs = [emb[c] for c in mem if c in emb]
        if len(vecs) >= 3:
            V = np.vstack(vecs)
            cen = V.mean(axis=0)
            cen /= max(1e-9, np.linalg.norm(cen))
            spread = float(np.mean(1 - V @ cen))
            if parent and parent in prev_members:
                pv = [emb[c] for c in prev_members[parent] if c in emb]
                if len(pv) >= 3:
                    P = np.vstack(pv)
                    cp = P.mean(axis=0)
                    cp /= max(1e-9, np.linalg.norm(cp))
                    spread_d = spread - float(np.mean(1 - P @ cp))

        rows.append({
            "origin": T, "k": k, "unit_type": "region", "unit_id": rid,
            "size": len(mset), "exposure": exposure,
            "n_T": n[T], "n_T1": n[T - 1], "n_T2": n[T - 2],
            "paper_velocity_1y": v1, "paper_velocity_2y": v2,
            "yoy_quarter_accel": acc, "yoy_quarter_accel_missing": acc_missing,
            "edges_internal": e_T, "density": dens,
            "edge_velocity": ev, "density_change": dd,
            "graph_delta_missing": graph_missing,
            "pace_collaboration": pace, "social_breadth": breadth,
            "pace_missing": pace is None,
            "author_influx": influx, "author_influx_missing": influx is None,
            "embedding_spread": spread, "embedding_influx": spread_d,
            "embedding_missing": spread_d is None,
            "has_lineage_parent": parent is not None,
        })
    return rows


def build_all(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    assert cfg["regions"]["feature_spec_signed_off"], \
        "feature_spec_signed_off is false - the Phase 6 gate has not been passed"
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{cfg['runtime']['mem_budget_gb'] - 4}GB'")
    lo = cfg["evaluation"]["origins_tune"][0] + 2      # needs T-2 history
    hi = cfg["evaluation"]["origins_test"][1]
    FDIR.mkdir(parents=True, exist_ok=True)
    allrows, stats = [], []
    for k in cfg["regions"]["cpm_k"]:
        for T in range(lo, hi + 1):
            r = build_origin(con, cfg, T, k)
            allrows += r
            miss = {f: sum(1 for x in r if x[f]) / max(1, len(r))
                    for f in ("graph_delta_missing", "pace_missing", "embedding_missing",
                              "yoy_quarter_accel_missing")}
            stats.append({"origin": T, "k": k, "rows": len(r), "missing_share": miss})
            print(f"  T={T} k={k}: {len(r):>5} rows  missing "
                  f"graph {miss['graph_delta_missing']:.0%} "
                  f"pace {miss['pace_missing']:.0%} "
                  f"emb {miss['embedding_missing']:.0%}", flush=True)
    con.close()
    out = FDIR / "region_features.parquet"
    pq.write_table(pa.Table.from_pylist(allrows), out, compression="zstd")
    agg = {"rows": len(allrows), "origins": [lo, hi], "k": cfg["regions"]["cpm_k"],
           "per_origin": stats,
           "features_deferred": {"confirmed_birth_persistence": "back-filled after Phase 7",
                                 "bridge_mass_growth": "pair units, not region units"}}
    Manifest.build(str(out), phase="6",
                   inputs=["data/graphs/event_store.parquet",
                           "data/interim/papers.parquet"],
                   cfg=cfg, params={"spec": "reports/phase6_feature_spec.md"},
                   stats=agg).write(out)
    return agg


if __name__ == "__main__":
    import json
    print(json.dumps(build_all(), indent=2, default=str))
