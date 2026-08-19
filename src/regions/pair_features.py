"""Phase 6 — pair units: region-pair bridge features.

A pair unit is an ordered-independent pair of regions with nonzero bridge mass at
T. Its target (Phase 7) is the mass of births multi-attached to both, which is
what makes intersection births expressible as a forecastable quantity rather than
an anecdote.

Features are spec item 7:

    bridge(A,B,T)   binarized edges with one endpoint in M(A) and the other in M(B)
    bridge_growth   bridge(A,B,T) / max(1, bridge(A,B,T-1)), membership fixed at T
    dual_citizens   |M(A) ∩ M(B)| — CPM regions overlap, so this is not always 0
    dual_papers     papers in T-2..T naming >=1 member of A AND >=1 member of B
    centroid_dist   cosine distance between the regions' mean embedding vectors

**Only bridge evolution up to T is admissible.** A bridge's later fame is
inadmissible under leakage-checklist item 6, which is why the growth term is a
ratio of two <=T quantities and never touches the target window.

Both bridge terms hold membership fixed at M(·) as of T, so the ratio measures
bridge growth rather than membership churn — the same rule as the region-level
edge velocity.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.manifest import Manifest, load_config  # noqa: E402
from src.regions.embedder import load as load_emb  # noqa: E402
from src.regions.features import _edges_at, _paper_concepts, _region_members  # noqa: E402

FDIR = Path("data/graphs/features")


def build_origin(con, cfg: dict, T: int, k: int) -> list[dict]:
    members = _region_members(T, k)
    if len(members) < 2:
        return []
    node_regions: dict[str, list[str]] = defaultdict(list)
    for rid, mem in members.items():
        for c in mem:
            node_regions[c].append(rid)

    def bridge_counts(edges) -> dict[tuple, int]:
        """Edges crossing between two regions, over fixed membership M(.)@T."""
        out: dict[tuple, int] = defaultdict(int)
        for u, v in edges:
            ru, rv = node_regions.get(u), node_regions.get(v)
            if not ru or not rv:
                continue
            for a in ru:
                for b in rv:
                    if a != b:
                        out[(a, b) if a < b else (b, a)] += 1
        return out

    br_T = bridge_counts(_edges_at(con, T, cfg))
    prev_path = Path(f"data/graphs/graph_{T-1}_edges.parquet")
    br_prev = bridge_counts(_edges_at(con, T - 1, cfg)) if prev_path.exists() else None

    # dual_papers: a paper touching r regions contributes to all C(r,2) pairs.
    pc = _paper_concepts(con, T, T - 2, T)
    dual_papers: dict[tuple, int] = defaultdict(int)
    for pid, cs in pc.items():
        touched = set()
        for c in cs:
            touched.update(node_regions.get(c, ()))
        if len(touched) < 2:
            continue
        tl = sorted(touched)
        for i in range(len(tl)):
            for j in range(i + 1, len(tl)):
                dual_papers[(tl[i], tl[j])] += 1

    emb = load_emb(T) if Path(f"data/graphs/embeddings/emb_{T}.npz").exists() else {}
    cent: dict[str, np.ndarray] = {}
    for rid, mem in members.items():
        vs = [emb[c] for c in mem if c in emb]
        if len(vs) >= 3:
            v = np.vstack(vs).mean(axis=0)
            cent[rid] = v / max(1e-9, np.linalg.norm(v))

    rows = []
    for (a, b), n in br_T.items():
        prev = br_prev.get((a, b)) if br_prev is not None else None
        growth = n / max(1, prev) if prev is not None else None
        ca, cb = cent.get(a), cent.get(b)
        dist = float(1 - float(ca @ cb)) if ca is not None and cb is not None else None
        rows.append({
            "origin": T, "k": k, "unit_type": "pair", "unit_id": f"{a}|{b}",
            "region_a": a, "region_b": b,
            "size_a": len(members[a]), "size_b": len(members[b]),
            "bridge_mass": n,
            "bridge_mass_prev": prev,
            "bridge_growth": growth,
            "bridge_growth_missing": growth is None,
            "dual_citizens": len(set(members[a]) & set(members[b])),
            "dual_papers": dual_papers.get((a, b), 0),
            "centroid_distance": dist,
            "centroid_distance_missing": dist is None,
        })
    return rows


def build_all(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    assert cfg["regions"]["feature_spec_signed_off"], "Phase 6 feature gate not passed"
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{cfg['runtime']['mem_budget_gb'] - 4}GB'")
    lo = cfg["evaluation"]["origins_tune"][0] + 2
    hi = cfg["evaluation"]["origins_test"][1]
    FDIR.mkdir(parents=True, exist_ok=True)
    allrows, stats = [], []
    for k in cfg["regions"]["cpm_k"]:
        for T in range(lo, hi + 1):
            r = build_origin(con, cfg, T, k)
            allrows += r
            nreg = len(_region_members(T, k))
            possible = nreg * (nreg - 1) // 2
            stats.append({"origin": T, "k": k, "regions": nreg,
                          "possible_pairs": possible, "pairs_with_bridge": len(r),
                          "share": round(len(r) / possible, 4) if possible else 0.0})
            print(f"  T={T} k={k}: {len(r):>6,} pair units of {possible:,} possible "
                  f"({len(r)/max(1,possible):.0%})", flush=True)
    con.close()
    out = FDIR / "pair_features.parquet"
    pq.write_table(pa.Table.from_pylist(allrows), out, compression="zstd")
    agg = {"rows": len(allrows), "origins": [lo, hi], "per_origin": stats}
    Manifest.build(str(out), phase="6",
                   inputs=["data/graphs/event_store.parquet"], cfg=cfg,
                   params={"spec": "reports/phase6_feature_spec.md item 7"},
                   stats=agg).write(out)
    return agg


if __name__ == "__main__":
    import json
    print(json.dumps(build_all(), indent=2, default=str))
