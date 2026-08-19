"""Phase 7 — attachment of births to units, and the fractional target table.

A birth crystallizing in (T, T+h] is attached to the regions at origin T whose
member sets its profile overlaps. Attachment is what turns "a concept was born"
into "a forecastable quantity attached to a unit that existed beforehand".

Two arms, per the plan:

  1. overlap >= attach_min_overlap
  2. hypergeometric tail P(X >= o | |P|, |R|, |vocab_T|) <= p_attach

The second arm is size-aware and replaces the arithmetically inert Jaccard arm:
a 3-concept overlap with a 12-member region is extraordinary, while the same
overlap with a 900-member region is close to expected. A raw ratio cannot tell
those apart at these size scales.

**Multi-attachment is allowed and is the point.** A birth's 1.0 of target mass
splits equally across the units it attaches to, which is what makes intersection
births expressible. That gives the invariant asserted here:

    total target mass at an origin == number of attached births

Profiles are outcome data. Using them for attachment and evaluation is explicitly
permitted (leakage-checklist item 9); using them as model *features* would not
be, and they never enter the feature tables.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import hypergeom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.manifest import Manifest, load_config, log_event  # noqa: E402
from src.regions.features import _region_members  # noqa: E402

ODIR = Path("data/registry/attachment")


def attach_origin(cfg: dict, births: dict, profiles: dict, T: int, k: int,
                  horizon: int) -> tuple[list[dict], dict]:
    ac = cfg["attachment"]
    members = _region_members(T, k)
    if not members:
        return [], {}
    vocab_n = len(pq.read_table(f"data/interim/vocab/vocab_{T}.parquet"))

    node_regions: dict[str, list[str]] = defaultdict(list)
    for rid, mem in members.items():
        for c in mem:
            node_regions[c].append(rid)
    rsize = {rid: len(m) for rid, m in members.items()}

    # Births crystallizing strictly after T, within the horizon, and gradeable.
    lo, hi = T + 1, T + horizon
    cand = [(c, cy) for c, cy in births.items() if lo <= cy <= hi]

    rows, n_attached, n_orphan, dual = [], 0, 0, 0
    overlaps_all, surprises_all = [], []
    for concept, cy in cand:
        prof = profiles.get(concept)
        if not prof:
            n_orphan += 1
            continue
        P = [c for c in prof if c in node_regions]
        counts: dict[str, int] = defaultdict(int)
        for c in P:
            for rid in node_regions[c]:
                counts[rid] += 1
        nP = len(prof)
        hits = []
        for rid, o in counts.items():
            # P(X >= o) under the null that the profile is a random draw from
            # vocab_T, given the region's size.
            surprise = float(hypergeom.sf(o - 1, vocab_n, rsize[rid], nP))
            overlaps_all.append(o)
            surprises_all.append(surprise)
            if o >= ac["attach_min_overlap"] or surprise <= ac["p_attach"]:
                hits.append((rid, o, surprise))
        if not hits:
            n_orphan += 1
            rows.append({"origin": T, "k": k, "horizon": horizon, "concept": concept,
                         "crystallization_year": cy, "unit_id": None, "overlap": 0,
                         "surprise": None, "mass": 0.0, "n_attachments": 0,
                         "orphan": True})
            continue
        n_attached += 1
        if len(hits) >= 2:
            dual += 1
        share = 1.0 / len(hits)
        for rid, o, s in hits:
            rows.append({"origin": T, "k": k, "horizon": horizon, "concept": concept,
                         "crystallization_year": cy, "unit_id": rid, "overlap": o,
                         "surprise": s, "mass": share, "n_attachments": len(hits),
                         "orphan": False})

    mass = sum(r["mass"] for r in rows)
    assert n_attached == 0 or abs(mass - n_attached) / n_attached < 1e-9, (
        f"target mass {mass} != {n_attached} attached births at T={T} k={k} h={horizon}")

    stats = {
        "origin": T, "k": k, "horizon": horizon,
        "births_in_window": len(cand), "attached": n_attached, "orphans": n_orphan,
        "orphan_rate": round(n_orphan / max(1, len(cand)), 4),
        "dual_attached": dual,
        "dual_rate": round(dual / max(1, n_attached), 4),
        "total_target_mass": round(mass, 6),
        "median_overlap": float(np.median(overlaps_all)) if overlaps_all else None,
    }
    return rows, stats


def build_all(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    con = duckdb.connect()
    births = dict(con.execute("""
        SELECT concept, crystallization_year FROM read_parquet('data/registry/births.parquet')
        WHERE crystallization_year IS NOT NULL AND NOT censored""").fetchall())
    con.close()
    profiles = {r["concept"]: r["profile"]
                for r in pq.read_table("data/graphs/features/birth_profiles.parquet").to_pylist()}

    lo = cfg["evaluation"]["origins_tune"][0] + 2
    complete = cfg["registry"]["complete_through"]
    ODIR.mkdir(parents=True, exist_ok=True)
    allrows, stats = [], []
    for k in cfg["regions"]["cpm_k"]:
        for horizon in cfg["evaluation"]["horizons"]:
            for T in range(lo, complete - horizon + 1):
                r, s = attach_origin(cfg, births, profiles, T, k, horizon)
                if not s:
                    continue
                allrows += r
                stats.append(s)
                if k == 4:
                    print(f"  T={T} k={k} h={horizon}: {s['births_in_window']:>5,} births  "
                          f"attached {s['attached']:>5,}  orphan {s['orphan_rate']:.0%}  "
                          f"dual {s['dual_attached']:>4,} ({s['dual_rate']:.0%})", flush=True)
    out = ODIR / "attachments.parquet"
    pq.write_table(pa.Table.from_pylist(allrows), out, compression="zstd")

    # C2 gate: dual-attachment power on the TUNE origins only.
    tune_lo, tune_hi = cfg["evaluation"]["origins_tune"]
    tune = [s for s in stats if s["k"] == 4 and s["horizon"] == 1
            and tune_lo <= s["origin"] <= tune_hi]
    per_year = [s["dual_attached"] for s in tune]
    mean_dual = float(np.mean(per_year)) if per_year else 0.0
    floor = cfg["attachment"]["dual_attachment_floor"]
    c2 = "PASS" if mean_dual >= floor else "FAIL"
    if c2 == "FAIL":
        log_event("logs/flags.jsonl", {
            "phase": "7", "kind": "c2_gate_failure",
            "mean_dual_attached_per_tune_origin": round(mean_dual, 2),
            "floor": floor, "per_origin": per_year,
            "action": "STOP - PI decides whether pair units get the pre-declared relaxed rule "
                      "BEFORE the bridge hypothesis is registered"})

    agg = {"rows": len(allrows), "per_origin": stats,
           "c2_gate": {"verdict": c2, "mean_dual_per_tune_origin": round(mean_dual, 2),
                       "floor": floor, "tune_origins": [tune_lo, tune_hi],
                       "per_origin_dual": per_year}}
    Manifest.build(str(out), phase="7",
                   inputs=["data/registry/births.parquet",
                           "data/graphs/features/birth_profiles.parquet"],
                   cfg=cfg, params={k2: cfg["attachment"][k2] for k2 in
                                    ("attach_min_overlap", "p_attach", "multi_attachment",
                                     "profile_n_papers", "attach_top_k")},
                   stats=agg).write(out)
    return agg


if __name__ == "__main__":
    import json
    r = build_all()
    print(json.dumps(r["c2_gate"], indent=2))
