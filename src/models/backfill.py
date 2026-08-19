"""Phase 7/8 bridge — feature 8, confirmed-birth persistence.

Spec item 8: the number of registry births attached to a region's lineage whose
crystallization falls in the two years ending at T−(m−1).

The upper bound is `T-(m-1)`, **not** T. A birth crystallizing in year t is not
confirmed until data through t+m-1 exists, so at origin T only births with
t <= T-(m-1) are knowable. With m=2 that is t <= T-1, and the window is
[T-2, T-1]. Using t <= T would import a year of future evidence into every row —
leakage-checklist item 4.

Past births are attached to `regions_T` with the same two-arm rule Phase 7 uses.
Using a birth's profile for this is explicitly permitted: leakage-checklist item 9
allows profiles for "attachment/evaluation and confirmed-birth back-fill", and
nowhere else. They never enter as model features in their own right.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import hypergeom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.manifest import Manifest, load_config  # noqa: E402
from src.regions.features import _region_members  # noqa: E402

OUT = "data/graphs/features/birth_history.parquet"


def build_all(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    ac = cfg["attachment"]
    m = cfg["registry"]["m"]
    con = duckdb.connect()
    births = dict(con.execute("""
        SELECT concept, crystallization_year FROM read_parquet('data/registry/births.parquet')
        WHERE crystallization_year IS NOT NULL AND NOT censored""").fetchall())
    con.close()
    profiles = {r["concept"]: r["profile"]
                for r in pq.read_table("data/graphs/features/birth_profiles.parquet").to_pylist()}

    lo = cfg["evaluation"]["origins_tune"][0] + 2
    hi = cfg["evaluation"]["origins_test"][1]
    rows, stats = [], []
    for k in cfg["regions"]["cpm_k"]:
        for T in range(lo, hi + 1):
            members = _region_members(T, k)
            if not members:
                continue
            vocab_n = len(pq.read_table(f"data/interim/vocab/vocab_{T}.parquet"))
            node_regions: dict[str, list[str]] = defaultdict(list)
            for rid, mem in members.items():
                for c in mem:
                    node_regions[c].append(rid)
            rsize = {rid: len(mm) for rid, mm in members.items()}

            confirm_hi = T - (m - 1)
            confirm_lo = confirm_hi - 1
            past = [c for c, cy in births.items() if confirm_lo <= cy <= confirm_hi]

            acc: dict[str, float] = defaultdict(float)
            cnt: dict[str, int] = defaultdict(int)
            n_att = 0
            for concept in past:
                prof = profiles.get(concept)
                if not prof:
                    continue
                counts: dict[str, int] = defaultdict(int)
                for c in prof:
                    for rid in node_regions.get(c, ()):
                        counts[rid] += 1
                hits = [rid for rid, o in counts.items()
                        if o >= ac["attach_min_overlap"]
                        or float(hypergeom.sf(o - 1, vocab_n, rsize[rid], len(prof)))
                        <= ac["p_attach"]]
                if not hits:
                    continue
                n_att += 1
                share = 1.0 / len(hits)
                for rid in hits:
                    acc[rid] += share
                    cnt[rid] += 1
            for rid in members:
                rows.append({"origin": T, "k": k, "unit_id": rid,
                             "confirmed_births_mass": round(acc.get(rid, 0.0), 6),
                             "confirmed_births_count": cnt.get(rid, 0),
                             "window_lo": confirm_lo, "window_hi": confirm_hi})
            stats.append({"origin": T, "k": k, "past_births": len(past),
                          "attached": n_att, "regions": len(members)})
            if k == 4:
                nz = sum(1 for rid in members if acc.get(rid, 0) > 0)
                print(f"  T={T} k={k}: {len(past):>5,} confirmed births in "
                      f"[{confirm_lo},{confirm_hi}]  attached {n_att:>5,}  "
                      f"regions with history {nz}/{len(members)}", flush=True)

    pq.write_table(pa.Table.from_pylist(rows), OUT, compression="zstd")
    agg = {"rows": len(rows), "per_origin": stats,
           "window_rule": "crystallization in [T-1-(m-1), T-(m-1)]"}
    Manifest.build(OUT, phase="7", inputs=["data/registry/births.parquet",
                                          "data/graphs/features/birth_profiles.parquet"],
                   cfg=cfg, params={"m": m, "attach_min_overlap": ac["attach_min_overlap"],
                                    "p_attach": ac["p_attach"]}, stats=agg).write(OUT)
    return agg


if __name__ == "__main__":
    import json
    print(json.dumps({k: v for k, v in build_all().items() if k != "per_origin"}, indent=2))
