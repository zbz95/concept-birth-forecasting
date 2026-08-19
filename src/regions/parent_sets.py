"""Phase 6, System B — parent-sets.

A parent-set is a small concept set that could plausibly be the parentage of a
future birth. The plan specifies two generators:

  A. **Frequent sub-profiles of past confirmed births.** Take the births already
     confirmed at T, look at the concepts their first papers actually sat among,
     and keep the small combinations that recur. This asks: which pairings have
     historically preceded a birth?

  B. **Dense high-pace triangles.** Triangles in graph_T whose three pairs are
     being re-used fastest. This asks: which pairings are currently hot,
     regardless of whether anything has been born from them?

Two causality rules do the work here:

  * Only births with crystallization <= T-(m-1) are admissible at origin T, since
    a birth in year t is not confirmed until data through t+m-1 exists. Using
    crystallization <= T would import a year of future evidence into every row.
  * A birth's profile is built from its first `profile_n_papers` papers. The
    birth is used at origin T only if ALL of those papers are dated <= T —
    otherwise the profile itself is partly made of the future. This is
    conservative: it drops slow-accumulating births rather than truncating them.

Profile ranking is by lift, never raw count, or every profile would be the same
list of hub concepts.
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
from src.regions.features import _edges_at, _paper_concepts  # noqa: E402

FDIR = Path("data/graphs/features")
PROFILES = FDIR / "birth_profiles.parquet"


def build_profiles(cfg: dict) -> dict:
    """Lift-ranked profile of every crystallized birth, with its completion date.

    Computed once. The per-origin admissibility test is a date comparison against
    `profile_max_date`, not a recomputation.
    """
    ps = cfg["parent_sets"]
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{cfg['runtime']['mem_budget_gb'] - 4}GB'")

    births = con.execute("""
        SELECT concept, crystallization_year FROM read_parquet('data/registry/births.parquet')
        WHERE crystallization_year IS NOT NULL""").fetchall()
    bset = {c for c, _ in births}

    # First N papers of each birth concept, chronologically.
    rows = con.execute(f"""
        SELECT concept, paper_id, v1_date FROM (
          SELECT CASE WHEN m.effective_date <= p.v1_date THEN m.cluster ELSE p.term END AS concept,
                 p.paper_id, p.v1_date,
                 row_number() OVER (PARTITION BY CASE WHEN m.effective_date <= p.v1_date
                                                 THEN m.cluster ELSE p.term END
                                    ORDER BY p.v1_date, p.paper_id) AS rn
          FROM read_parquet('data/graphs/event_store.parquet') p
          JOIN read_parquet('data/interim/merge_map.parquet') m ON m.cand = p.term)
        WHERE rn <= {ps['profile_n_papers']}""").fetchall()
    first_papers: dict[str, list] = defaultdict(list)
    for c, pid, d in rows:
        if c in bset:
            first_papers[c].append((pid, d))
    del rows

    pc = _paper_concepts(con, 2025, 1900, 2025)
    total_papers = len(pc)
    global_df: dict[str, int] = defaultdict(int)
    for cs in pc.values():
        for c in cs:
            global_df[c] += 1

    out = []
    for concept, papers in first_papers.items():
        pids = [p for p, _ in papers]
        maxd = max(d for _, d in papers)
        counts: dict[str, int] = defaultdict(int)
        for pid in pids:
            for c in pc.get(pid, ()):
                if c != concept:
                    counts[c] += 1
        n = len(pids)
        # lift = P(c | birth's first papers) / P(c | corpus)
        # Lift is the RANKING; profile_min_count is an eligibility floor. Without
        # it, every concept appearing once among 20 papers at the df=9 vocabulary
        # floor scores an identical 1499 and the top-k is an arbitrary draw from
        # the corpus's rarest terms. See configs/config.yaml for the measurement.
        scored = sorted(
            ((c, (v / n) / max(1e-9, global_df[c] / total_papers))
             for c, v in counts.items() if v >= ps["profile_min_count"]),
            key=lambda x: -x[1])[:ps["profile_top_k"]]
        out.append({"concept": concept, "n_papers": n,
                    "profile_max_date": maxd,
                    "profile": [c for c, _ in scored],
                    "profile_lift": [float(s) for _, s in scored]})
    con.close()
    FDIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(out), PROFILES, compression="zstd")
    stats = {"births_with_profile": len(out),
             "mean_profile_len": round(float(np.mean([len(o["profile"]) for o in out])), 2)}
    Manifest.build(str(PROFILES), phase="6",
                   inputs=["data/registry/births.parquet",
                           "data/graphs/event_store.parquet"],
                   cfg=cfg, params={k: ps[k] for k in ("profile_n_papers", "profile_top_k")},
                   stats=stats).write(PROFILES)
    return stats


def build_origin(con, cfg: dict, T: int) -> list[dict]:
    ps = cfg["parent_sets"]
    m = cfg["registry"]["m"]
    confirm_by = T - (m - 1)

    prof = pq.read_table(PROFILES).to_pylist()
    births = dict(con.execute("""
        SELECT concept, crystallization_year FROM read_parquet('data/registry/births.parquet')
        WHERE crystallization_year IS NOT NULL""").fetchall())

    # --- Generator A: frequent sub-profiles of confirmed births -------------
    support: dict[tuple, int] = defaultdict(int)
    n_admissible = 0
    for r in prof:
        cy = births.get(r["concept"])
        if cy is None or cy > confirm_by:
            continue
        if r["profile_max_date"].year > T:
            continue                       # profile not complete by T
        n_admissible += 1
        p = sorted(set(r["profile"]))
        for size in ps["sizes"]:
            for comb in combinations(p, size):
                support[comb] += 1
    from_profiles = {s for s, n in support.items() if n >= ps["min_support"]}

    # --- Generator B: dense high-pace triangles -----------------------------
    edges = _edges_at(con, T, cfg)
    adj: dict[str, set] = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    pcw = _paper_concepts(con, T, T - 2, T)
    cp: dict[str, set] = defaultdict(set)
    for pid, cs in pcw.items():
        for c in cs:
            cp[c].add(pid)

    def pair_pace(u, v):
        co = cp[u] & cp[v]
        return len(co) / 3.0 if co else 0.0        # window is graph_window years

    tri = []
    for u, v in edges:
        for w in adj[u] & adj[v]:
            if w > v > u:
                p = min(pair_pace(u, v), pair_pace(u, w), pair_pace(v, w))
                if p > 0:
                    tri.append((p, (u, v, w)))
    tri.sort(key=lambda x: -x[0])
    from_triangles = {t for _, t in tri[:ps["triangle_top_n"]]}

    # Take the two generators' best candidates SEPARATELY before capping.
    # Ranking the union by support silently discards every triangle, since a
    # triangle that no past birth grew out of has support 0 and sorts last.
    half = ps["max_units_per_origin"] // 2
    top_prof = sorted(from_profiles, key=lambda s: -support.get(s, 0))[:half]
    top_tri = [t for _, t in tri[:ps["triangle_top_n"]]]
    units = {}
    for s in top_prof:
        units[s] = "sub_profile"
    for s in top_tri:
        units[s] = "both" if s in units else "triangle"
    ranked = list(units.items())[:ps["max_units_per_origin"]]

    # --- features, mirroring the region spec where it applies ---------------
    meta = dict(con.execute(f"""
        SELECT id, struct_pack(y := v1_year, q := v1_quarter, a := authors_parsed)
        FROM read_parquet('data/interim/papers.parquet')
        WHERE v1_year BETWEEN {T-4} AND {T}""").fetchall())
    pc5 = _paper_concepts(con, T, T - 4, T)
    cp5: dict[str, set] = defaultdict(set)
    for pid, cs in pc5.items():
        for c in cs:
            cp5[c].add(pid)
    emb = load_emb(T) if Path(f"data/graphs/embeddings/emb_{T}.npz").exists() else {}

    rows = []
    for s, origin_kind in ranked:
        # Exposure is the OFFSET -- how much activity surrounds this unit -- and
        # is defined as for regions: papers naming at least one member. Defining
        # it as the conjunction instead made most sub-profile sets zero-exposure
        # and dropped them, which is exactly backwards: a set whose members have
        # not met yet is the interesting parent-set, not the disqualified one.
        # Whether the members are actually meeting is a separate feature.
        hits: dict[str, int] = defaultdict(int)
        for c in s:
            for pid in cp5.get(c, ()):
                hits[pid] += 1
        joint = set(hits)                                  # union: >= 1 member
        conj = {pid for pid, h in hits.items() if h >= min(2, len(s))}
        conj_T = sum(1 for pid in conj if meta.get(pid) and meta[pid]["y"] == T)
        by_year: dict[int, int] = defaultdict(int)
        quarter: dict[tuple, int] = defaultdict(int)
        for pid in joint:
            mm = meta.get(pid)
            if mm:
                by_year[mm["y"]] += 1
                quarter[(mm["y"], mm["q"])] += 1
        n = {y: by_year.get(y, 0) for y in range(T - 4, T + 1)}
        if n[T] == 0:
            continue
        acc = float(np.mean([
            len_q(quarter, T, q) / max(1, len_q(quarter, T - 1, q))
            - len_q(quarter, T - 1, q) / max(1, len_q(quarter, T - 2, q))
            for q in (1, 2, 3, 4)]))
        auth = {a for pid in joint if meta.get(pid) and meta[pid]["y"] == T
                for a in (meta[pid]["a"] or ())}
        auth_prev = {a for pid in joint if meta.get(pid) and T - 2 <= meta[pid]["y"] <= T - 1
                     for a in (meta[pid]["a"] or ())}
        vs = [emb[c] for c in s if c in emb]
        spread = None
        if len(vs) >= 2:
            V = np.vstack(vs)
            cen = V.mean(axis=0)
            cen /= max(1e-9, np.linalg.norm(cen))
            spread = float(np.mean(1 - V @ cen))
        rows.append({
            "origin": T, "unit_type": "parent_set", "unit_id": " + ".join(s),
            "size": len(s), "generator": origin_kind,
            "support_past_births": support.get(s, 0),
            "exposure": n[T], "n_T": n[T], "n_T1": n[T - 1], "n_T2": n[T - 2],
            "paper_velocity_1y": n[T] / max(1, n[T - 1]),
            "paper_velocity_2y": n[T] / max(1.0, (n[T - 1] + n[T - 2]) / 2),
            "yoy_quarter_accel": acc,
            "pace": len(joint) / 3.0,
            "joint_papers_T": conj_T,
            "joint_papers_window": len(conj),
            "joint_share": conj_T / max(1, n[T]),
            "members_have_met": bool(conj),
            "author_influx": len(auth - auth_prev) / max(1, len(auth)) if auth else None,
            "embedding_spread": spread,
            "embedding_missing": spread is None,
            "members": list(s),
        })
    return rows


def len_q(q, y, k):
    return q.get((y, k), 0)


def build_all(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    assert cfg["regions"]["feature_spec_signed_off"], "Phase 6 feature gate not passed"
    if not PROFILES.exists():
        print("building birth profiles...", flush=True)
        print("  ", build_profiles(cfg), flush=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{cfg['runtime']['mem_budget_gb'] - 4}GB'")
    lo = cfg["evaluation"]["origins_tune"][0] + 2
    hi = cfg["evaluation"]["origins_test"][1]
    allrows, stats = [], []
    for T in range(lo, hi + 1):
        r = build_origin(con, cfg, T)
        allrows += r
        gen = defaultdict(int)
        for x in r:
            gen[x["generator"]] += 1
        stats.append({"origin": T, "rows": len(r), "by_generator": dict(gen)})
        print(f"  T={T}: {len(r):>5,} parent-set units  {dict(gen)}", flush=True)
    con.close()
    out = FDIR / "parent_set_features.parquet"
    pq.write_table(pa.Table.from_pylist(allrows), out, compression="zstd")
    agg = {"rows": len(allrows), "origins": [lo, hi], "per_origin": stats}
    Manifest.build(str(out), phase="6", inputs=[str(PROFILES)], cfg=cfg,
                   params=cfg["parent_sets"], stats=agg).write(out)
    return agg


if __name__ == "__main__":
    import json
    print(json.dumps(build_all(), indent=2, default=str))
