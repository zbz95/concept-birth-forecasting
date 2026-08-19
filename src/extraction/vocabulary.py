"""Phase 3 — vocab_T: the modeling vocabulary, rebuilt causally at every origin.

Every filter in the stack is recomputed from data dated <= T. That is not a
refinement, it is the whole point: a global filter would admit a term into
vocab_2019 partly *because* it became popular in 2023, which is leakage-checklist
item 1 in its purest form.

Per origin T:
  1. df_T        — papers dated <= T naming the term
  2. frequency   — df_T >= min_total_freq
  3. pattern kills — using a POS table derived from abstracts dated <= T only
  4. C-value     — computed from df_T and containment among the <= T survivors
  5. merges      — only edges whose evidence date is <= T are active
  6. cluster df_T — a paper naming several members of one cluster counts ONCE

No C-value floor is applied here. `c_value_min` is null in config by design; the
plan reserves it for the PI at the Phase 3 gate, and picking it to land the
vocabulary inside its expected band is precisely what Principle 6 forbids.
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.extraction.filters import kill_reason  # noqa: E402
from src.manifest import Manifest, load_config  # noqa: E402

LEMMA_POS_YEAR = "data/interim/lemma_pos_year.parquet"
OUTDIR = Path("data/interim/vocab")


def build_lemma_pos_year(cfg: dict) -> None:
    """(lemma, pos, year) counts, unnested once so every origin is a cheap query."""
    if Path(LEMMA_POS_YEAR).exists():
        return
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{cfg['runtime']['mem_budget_gb'] - 3}GB'")
    con.execute(f"""COPY (
        SELECT lemma, pos, year, count(*)::BIGINT AS n FROM (
          SELECT unnest(lemmas) AS lemma, unnest(pos) AS pos, v1_year AS year
          FROM read_parquet('data/interim/tokens.parquet'))
        GROUP BY 1,2,3
      ) TO '{LEMMA_POS_YEAR}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    con.close()


def _pos_map_at(con, T: int) -> dict[str, str]:
    rows = con.execute(f"""
        SELECT lemma, arg_max(pos, n) FROM (
          SELECT lemma, pos, sum(n) AS n FROM read_parquet('{LEMMA_POS_YEAR}')
          WHERE year <= {T} GROUP BY 1,2)
        GROUP BY lemma""").fetchall()
    return dict(rows)


def _c_value(freq: dict[str, int]) -> dict[str, float]:
    cands = set(freq)
    nested_sum: dict[str, int] = defaultdict(int)
    nested_n: dict[str, int] = defaultdict(int)
    for cand in cands:
        toks = cand.split()
        n = len(toks)
        if n < 2:
            continue
        seen = set()
        for ln in range(1, n):
            for s in range(n - ln + 1):
                sub = " ".join(toks[s:s + ln])
                if sub in seen or sub not in cands:
                    continue
                seen.add(sub)
                nested_sum[sub] += freq[cand]
                nested_n[sub] += 1
    out = {}
    for c, f in freq.items():
        w = math.log2(len(c.split()) + 1)
        k = nested_n.get(c, 0)
        out[c] = w * (f - nested_sum[c] / k) if k else w * f
    return out


def build_origin(con, cfg: dict, T: int) -> dict:
    min_df = cfg["vocabulary"]["min_total_freq"]

    # 1-2. frequency, from data <= T only
    freq = dict(con.execute(f"""
        SELECT cand, sum(n_papers)::INTEGER FROM read_parquet('data/interim/ledger_yearly.parquet')
        WHERE year <= {T} GROUP BY 1 HAVING sum(n_papers) >= {min_df}""").fetchall())

    # 3. pattern kills against a <= T POS table
    pos_map = _pos_map_at(con, T)
    kills: dict[str, int] = defaultdict(int)
    surv = {}
    for c, f in freq.items():
        r = kill_reason(c, pos_map)
        if r is None:
            surv[c] = f
        else:
            kills[r] += 1

    # 4. C-value among the <= T survivors
    cv = _c_value(surv)

    # 5. merges active at T
    cutoff = date(T, 12, 31)
    m = con.execute(f"""
        SELECT cand, cluster FROM read_parquet('data/interim/merge_map.parquet')
        WHERE effective_date <= DATE '{cutoff}'""").fetchall()
    cl_of = {c: cl for c, cl in m}
    # A member whose merge is not yet effective stands alone at this origin.
    cluster = {c: (cl_of.get(c, c) if cl_of.get(c, c) in surv else c) for c in surv}

    # 6. cluster-level df_T, deduplicated per paper
    con.execute("DROP TABLE IF EXISTS t2c")
    con.execute("CREATE TEMP TABLE t2c (term VARCHAR, cluster VARCHAR)")
    con.executemany("INSERT INTO t2c VALUES (?,?)", list(cluster.items()))
    con.execute("DROP TABLE IF EXISTS cvt")
    con.execute("CREATE TEMP TABLE cvt (term VARCHAR, c_value DOUBLE)")
    con.executemany("INSERT INTO cvt VALUES (?,?)", list(cv.items()))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"vocab_{T}.parquet"
    con.execute(f"""COPY (
        SELECT c.cluster,
               count(DISTINCT p.paper_id)::INTEGER AS df_T,
               count(DISTINCT p.term)::INTEGER     AS n_members,
               max(v.c_value)                      AS c_value,
               min(p.v1_date)                      AS first_date
        FROM read_parquet('data/interim/concept_postings.parquet') p
        JOIN t2c c ON c.term = p.term
        JOIN cvt v ON v.term = p.term
        WHERE p.year <= {T}
        GROUP BY c.cluster
      ) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)""")

    n_clusters, n_multi = con.execute(f"""
        SELECT count(*), sum(CASE WHEN n_members>1 THEN 1 ELSE 0 END)
        FROM read_parquet('{out}')""").fetchone()
    grid = {f: con.execute(
        f"SELECT count(*) FROM read_parquet('{out}') WHERE c_value > {f}").fetchone()[0]
        for f in (0, 5, 10, 15, 25, 50)}

    stats = {"origin": T, "terms_at_min_df": len(freq), "killed": dict(kills),
             "surviving_terms": len(surv), "clusters": n_clusters,
             "multi_member_clusters": int(n_multi or 0), "clusters_by_c_value_floor": grid}
    Manifest.build(str(out), phase="3", as_of=cutoff, max_observed_date=cutoff,
                   inputs=["data/interim/ledger_yearly.parquet",
                           "data/interim/concept_postings.parquet",
                           "data/interim/merge_map.parquet"],
                   cfg=cfg, params={"min_total_freq": min_df,
                                    "c_value_min": cfg["vocabulary"]["c_value_min"]},
                   stats=stats).write(out)
    return stats


def build_all(cfg: dict | None = None) -> list[dict]:
    cfg = cfg or load_config()
    build_lemma_pos_year(cfg)
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{cfg['runtime']['mem_budget_gb'] - 3}GB'")
    lo = cfg["evaluation"]["origins_tune"][0]
    hi = cfg["evaluation"]["origins_test"][1]
    out = []
    for T in range(lo, hi + 1):
        s = build_origin(con, cfg, T)
        print(f"  T={T}  terms>=df9 {s['terms_at_min_df']:>7,} -> surviving "
              f"{s['surviving_terms']:>7,} -> clusters {s['clusters']:>7,}", flush=True)
        out.append(s)
    con.close()
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(build_all(), indent=2, default=str))
