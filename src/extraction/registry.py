"""Phase 4 — the birth registry.

Per concept, two dates and a fate:

  coinage         first occurrence anywhere in the ledger
  crystallization first year t such that, using data dated <= t+m-1 ONLY,
                  the concept has >= k_year papers in EACH of the m years
                  t..t+m-1, and the papers qualifying across that whole window
                  fall into >= min_groups disjoint author groups
  fate            coinage-only | crystallized-then-declined | persisted

Three plan readings confirmed by the PI on 2026-08-19 are load-bearing here:
per-year (not pooled) frequency across the window; author groups pooled over the
whole window rather than per year; and a paper naming several members of one
merge cluster counting once.

Merge activation is evidence-dated: a cluster's count at year y sums only the
members whose merge is effective by y. A member whose merge is not yet effective
counts toward its own series, as its own concept. This is what stops a 2024
identity judgement from rewriting a 2016 count.

Censoring. Crystallization at t needs data through t+m-1, so with the corpus
complete through 2025 the registry is complete through 2024. Entries that would
require data past the corpus cutoff are marked censored and excluded from
training targets at affected origins.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.manifest import Manifest, load_config, log_event  # noqa: E402

OUT = "data/registry/births.parquet"
SERIES = "data/registry/concept_year.parquet"

SCHEMA = pa.schema([
    ("concept", pa.string()),
    ("coinage_date", pa.date32()),
    ("coinage_year", pa.int16()),
    ("crystallization_year", pa.int16()),      # null if never
    ("n_author_groups", pa.int16()),           # at crystallization
    ("decline_year", pa.int16()),              # null unless declined
    ("fate", pa.string()),
    ("censored", pa.bool_()),
    ("peak_year", pa.int16()),
    ("peak_papers", pa.int32()),
    ("total_papers", pa.int32()),
    ("as_of", pa.date32()),
])


def build_series(cfg: dict) -> None:
    """Per-concept, per-year deduplicated paper counts under evidence-dated merges."""
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{cfg['runtime']['mem_budget_gb'] - 3}GB'")
    Path("data/registry").mkdir(parents=True, exist_ok=True)
    # A term maps to its cluster only once the merge evidence exists; before
    # that it is its own concept.
    con.execute(f"""COPY (
        SELECT concept, year, count(DISTINCT paper_id)::INTEGER AS n_papers
        FROM (
          SELECT CASE WHEN m.effective_date <= make_date(p.year, 12, 31)
                      THEN m.cluster ELSE p.term END AS concept,
                 p.year, p.paper_id
          FROM read_parquet('data/interim/concept_postings.parquet') p
          JOIN read_parquet('data/interim/merge_map.parquet') m ON m.cand = p.term
          JOIN read_parquet('data/interim/termhood.parquet') t ON t.cand = p.term
          WHERE t.c_value > {cfg['vocabulary']['c_value_min']}
        )
        GROUP BY 1, 2
      ) TO '{SERIES}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    con.close()


def _author_groups(papers: set[str], paper_authors: dict[str, list[str]]) -> int:
    """Disjoint author-group components over the papers attesting a concept.

    Papers are linked when they share any author (exact parsed-name match).
    Name collisions merge groups, which makes crystallization *harder* and so
    biases away from false births. The plan accepts that direction explicitly.
    """
    if not papers:
        return 0
    by_author: dict[str, list[str]] = defaultdict(list)
    for pid in papers:
        for a in paper_authors.get(pid, ()):
            by_author[a].append(pid)

    parent = {p: p for p in papers}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for pids in by_author.values():
        r = find(pids[0])
        for other in pids[1:]:
            ro = find(other)
            if ro != r:
                parent[ro] = r
    return len({find(p) for p in papers})


def build(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    rc = cfg["registry"]
    k, m = rc["k_year"], rc["m"]
    last_complete = rc["last_complete_year"]
    complete_through = last_complete - (m - 1)

    if not Path(SERIES).exists():
        print("building concept-year series...", flush=True)
        build_series(cfg)

    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{cfg['runtime']['mem_budget_gb'] - 3}GB'")

    series: dict[str, dict[int, int]] = defaultdict(dict)
    for c, y, n in con.execute(f"SELECT concept, year, n_papers FROM read_parquet('{SERIES}')").fetchall():
        series[c][y] = n
    print(f"concepts in series: {len(series):,}", flush=True)

    # Members of each cluster, for the author-group query.
    members: dict[str, list[str]] = defaultdict(list)
    for cand, cl in con.execute(
            "SELECT cand, cluster FROM read_parquet('data/interim/merge_map.parquet')").fetchall():
        members[cl].append(cand)

    # Coinage comes from the ledger — the untimed, unfiltered tier.
    coinage = dict(con.execute(
        "SELECT cand, first_date FROM read_parquet('data/interim/ledger.parquet')").fetchall())

    # ---- candidate windows first, so the author-group data can be fetched once
    # A per-(concept, t) query against a 77M-row table would run thousands of
    # times; instead every window that passes the frequency test is collected,
    # and exactly the postings those windows need are fetched in one pass.
    candidates: dict[str, list[int]] = {}
    censored_set: set[str] = set()
    for concept, ys in series.items():
        if not ys:
            continue
        ts = []
        for t in range(min(ys), last_complete + 1):
            # A window running past the corpus is INDETERMINATE, not failed. At
            # t = 2025 with m = 2 the rule needs 2026, which does not exist, so
            # testing all m years would silently classify a concept that may well
            # be crystallizing right now as coinage-only. Judge such a window on
            # the years actually observable and mark it censored.
            if t + m - 1 > last_complete:
                # Only censored if nothing earlier already settled the question.
                # A concept that crystallized in 2018 and is still active in 2025
                # has a determinate birth date; its boundary window is irrelevant.
                if not ts and all(ys.get(y, 0) >= k for y in range(t, last_complete + 1)):
                    censored_set.add(concept)
                break
            if not all(ys.get(t + i, 0) >= k for i in range(m)):
                continue
            ts.append(t)
        if ts:
            candidates[concept] = ts
    print(f"concepts with a qualifying frequency window: {len(candidates):,}", flush=True)

    needed_terms = set()
    for concept in candidates:
        needed_terms.update(members.get(concept, [concept]))
    con.execute("DROP TABLE IF EXISTS need")
    con.execute("CREATE TEMP TABLE need (term VARCHAR)")
    con.executemany("INSERT INTO need VALUES (?)", [(t,) for t in needed_terms])
    postings = con.execute("""
        SELECT p.term, p.year, p.paper_id
        FROM read_parquet('data/interim/concept_postings.parquet') p
        JOIN need n ON n.term = p.term""").fetchall()
    term_year_papers: dict[tuple[str, int], list[str]] = defaultdict(list)
    for term, y, pid in postings:
        term_year_papers[(term, y)].append(pid)
    del postings
    print(f"postings loaded for author grouping: {len(term_year_papers):,} (term, year) keys",
          flush=True)

    paper_authors: dict[str, list[str]] = {}
    for pid, auth in con.execute(
            "SELECT id, authors_parsed FROM read_parquet('data/interim/papers.parquet')").fetchall():
        paper_authors[pid] = auth

    rows, n_group_tests, censored_n = [], 0, 0
    fates = defaultdict(int)
    for concept, ys in series.items():
        if not ys:
            continue
        total = sum(ys.values())
        peak_year = max(ys, key=lambda y: ys[y])
        # Coinage of a cluster is the EARLIEST first-occurrence among its
        # members, not the label's own. Taking the label's date let a cluster
        # crystallize before its own coinage (observed lag of -1 years) whenever
        # a member had appeared earlier than the surface form that names it.
        cd = min((coinage[t] for t in members.get(concept, [concept]) if t in coinage),
                 default=coinage.get(concept))
        if cd is None:
            continue

        cry = groups = None
        censored = concept in censored_set
        terms = members.get(concept, [concept])
        for t in candidates.get(concept, ()):
            papers = set()
            for i in range(m):
                for term in terms:
                    papers.update(term_year_papers.get((term, t + i), ()))
            g = _author_groups(papers, paper_authors)
            n_group_tests += 1
            if g >= rc["min_groups"]:
                cry, groups = t, g
                break

        decline = None
        if cry is not None:
            # decline: < k papers per year for m consecutive years after crystallization
            for t in range(cry + m, last_complete - m + 2):
                if all(ys.get(t + i, 0) < k for i in range(m)):
                    decline = t
                    break

        if cry is None:
            fate = "censored" if censored else "coinage_only"
        elif decline is not None:
            fate = "crystallized_then_declined"
        else:
            fate = "persisted"
        fates[fate] += 1
        if censored:
            censored_n += 1

        rows.append({
            "concept": concept, "coinage_date": cd, "coinage_year": cd.year,
            "crystallization_year": cry, "n_author_groups": groups,
            "decline_year": decline, "fate": fate, "censored": censored,
            "peak_year": peak_year, "peak_papers": ys[peak_year],
            "total_papers": total,
            "as_of": __import__("datetime").date(complete_through, 12, 31),
        })

    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), OUT, compression="zstd")
    con.close()

    con = duckdb.connect()
    per_year = dict(con.execute(f"""SELECT crystallization_year, count(*)
        FROM read_parquet('{OUT}') WHERE crystallization_year IS NOT NULL
        GROUP BY 1 ORDER BY 1""").fetchall())
    con.close()

    stats = {"concepts": len(rows), "fates": dict(fates), "censored": censored_n,
             "author_group_tests": n_group_tests,
             "crystallizations_per_year": {int(y): int(n) for y, n in per_year.items()},
             "complete_through": complete_through}

    # Phase 4 power gate.
    floor = cfg["evaluation"]["power_floor"]
    lo, hi = cfg["evaluation"]["origins_test"]
    starved = {y: n for y, n in per_year.items() if lo <= y <= min(hi, complete_through) and n < floor}
    if starved:
        log_event("logs/flags.jsonl", {
            "phase": "4", "kind": "power_gate_failure", "power_floor": floor,
            "starved_origins": {int(y): int(n) for y, n in starved.items()},
            "action": "STOP — design starves. PI decides: lower k_year, or widen corpus to cs.LG."})
    stats["power_gate"] = "FAIL" if starved else "PASS"

    Manifest.build(OUT, phase="4",
                   inputs=[SERIES, "data/interim/merge_map.parquet",
                           "data/interim/ledger.parquet", "data/interim/papers.parquet"],
                   cfg=cfg, params={k2: rc[k2] for k2 in
                                    ("k_year", "m", "min_groups", "window_rule",
                                     "groups_scope", "author_match")},
                   stats=stats).write(OUT)
    return stats


if __name__ == "__main__":
    import json
    print(json.dumps(build(), indent=2, default=str))
