"""Phase 5 — event store and the derived projection graphs.

The **event store** is the primary object: (paper, term, v1_date). It is never
capped, filtered, or reweighted. Every graph is a disposable view derived from
it, so a change to the projection rule never touches the underlying record of
what appeared where and when.

Terms rather than concepts are stored, because concept identity is a function of
time: a merge becomes effective on the date its evidence appears, so the same
term resolves to different concepts at different origins. Resolving at view-build
time is what keeps a 2024 identity judgement out of a 2016 graph.

**graph_T** projects the events in the trailing `graph_window` years onto
concept-concept edges. Weighting is fractional: each paper spends exactly 1.0 of
edge mass, split evenly over its C(k,2) concept pairs, so a paper naming forty
concepts cannot outvote forty papers naming two. That gives the invariant this
module asserts:

    total edge mass in graph_T == number of papers in the window with >= 2 concepts

The cap (`max_concepts_per_paper`) applies to the projection layer ONLY, never to
the event store, and keeps the *lowest document-frequency* concepts — the rarest,
most specific ones — because a paper's hub terms carry the least information
about what makes it distinctive.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.manifest import Manifest, load_config, log_event  # noqa: E402

EVENTS = "data/graphs/event_store.parquet"
GDIR = Path("data/graphs")

EDGE_SCHEMA = pa.schema([("u", pa.string()), ("v", pa.string()),
                         ("weight", pa.float64()), ("n_papers", pa.int32())])


def build_event_store(cfg: dict) -> dict:
    """(paper, term, v1_date) over every term that can ever be a node.

    Scope is the global filter survivors — pattern kills, C-value, and the LLM
    judge if enabled. Frequency is deliberately NOT applied per origin here; that
    is a view-level decision, and the store must be able to serve every origin.
    """
    GDIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{cfg['runtime']['mem_budget_gb'] - 3}GB'")
    judge_clause = ""
    if cfg["llm_judge"]["enabled"]:
        judge_clause = """
          AND NOT EXISTS (SELECT 1 FROM read_parquet('data/interim/judge_verdicts.parquet') j
                          WHERE j.term = m.cluster AND j.verdict = 'DROP')"""
    con.execute(f"""COPY (
        SELECT p.paper_id, p.term, p.v1_date, p.year
        FROM read_parquet('data/interim/concept_postings.parquet') p
        JOIN read_parquet('data/interim/termhood.parquet') t ON t.cand = p.term
        JOIN read_parquet('data/interim/merge_map.parquet') m ON m.cand = p.term
        WHERE t.c_value > {cfg['vocabulary']['c_value_min']}{judge_clause}
      ) TO '{EVENTS}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    n, npap, nterm, d0, d1 = con.execute(f"""
        SELECT count(*), count(DISTINCT paper_id), count(DISTINCT term),
               min(v1_date), max(v1_date) FROM read_parquet('{EVENTS}')""").fetchone()
    con.close()
    stats = {"events": n, "papers": npap, "terms": nterm,
             "first_date": str(d0), "last_date": str(d1),
             "judge_applied": cfg["llm_judge"]["enabled"]}
    Manifest.build(EVENTS, phase="5",
                   inputs=["data/interim/concept_postings.parquet",
                           "data/interim/termhood.parquet",
                           "data/interim/merge_map.parquet"],
                   cfg=cfg, params={"scope": "global filter survivors; never capped or reweighted"},
                   stats=stats).write(EVENTS)
    return stats


def build_graph(con, cfg: dict, T: int) -> dict:
    """Project graph_T from the trailing window. Returns stats; writes edges."""
    gc = cfg["graph"]
    win = gc["graph_window"]
    cap = gc["max_concepts_per_paper"]
    lo = T - win + 1

    # Nodes: vocab_T, with df_T for the cap's lowest-DF rule.
    vocab = dict(con.execute(
        f"SELECT cluster, df_T FROM read_parquet('data/interim/vocab/vocab_{T}.parquet')").fetchall())

    # Events in the window, resolved term -> concept using only merges whose
    # evidence date is <= T. A member whose merge is not yet effective stands
    # alone, exactly as in the registry.
    rows = con.execute(f"""
        SELECT p.paper_id,
               CASE WHEN m.effective_date <= DATE '{T}-12-31' THEN m.cluster ELSE p.term END
        FROM read_parquet('{EVENTS}') p
        JOIN read_parquet('data/interim/merge_map.parquet') m ON m.cand = p.term
        WHERE p.year BETWEEN {lo} AND {T}""").fetchall()

    per_paper: dict[str, set] = {}
    for pid, concept in rows:
        if concept in vocab:
            per_paper.setdefault(pid, set()).add(concept)
    del rows

    n_papers_window = con.execute(
        f"SELECT count(*) FROM read_parquet('data/interim/papers.parquet') "
        f"WHERE v1_year BETWEEN {lo} AND {T}").fetchone()[0]

    edges: dict[tuple, list] = {}
    n_capped = n_ge2 = 0
    node_papers: dict[str, int] = {}
    for pid, concepts in per_paper.items():
        cs = concepts
        if len(cs) > cap:
            # Projection layer only. Keep the rarest concepts: a paper's hub
            # terms say least about what makes it distinctive.
            cs = set(sorted(cs, key=lambda c: vocab[c])[:cap])
            n_capped += 1
        for c in cs:
            node_papers[c] = node_papers.get(c, 0) + 1
        k = len(cs)
        if k < 2:
            continue
        n_ge2 += 1
        share = 1.0 / (k * (k - 1) / 2)          # each paper spends exactly 1.0
        for u, v in combinations(sorted(cs), 2):
            e = edges.get((u, v))
            if e is None:
                edges[(u, v)] = [share, 1]
            else:
                e[0] += share
                e[1] += 1

    total_mass = sum(w for w, _ in edges.values())
    # INVARIANT: fractional weighting conserves mass exactly in exact arithmetic.
    # In floating point, summing millions of 1/C(k,2) terms accumulates rounding,
    # so the check is relative: at n_ge2 ~ 2.5e4 the observed drift is ~4e-11.
    assert n_ge2 == 0 or abs(total_mass - n_ge2) / n_ge2 < 1e-9, (
        f"graph_{T}: edge mass {total_mass:.6f} != {n_ge2} papers with >=2 concepts")

    out = GDIR / f"graph_{T}_edges.parquet"
    pq.write_table(pa.Table.from_pydict({
        "u": [u for u, _ in edges], "v": [v for _, v in edges],
        "weight": [w for w, _ in edges.values()],
        "n_papers": [n for _, n in edges.values()],
    }, schema=EDGE_SCHEMA), out, compression="zstd")

    thr = gc["binarize_min_weight"]
    bin_edges = sum(1 for w, _ in edges.values() if w >= thr)
    bin_nodes = len({x for (u, v), (w, _) in zip(edges.keys(), edges.values())
                     if w >= thr for x in (u, v)})

    stats = {
        "origin": T, "window": [lo, T],
        "papers_in_window": n_papers_window,
        "papers_with_any_concept": len(per_paper),
        "papers_with_2plus_concepts": n_ge2,
        "coverage_2plus": round(n_ge2 / n_papers_window, 4),
        "papers_capped": n_capped,
        "vocab_T_nodes": len(vocab),
        "nodes_in_graph": len(node_papers),
        "edges_weighted": len(edges),
        "total_edge_mass": round(total_mass, 6),
        "invariant_holds": True,
        "nodes_binarized": bin_nodes,
        "edges_binarized": bin_edges,
    }
    Manifest.build(str(out), phase="5", as_of=f"{T}-12-31", max_observed_date=f"{T}-12-31",
                   inputs=[EVENTS, f"data/interim/vocab/vocab_{T}.parquet",
                           "data/interim/merge_map.parquet"],
                   cfg=cfg, params={k: gc[k] for k in
                                    ("graph_window", "max_concepts_per_paper",
                                     "cap_keeps", "weighting", "binarize_min_weight")},
                   stats=stats).write(out)
    if n_capped:
        log_event("logs/flags.jsonl", {
            "phase": "5", "kind": "papers_capped", "origin": T,
            "n_capped": n_capped, "cap": cap,
            "note": "projection layer only; event store untouched"})
    return stats


def build_all(cfg: dict | None = None) -> list[dict]:
    cfg = cfg or load_config()
    if not Path(EVENTS).exists():
        print("building event store...", flush=True)
        print("  ", build_event_store(cfg), flush=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{cfg['runtime']['mem_budget_gb'] - 3}GB'")
    lo = cfg["evaluation"]["origins_tune"][0]
    hi = cfg["evaluation"]["origins_test"][1]
    out = []
    for T in range(lo, hi + 1):
        s = build_graph(con, cfg, T)
        print(f"  T={T}  nodes {s['nodes_in_graph']:>7,} edges {s['edges_weighted']:>10,} "
              f"| binarized {s['nodes_binarized']:>7,} / {s['edges_binarized']:>9,} "
              f"| coverage {s['coverage_2plus']:.1%} capped {s['papers_capped']:,}", flush=True)
        out.append(s)
    con.close()
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(build_all(), indent=2, default=str))
