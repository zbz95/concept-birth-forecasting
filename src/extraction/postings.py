"""Phase 3 — term-level postings for every candidate that could ever be a node.

The ledger aggregates away which papers a candidate appeared in, but three later
stages need exactly that: Phase 4's author-group components (over the papers
attesting a concept in a window), Phase 5's event store, and cluster-level
counting (a paper naming both `cnn` and `convolutional neural network` must
count once for the merged cluster, not twice).

Scope is every candidate with global df >= min_total_freq. That is a strict
superset of every possible vocab_T, because a term's document frequency only
grows with time: df_T >= min_total_freq implies global df >= min_total_freq. So
this table can be built once, globally, and every per-origin filter — frequency,
pattern kills, C-value, merge activation — recomputed from it against data <= T
without leaking anything.
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.extraction.candidates import _segments  # noqa: E402
from src.manifest import Manifest, load_config  # noqa: E402

OUT = "data/interim/concept_postings.parquet"
BATCH = 8_000

SCHEMA = pa.schema([("paper_id", pa.string()), ("term", pa.string()),
                    ("v1_date", pa.date32()), ("year", pa.int16())])


def _scan(args):
    lemmas, pos, year, d, pid = args
    hits = set()
    for seg in _segments(lemmas, pos):
        L = len(seg)
        for n in range(1, _MAXN + 1):
            if n > L:
                break
            for s in range(L - n + 1):
                g = " ".join(seg[s:s + n]) if n > 1 else seg[s]
                if g in _TERMS:
                    hits.add(g)
    return pid, d, year, hits


def build(cfg: dict | None = None) -> dict:
    global _TERMS, _MAXN
    cfg = cfg or load_config()
    min_df = cfg["vocabulary"]["min_total_freq"]

    con = duckdb.connect()
    rows = con.execute(f"""SELECT cand FROM read_parquet('data/interim/ledger.parquet')
                           WHERE df >= {min_df}""").fetchall()
    con.close()
    _TERMS = {c for (c,) in rows}
    _MAXN = max(c.count(" ") for c in _TERMS) + 1
    print(f"scanning for {len(_TERMS):,} terms, max {_MAXN} tokens", flush=True)

    pf = pq.ParquetFile("data/interim/tokens.parquet")
    total = pf.metadata.num_rows
    writer = pq.ParquetWriter(OUT, SCHEMA, compression="zstd")
    done = n_rows = 0
    with Pool(cfg["runtime"]["n_workers"]) as pool:
        for batch in pf.iter_batches(batch_size=BATCH):
            d = batch.to_pydict()
            work = list(zip(d["lemmas"], d["pos"], d["v1_year"], d["v1_date"], d["id"]))
            cols = {"paper_id": [], "term": [], "v1_date": [], "year": []}
            for pid, dt, yr, hits in pool.imap_unordered(_scan, work, chunksize=100):
                for h in hits:
                    cols["paper_id"].append(pid); cols["term"].append(h)
                    cols["v1_date"].append(dt); cols["year"].append(yr)
            n_rows += len(cols["term"])
            writer.write_table(pa.Table.from_pydict(cols, schema=SCHEMA))
            done += len(work)
            print(f"  {done:>7,}/{total:,}  postings {n_rows:>12,}", flush=True)
    writer.close()

    stats = {"terms": len(_TERMS), "papers": done, "postings": n_rows,
             "min_total_freq": min_df}
    Manifest.build(OUT, phase="3", inputs=["data/interim/tokens.parquet",
                                          "data/interim/ledger.parquet"],
                   cfg=cfg, params={"min_total_freq": min_df,
                                    "scope": "global superset of every vocab_T"},
                   stats=stats).write(OUT)
    return stats


if __name__ == "__main__":
    import json
    print(json.dumps(build(), indent=2, default=str))
