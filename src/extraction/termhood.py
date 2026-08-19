"""Phase 3 — C-value termhood.

C-value (Frantzi, Ananiadou & Mima) ranks a candidate by how much it behaves
like a term in its own right rather than as a fragment of longer terms. A
candidate that occurs almost exclusively inside longer candidates gets its
frequency discounted by theirs; one that stands alone keeps it.

    C-value(t) = log2(|t|+1) * ( f(t) - (1/|T_t|) * sum_{b in T_t} f(b) )

where T_t is the set of candidates properly containing t, and the discount term
is dropped when T_t is empty.

The textbook form uses log2(|t|), which zeroes every single-token candidate.
That is unusable here: four of the plan's ten Phase 2 spot-list terms are
unigrams (transformer, BERT, NeRF, GAN), so log2(|t|+1) is used instead.

No floor is applied. `c_value_min` is null in config by design — the plan
reserves that threshold for the PI at the Phase 3 gate, and choosing it here to
land the vocabulary inside its expected band would be exactly the Goodharting
Principle 6 forbids.
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.manifest import Manifest, load_config  # noqa: E402

SRC = "data/interim/vocab_candidates.parquet"
OUT = "data/interim/termhood.parquet"


def compute(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    con = duckdb.connect()
    rows = con.execute(f"SELECT cand, df, first_date, n_tokens, src_mask "
                       f"FROM read_parquet('{SRC}')").fetchall()
    con.close()

    freq = {c: df for c, df, *_ in rows}
    cands = set(freq)

    # For each candidate, find the candidates properly containing it. Walk the
    # containers once and register each of their proper sub-grams.
    nested_sum: dict[str, int] = defaultdict(int)
    nested_n: dict[str, int] = defaultdict(int)
    for cand in cands:
        toks = cand.split()
        n = len(toks)
        if n < 2:
            continue
        seen_sub = set()
        for ln in range(1, n):
            for s in range(n - ln + 1):
                sub = " ".join(toks[s:s + ln])
                if sub in seen_sub or sub not in cands:
                    continue
                seen_sub.add(sub)
                nested_sum[sub] += freq[cand]
                nested_n[sub] += 1

    out = []
    for cand, df, first_date, n_tokens, src_mask in rows:
        weight = math.log2(len(cand.split()) + 1)
        k = nested_n.get(cand, 0)
        cv = weight * (df - nested_sum[cand] / k) if k else weight * df
        out.append((cand, df, float(cv), k, first_date, n_tokens, src_mask))

    schema = pa.schema([("cand", pa.string()), ("df", pa.int32()),
                        ("c_value", pa.float64()), ("n_containers", pa.int32()),
                        ("first_date", pa.date32()), ("n_tokens", pa.int8()),
                        ("src_mask", pa.int8())])
    pq.write_table(pa.Table.from_pylist(
        [dict(zip([f.name for f in schema], r)) for r in out], schema=schema),
        OUT, compression="zstd")

    con = duckdb.connect()
    qs = con.execute(f"""SELECT
        min(c_value), quantile_cont(c_value,0.25), median(c_value),
        quantile_cont(c_value,0.75), max(c_value),
        sum(CASE WHEN c_value <= 0 THEN 1 ELSE 0 END)
        FROM read_parquet('{OUT}')""").fetchone()
    # What a floor would cost, so the PI can choose one with the trade-off visible.
    grid = {}
    for f in (0, 1, 5, 10, 25, 50, 100, 200, 500, 1000):
        grid[f] = con.execute(
            f"SELECT count(*) FROM read_parquet('{OUT}') WHERE c_value > {f}").fetchone()[0]
    con.close()

    stats = {"candidates": len(rows), "nested_candidates": len(nested_n),
             "c_value_min": qs[0], "q25": qs[1], "median": qs[2], "q75": qs[3],
             "max": qs[4], "n_non_positive": qs[5], "survivors_by_floor": grid}
    Manifest.build(OUT, phase="3", inputs=[SRC], cfg=cfg,
                   params={"formula": "log2(|t|+1)*(f - mean_container_f)",
                           "c_value_min": cfg["vocabulary"]["c_value_min"]},
                   stats=stats).write(OUT)
    return stats


if __name__ == "__main__":
    import json
    print(json.dumps(compute(), indent=2, default=str))
