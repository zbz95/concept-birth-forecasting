"""Phase 2 accept check — spot-list recall against the coinage ledger.

The plan's Phase 2 accept criterion is a global-recall check: ten terms that
must be present as candidates. It deliberately tests the ledger, not vocab_T —
the ledger is the untimed, unfiltered tier.

The coinage dates printed here are also the first real evidence for the Phase 4
spot check, so they are reported alongside.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.manifest import load_config, log_event  # noqa: E402

SPOT = ["transformer", "attention mechanism", "object detection", "bert", "nerf",
        "diffusion model", "capsule network", "semantic segmentation",
        "word embedding", "gan"]

SRC = {1: "rake", 2: "chunk", 4: "ngram"}


def _prov(mask: int) -> str:
    return "+".join(v for k, v in SRC.items() if mask & k) or "-"


def main() -> int:
    cfg = load_config()
    con = duckdb.connect()
    L = "read_parquet('data/interim/ledger.parquet')"
    Y = "read_parquet('data/interim/ledger_yearly.parquet')"

    tot, post = con.execute(f"SELECT count(*), sum(df) FROM {L}").fetchone()
    print(f"ledger: {tot:,} distinct candidates, {post:,} candidate-paper postings\n")

    print(f"{'term':26} {'df':>7} {'coinage':>11} {'first paper':>12}  peak yr  provenance")
    print("-" * 84)
    missing = []
    for term in SPOT:
        row = con.execute(
            f"SELECT df, first_date, first_paper_id, src_mask FROM {L} WHERE cand = ?", [term]
        ).fetchone()
        if row is None:
            missing.append(term)
            print(f"{term:26} {'MISSING':>7}")
            continue
        peak = con.execute(
            f"SELECT year FROM {Y} WHERE cand = ? ORDER BY n_papers DESC LIMIT 1", [term]
        ).fetchone()
        print(f"{term:26} {row[0]:>7,} {str(row[1]):>11} {row[2]:>12}  "
              f"{peak[0] if peak else '-':>7}  {_prov(row[3])}")

    print("\ncandidates by token length:")
    for n, c in con.execute(f"SELECT n_tokens, count(*) FROM {L} GROUP BY 1 ORDER BY 1").fetchall():
        print(f"  {n} token{'s' if n > 1 else ' '}: {c:>12,}")

    print("\nprovenance (extractor agreement — a Phase 3 feature):")
    for m, c in con.execute(f"SELECT src_mask, count(*) FROM {L} GROUP BY 1 ORDER BY 2 DESC").fetchall():
        print(f"  {_prov(m):18} {c:>12,}")

    print(f"\nhow many candidates clear min_total_freq={cfg['vocabulary']['min_total_freq']}?")
    n_pass = con.execute(
        f"SELECT count(*) FROM {L} WHERE df >= {cfg['vocabulary']['min_total_freq']}").fetchone()[0]
    print(f"  {n_pass:,}  (pre-filter, pre-merge — Phase 3 reduces this further)")

    if missing:
        log_event("logs/flags.jsonl", {
            "phase": "2", "kind": "accept_failure",
            "quantity": "spot_list_missing", "value": missing,
            "action": "STOP — extraction recall defect, consult PI",
        })
        print(f"\nACCEPT FAILED — missing: {missing}")
        return 1
    print("\nACCEPT PASSED — all 10 spot-list terms present in the ledger.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
