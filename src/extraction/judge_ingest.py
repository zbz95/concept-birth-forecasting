"""Phase 3 — ingest external LLM-judge verdicts, and audit them for hindsight.

The judge runs outside the pipeline (PI decision), so its verdicts arrive as
files. This module reattaches them to terms, logs every deletion reversibly, and
then tries to prove the judge cheated.

That last part is the point. The plan forbids the judge from using familiarity,
novelty or importance, because a 2026-vintage model knows which concepts became
famous — and a filter that keeps `diffusion model` because it recognises it,
while dropping an obscure but real 2019 term, has written hindsight into
vocab_2019. Structural precautions (the judge sees only the bare string, batches
are shuffled, no counts or dates are shown) make that harder but cannot make it
impossible, so it is measured.

**The test.** Concepts that crystallized and then DECLINED are real concepts that
did not become famous. Concepts that crystallized and PERSISTED did. A judge
reading linguistic form alone must keep both at the same rate, once they are
matched on how large they were at crystallization. A judge leaning on
familiarity keeps the persisted ones more often. The gap between those two rates
is the hindsight the filter injected.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.manifest import Manifest, load_config, log_event  # noqa: E402

JUDGE = Path("judge")
VERDICTS = JUDGE / "verdicts"
OUT = "data/interim/judge_verdicts.parquet"

_LINE = re.compile(r"^\s*(\d+)\s*[\t,: ]\s*(KEEP|DROP)\s*$", re.I)


def load_index() -> dict[tuple[int, int], str]:
    idx = {}
    for line in open(JUDGE / "index.jsonl"):
        d = json.loads(line)
        idx[(d["batch"], d["line"])] = d["term"]
    return idx


def ingest() -> dict:
    """Read judge/verdicts/batch_NNNN.txt and reattach verdicts to terms."""
    idx = load_index()
    n_batches = 1 + max(b for b, _ in idx)
    verdicts: dict[str, str] = {}
    missing, malformed, seen_batches = [], [], set()

    for f in sorted(VERDICTS.glob("batch_*.txt")):
        b = int(f.stem.split("_")[1])
        seen_batches.add(b)
        for raw in open(f):
            if not raw.strip():
                continue
            m = _LINE.match(raw)
            if not m:
                malformed.append((f.name, raw.strip()[:60]))
                continue
            line_no, verdict = int(m.group(1)), m.group(2).upper()
            term = idx.get((b, line_no))
            if term is None:
                malformed.append((f.name, f"line {line_no} not in index"))
                continue
            verdicts[term] = verdict

    for (b, ln), term in idx.items():
        if b in seen_batches and term not in verdicts:
            missing.append(term)

    con = duckdb.connect()
    con.execute("CREATE TABLE v (term VARCHAR, verdict VARCHAR)")
    con.executemany("INSERT INTO v VALUES (?,?)", list(verdicts.items()))
    con.execute(f"COPY v TO '{OUT}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.close()

    dropped = [t for t, v in verdicts.items() if v == "DROP"]
    stats = {"batches_expected": n_batches, "batches_received": len(seen_batches),
             "terms_judged": len(verdicts), "terms_total": len(idx),
             "dropped": len(dropped), "kept": len(verdicts) - len(dropped),
             "unjudged_in_received_batches": len(missing), "malformed_lines": len(malformed)}

    # Deletions are logged in reason-grouped rows with a bounded sample; the full
    # set is always recoverable as the parquet minus the kept terms.
    log_event("logs/kills.jsonl", {
        "phase": "3", "kind": "llm_judge_deletion", "n_killed": len(dropped),
        "sample": dropped[:50], "criteria": "content-free only (names vs describes)",
        "forbidden_criteria_declared": ["familiarity", "novelty", "importance"],
        "prompt_file": "judge/PROMPT.md",
        "recoverable_as": "judge_verdicts.parquet WHERE verdict='KEEP'"})
    if malformed:
        log_event("logs/flags.jsonl", {"phase": "3", "kind": "judge_parse_problem",
                                       "n_malformed": len(malformed),
                                       "sample": malformed[:20]})
    Manifest.build(OUT, phase="3", inputs=[str(JUDGE / "index.jsonl")],
                   params={"prompt": "judge/PROMPT.md"}, stats=stats).write(OUT)
    return stats


def audit_hindsight() -> dict:
    """Did the judge use familiarity? Declined vs persisted, matched on size.

    Both groups are real, crystallized concepts. Only one became well known. A
    form-blind judge keeps them at equal rates within a size stratum.
    """
    con = duckdb.connect()
    rows = con.execute(f"""
        SELECT b.concept, b.fate, b.total_papers, v.verdict
        FROM read_parquet('data/registry/births.parquet') b
        JOIN read_parquet('{OUT}') v ON v.term = b.concept
        WHERE b.crystallization_year BETWEEN 2014 AND 2022
          AND b.fate IN ('persisted', 'crystallized_then_declined')""").fetchall()
    con.close()
    if not rows:
        return {"error": "no overlap between verdicts and registry"}

    strata = [(9, 25), (26, 60), (61, 150), (151, 500), (501, 10 ** 9)]
    out, keeps = {}, defaultdict(lambda: [0, 0])
    for concept, fate, total, verdict in rows:
        for lo, hi in strata:
            if lo <= total <= hi:
                k = (f"{lo}-{hi if hi < 10 ** 8 else '+'}", fate)
                keeps[k][0] += 1
                keeps[k][1] += (verdict == "KEEP")
                break
    for (stratum, fate), (n, k) in sorted(keeps.items()):
        out.setdefault(stratum, {})[fate] = {"n": n, "keep_rate": round(k / n, 3) if n else None}

    gaps = []
    for stratum, d in out.items():
        p = d.get("persisted", {}).get("keep_rate")
        dec = d.get("crystallized_then_declined", {}).get("keep_rate")
        if p is not None and dec is not None:
            gaps.append(p - dec)
    verdict = ("no evidence of hindsight" if gaps and max(gaps) < 0.05 else
               "POSSIBLE HINDSIGHT — judge keeps famous concepts more" if gaps else
               "insufficient data")
    res = {"by_stratum": out, "max_keep_rate_gap": round(max(gaps), 3) if gaps else None,
           "interpretation": verdict}
    if gaps and max(gaps) >= 0.05:
        log_event("logs/flags.jsonl", {
            "phase": "3", "kind": "judge_hindsight_suspected",
            "max_keep_rate_gap": round(max(gaps), 3), "by_stratum": out,
            "action": "STOP — the judge appears to use familiarity, which the plan forbids"})
    return res


if __name__ == "__main__":
    if not VERDICTS.exists() or not any(VERDICTS.glob("batch_*.txt")):
        print(f"no verdict files found in {VERDICTS}/ — expected batch_NNNN.txt")
        raise SystemExit(1)
    s = ingest()
    print(json.dumps(s, indent=2))
    print("\n--- hindsight audit ---")
    print(json.dumps(audit_hindsight(), indent=2))
