"""Phase 1 — Corpus ingest.

Streams the arXiv Kaggle snapshot straight out of the zip, keeps papers with a
primary or cross-listed category in {cs.CL, cs.CV}, and writes a Parquet table.
The 4.5 GB JSON is never materialized on disk and never held in memory; every
later phase reads the ~10^5-row Parquet instead.

Date is the v1 submission date (`versions[0].created`) only. `update_date` is
post-hoc information and is deliberately not carried forward as a date field —
using it anywhere would breach the causality gate.
"""

from __future__ import annotations

import re
import sys
import zipfile
from collections import Counter
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.manifest import Manifest, load_config, log_event  # noqa: E402

CHUNK_ROWS = 50_000

SCHEMA = pa.schema([
    ("id", pa.string()),
    ("title", pa.string()),
    ("abstract", pa.string()),
    ("categories", pa.string()),
    ("primary_category", pa.string()),
    ("is_cs_cl", pa.bool_()),
    ("is_cs_cv", pa.bool_()),
    ("authors_parsed", pa.list_(pa.string())),
    ("n_authors", pa.int32()),
    ("v1_date", pa.date32()),
    ("v1_year", pa.int16()),
    ("v1_quarter", pa.int8()),
    ("n_versions", pa.int16()),
])

# Withdrawal shows up in the comments field far more reliably than in the
# abstract; both are checked. Kept deliberately narrow — "withdraw" appearing
# mid-abstract in a topical sense must not kill a real paper.
_WITHDRAWN_COMMENT = re.compile(r"\bwithdraw", re.I)
_WITHDRAWN_ABSTRACT = re.compile(
    r"^\s*(this\s+(paper|submission|manuscript|article|entry)\s+has\s+been\s+withdrawn"
    r"|withdrawn\s+by\s+the\s+author)", re.I,
)


def _v1_date(rec: dict) -> date | None:
    versions = rec.get("versions") or []
    if not versions:
        return None
    created = versions[0].get("created")
    if not created:
        return None
    try:
        return parsedate_to_datetime(created).date()
    except (TypeError, ValueError):
        return None


def _author_names(rec: dict) -> list[str]:
    """Exact parsed-name keys for Phase 4 author-group components.

    `authors_parsed` rows are [last, first, suffix]. The key is the joined,
    whitespace-normalized, lowercased triple — exact match, per the plan.
    Collisions merge groups, which makes crystallization *harder*; that bias
    points away from false births and is accepted.
    """
    out = []
    for parts in rec.get("authors_parsed") or []:
        key = " ".join(" ".join(str(p).split()) for p in parts if p).strip().lower()
        if key:
            out.append(key)
    return out


def ingest(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    ccfg = cfg["corpus"]
    zip_path = Path("data/raw") / f"arxiv-snapshot-v{ccfg['kaggle_version']}.zip"
    out_path = Path("data/interim/papers.parquet")
    cutoff = date.fromisoformat(ccfg["corpus_cutoff"])
    wanted = set(ccfg["categories"])

    counts = Counter()
    per_year: Counter = Counter()
    seen_ids: set[str] = set()
    buf: list[dict] = []
    max_date: date | None = None
    writer = pq.ParquetWriter(out_path, SCHEMA, compression="zstd")

    def flush():
        if not buf:
            return
        writer.write_table(pa.Table.from_pylist(buf, schema=SCHEMA))
        buf.clear()

    try:
        with zipfile.ZipFile(zip_path) as z, z.open(ccfg["snapshot_file"]) as fh:
            for raw in fh:
                counts["scanned"] += 1
                if not raw.strip():
                    continue
                rec = orjson.loads(raw)

                cats = (rec.get("categories") or "").split()
                if not wanted.intersection(cats):
                    continue
                counts["category_match"] += 1

                pid = rec.get("id")
                if not pid or pid in seen_ids:
                    counts["dropped_duplicate"] += 1
                    continue

                if ccfg["drop_withdrawn"]:
                    comment = rec.get("comments") or ""
                    abstract_raw = rec.get("abstract") or ""
                    if _WITHDRAWN_COMMENT.search(comment) or _WITHDRAWN_ABSTRACT.search(abstract_raw):
                        counts["dropped_withdrawn"] += 1
                        continue

                d = _v1_date(rec)
                if d is None:
                    counts["dropped_no_v1_date"] += 1
                    continue
                if d > cutoff:
                    counts["dropped_after_cutoff"] += 1
                    continue

                title = " ".join((rec.get("title") or "").split())
                abstract = " ".join((rec.get("abstract") or "").split())
                if not abstract:
                    counts["dropped_empty_abstract"] += 1
                    continue

                authors = _author_names(rec)
                seen_ids.add(pid)
                max_date = d if max_date is None or d > max_date else max_date
                per_year[d.year] += 1
                counts["kept"] += 1

                buf.append({
                    "id": pid,
                    "title": title,
                    "abstract": abstract,
                    "categories": rec.get("categories") or "",
                    "primary_category": cats[0] if cats else "",
                    "is_cs_cl": "cs.CL" in cats,
                    "is_cs_cv": "cs.CV" in cats,
                    "authors_parsed": authors,
                    "n_authors": len(authors),
                    "v1_date": d,
                    "v1_year": d.year,
                    "v1_quarter": (d.month - 1) // 3 + 1,
                    "n_versions": len(rec.get("versions") or []),
                })
                if len(buf) >= CHUNK_ROWS:
                    flush()
            flush()
    finally:
        writer.close()

    stats = {
        **dict(counts),
        "papers_per_year": dict(sorted(per_year.items())),
        "max_v1_date": max_date.isoformat() if max_date else None,
    }
    # Causality gate: the corpus artifact is indexed by corpus_cutoff and must
    # not contain a paper dated after it.
    Manifest.build(
        "data/interim/papers.parquet", phase="1",
        inputs=[zip_path], cfg=cfg,
        params={k: ccfg[k] for k in ("categories", "corpus_cutoff", "date_field",
                                     "kaggle_ref", "kaggle_version")},
        stats=stats, as_of=cutoff, max_observed_date=max_date,
    ).write(out_path)

    band = cfg["expected_bands"]["phase1_papers_kept"]
    if not (band[0] <= counts["kept"] <= band[1]):
        log_event("logs/flags.jsonl", {
            "phase": "1", "kind": "out_of_band",
            "quantity": "papers_kept", "value": counts["kept"], "band": band,
            "action": "STOP — consult PI (Principle 6: never tune toward the band)",
        })
    return out_path


if __name__ == "__main__":
    p = ingest()
    import json
    m = json.loads(Path(str(p) + ".manifest.json").read_text())
    print(json.dumps(m["stats"], indent=2))
