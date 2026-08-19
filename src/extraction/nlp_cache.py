"""Phase 2a — the token cache.

One spaCy pass over the corpus produces the lemma and POS sequence for every
paper. All three extractors and the dictionary re-scan read this cache instead
of re-tokenizing, which is what makes the re-scan affordable: the expensive NLP
happens once, and every later stage is a cheap sequence operation.

Storing POS is not incidental — Phase 3's verb-led and generic-head pattern
kills need it, and re-deriving it later would risk a tagger-version mismatch
against the lemmas the ledger was built from.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import spacy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.extraction.text import paper_text  # noqa: E402
from src.manifest import Manifest, load_config  # noqa: E402

BATCH_ROWS = 20_000

# spaCy splits `few-shot` into few / - / shot and tags the hyphen PUNCT. Left
# alone that makes the hyphen a segment barrier downstream, so `few-shot
# learning`, `zero-shot learning` and `one-shot learning` all collapse to the
# single candidate `shot learning` — three concepts with three different birth
# dates merged into one, permanently, in a ledger the plan forbids splitting.
# 96.2% of the corpus contains an intra-word hyphen, so this is not an edge case.
# Compounds are rejoined here, at the one place that owns tokenization.
_HYPHENS = frozenset("-‐‑‒–—−")


def _merge_hyphenated(doc):
    """Rejoin `A-B(-C)*` written without spaces into a single token.

    Returns (lemmas, pos). The merged lemma joins component lemmas with "-", so
    `pre-trained` -> `pre-train` under the plan's light-lemma rule. The merged
    POS is the head (last) component's, except that a trailing VERB becomes ADJ:
    a hyphenated participle before a noun is adjectival, which keeps
    `self-supervised learning` reachable by the ADJ*(NOUN|PROPN)+ arm.
    """
    lemmas, pos = [], []
    i, n = 0, len(doc)
    while i < n:
        # A compound continues while: token, no trailing space, hyphen, no
        # trailing space, token.
        j = i
        parts = [doc[i]]
        while (j + 2 < n and doc[j].whitespace_ == "" and doc[j + 1].text in _HYPHENS
               and doc[j + 1].whitespace_ == "" and not doc[j + 2].is_punct
               and not doc[j].is_punct):
            parts.append(doc[j + 2])
            j += 2
        if len(parts) > 1:
            lemmas.append("-".join(p.lemma_.lower() for p in parts))
            head = parts[-1].pos_
            pos.append("ADJ" if head == "VERB" else head)
            i = j + 1
        else:
            lemmas.append(doc[i].lemma_.lower())
            pos.append(doc[i].pos_)
            i += 1
    return lemmas, pos

SCHEMA = pa.schema([
    ("id", pa.string()),
    ("v1_date", pa.date32()),
    ("v1_year", pa.int16()),
    ("lemmas", pa.list_(pa.string())),
    ("pos", pa.list_(pa.string())),
    ("n_tokens", pa.int32()),
])


def build_cache(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    model = cfg["candidates"]["spacy_model"]
    n_proc = cfg["runtime"]["n_workers"]
    src = Path("data/interim/papers.parquet")
    out = Path("data/interim/tokens.parquet")

    nlp = spacy.load(model, disable=["parser", "ner", "senter"])
    nlp.max_length = 2_000_000

    pf = pq.ParquetFile(src)
    total = pf.metadata.num_rows
    writer = pq.ParquetWriter(out, SCHEMA, compression="zstd")
    done = 0
    t0 = time.time()
    tok_total = 0

    try:
        for batch in pf.iter_batches(batch_size=BATCH_ROWS,
                                     columns=["id", "title", "abstract", "v1_date", "v1_year"]):
            d = batch.to_pydict()
            texts = [paper_text(t, a) for t, a in zip(d["title"], d["abstract"])]
            lemmas_col, pos_col, ntok = [], [], []
            for doc in nlp.pipe(texts, batch_size=256, n_process=n_proc):
                lem, pos = _merge_hyphenated(doc)
                lemmas_col.append(lem)
                pos_col.append(pos)
                ntok.append(len(lem))
            tok_total += sum(ntok)
            writer.write_table(pa.Table.from_pydict({
                "id": d["id"], "v1_date": d["v1_date"], "v1_year": d["v1_year"],
                "lemmas": lemmas_col, "pos": pos_col, "n_tokens": ntok,
            }, schema=SCHEMA))
            done += len(texts)
            el = time.time() - t0
            print(f"  {done:>7,}/{total:,}  {done/el:>6.0f} docs/s  "
                  f"eta {(total-done)/(done/el)/60:>5.1f} min", flush=True)
    finally:
        writer.close()

    Manifest.build(
        str(out), phase="2a", inputs=[src], cfg=cfg,
        params={"spacy_model": model, "pipes": nlp.pipe_names},
        stats={"papers": done, "tokens": tok_total,
               "mean_tokens_per_paper": round(tok_total / max(done, 1), 1),
               "wall_seconds": round(time.time() - t0, 1)},
    ).write(out)
    return out


if __name__ == "__main__":
    p = build_cache()
    print("wrote", p, f"({p.stat().st_size/1e6:.0f} MB)")
