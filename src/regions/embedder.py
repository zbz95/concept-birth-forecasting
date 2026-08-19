"""Phase 6 — per-origin embedder.

One word2vec model per origin, trained from scratch on abstracts dated <= T.
No pretrained encoders anywhere, ever: a pretrained model has read the whole
internet including the future of every origin, and using one would make the
embedding features the single largest leak in the pipeline.

Concepts are multi-token, so they are trained as units. Each abstract is
rewritten with its vocabulary concepts joined by underscores before training,
which gives a vector per concept rather than per word, and per-concept vectors
are what features 7 (centroid distance) and 10 (embedding-density influx) need.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq
from gensim.models import Word2Vec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.manifest import Manifest, load_config  # noqa: E402

EDIR = Path("data/graphs/embeddings")


def _sentences(con, T: int) -> list[list[str]]:
    """Abstracts dated <= T as concept-and-word token lists."""
    rows = con.execute(f"""
        SELECT p.paper_id, list(DISTINCT
                 CASE WHEN m.effective_date <= DATE '{T}-12-31'
                      THEN m.cluster ELSE p.term END)
        FROM read_parquet('data/graphs/event_store.parquet') p
        JOIN read_parquet('data/interim/merge_map.parquet') m ON m.cand = p.term
        WHERE p.year <= {T}
        GROUP BY p.paper_id""").fetchall()
    vocab = set(r[0] for r in con.execute(
        f"SELECT cluster FROM read_parquet('data/interim/vocab/vocab_{T}.parquet')").fetchall())
    out = []
    for _, concepts in rows:
        toks = [c.replace(" ", "_") for c in concepts if c in vocab]
        if len(toks) >= 2:
            out.append(toks)
    return out


def build_origin(con, cfg: dict, T: int) -> dict:
    ec = cfg["embedder"]
    assert not ec["pretrained_allowed"], "pretrained encoders are forbidden by the plan"
    sents = _sentences(con, T)
    model = Word2Vec(
        sentences=sents, vector_size=ec["dim"], window=ec["window"],
        min_count=ec["min_count"], epochs=ec["epochs"], seed=ec["seed"],
        workers=cfg["runtime"]["n_workers"], sg=1,
    )
    EDIR.mkdir(parents=True, exist_ok=True)
    out = EDIR / f"emb_{T}.npz"
    keys = list(model.wv.index_to_key)
    vecs = np.vstack([model.wv[k] for k in keys]).astype(np.float32)
    # Unit-normalize once: every downstream use is cosine.
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
    np.savez_compressed(out, keys=np.array([k.replace("_", " ") for k in keys]), vecs=vecs)

    stats = {"origin": T, "sentences": len(sents), "vocab_in_model": len(keys),
             "dim": ec["dim"], "algo": "word2vec-sg", "pretrained": False}
    Manifest.build(str(out), phase="6", as_of=f"{T}-12-31", max_observed_date=f"{T}-12-31",
                   inputs=["data/graphs/event_store.parquet",
                           f"data/interim/vocab/vocab_{T}.parquet"],
                   cfg=cfg, params={k: ec[k] for k in
                                    ("algo", "dim", "window", "min_count", "epochs", "seed",
                                     "pretrained_allowed")},
                   stats=stats).write(out)
    return stats


def load(T: int) -> dict[str, np.ndarray]:
    d = np.load(EDIR / f"emb_{T}.npz", allow_pickle=True)
    return {k: v for k, v in zip(d["keys"], d["vecs"])}


def build_all(cfg: dict | None = None) -> list[dict]:
    cfg = cfg or load_config()
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{cfg['runtime']['mem_budget_gb'] - 4}GB'")
    lo = cfg["evaluation"]["origins_tune"][0]
    hi = cfg["evaluation"]["origins_test"][1]
    out = []
    for T in range(lo, hi + 1):
        s = build_origin(con, cfg, T)
        print(f"  T={T}: {s['sentences']:>7,} abstracts -> {s['vocab_in_model']:>7,} vectors",
              flush=True)
        out.append(s)
    con.close()
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(build_all(), indent=2, default=str))
