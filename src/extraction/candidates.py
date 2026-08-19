"""Phase 2b — candidate generation and the permanent coinage ledger.

Three extractors nominate candidates from the token cache: RAKE, POS-pattern
noun chunks, and raw 2-3-grams. Their union is provenance-tagged (extractor
agreement becomes a Phase 3 feature).

Counting is by DICTIONARY RE-SCAN, per the PI decision of 2026-08-19: extractors
decide *what is a candidate*, never *where it occurs*. A term nominated in one
paper is then counted in every paper containing it, so crystallization years do
not inherit extractor recall variance.

The re-scan is exact but cheap, because of one observation: the n-gram extractor
already enumerates every 2-3-gram in every paper, so for lengths 2-3 extraction
*is* the re-scan. Only lengths 1 and 4-5 need a second pass against a candidate
set, and those sets are small enough to hold in memory. The two passes emit
disjoint length ranges, so nothing is double-counted.

This ledger is never filtered, capped, or deleted (Phase 2 of the plan). Phase 4
reads coinage dates from it, including for terms that only much later clear
`min_total_freq`.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from spacy.lang.en.stop_words import STOP_WORDS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.manifest import Manifest, load_config  # noqa: E402

SRC_RAKE, SRC_CHUNK, SRC_NGRAM = 1, 2, 4

# Segments never span these: an n-gram must not cross a punctuation boundary.
BARRIER_POS = frozenset({"PUNCT", "SPACE", "SYM", "X"})
# ADJ*(NOUN|PROPN)+ — PROPN is included deliberately. spaCy tags BERT and NeRF
# as PROPN, and a literal NOUN-only reading would drop them from this arm.
CHUNK_HEAD = frozenset({"NOUN", "PROPN"})
CHUNK_MOD = frozenset({"ADJ"})

BATCH_ROWS = 8_000
# Flush the accumulator on size, not on batch count. A batch of abstracts yields
# roughly 300 distinct 2-3-grams each; left unbounded the dict outgrows the
# machine long before the batch loop would flush it.
ACC_FLUSH_ROWS = 1_500_000
_STOP = frozenset(STOP_WORDS) | {"-", "--"}

PART_SCHEMA = pa.schema([
    ("cand", pa.string()), ("year", pa.int16()), ("n_papers", pa.int32()),
    ("min_date", pa.date32()), ("min_paper", pa.string()), ("src_mask", pa.int8()),
])
NOM_SCHEMA = pa.schema([("cand", pa.string()), ("src_mask", pa.int8())])

_CFG: dict = {}


def _init_worker(max_len: int):
    global _MAXLEN
    _MAXLEN = max_len


def _is_punct_only(lem: str) -> bool:
    return not any(c.isalnum() for c in lem)


def _segments(lemmas: list[str], pos: list[str]) -> list[list[str]]:
    """Runs of non-barrier tokens. N-grams are enumerated within these only.

    A punctuation-only lemma is a barrier regardless of its tag: spaCy tags a
    bare "-" as ADJ/NOUN/PROPN/VERB about 16% of the time, and BARRIER_POS alone
    would let those through into the permanent ledger.
    """
    segs, cur = [], []
    for lem, p in zip(lemmas, pos):
        if p in BARRIER_POS or not lem or _is_punct_only(lem):
            if len(cur) > 0:
                segs.append(cur)
                cur = []
        else:
            cur.append(lem)
    if cur:
        segs.append(cur)
    return segs


def _chunk_spans(lemmas: list[str], pos: list[str], max_len: int) -> list[str]:
    """ADJ*(NOUN|PROPN)+ maximal spans, plus every suffix of each span.

    Suffixes rather than all sub-spans: a suffix keeps the head noun, so
    "deep convolutional neural network" yields "convolutional neural network",
    "neural network", "network" — the linguistically real nested terms, and the
    source of the unigram candidates the spot-list depends on.
    """
    out, i, n = [], 0, len(lemmas)
    while i < n:
        if _is_punct_only(lemmas[i]):
            i += 1
            continue
        if pos[i] in CHUNK_MOD or pos[i] in CHUNK_HEAD:
            j = i
            while j < n and pos[j] in CHUNK_MOD:
                j += 1
            k = j
            while k < n and pos[k] in CHUNK_HEAD:
                k += 1
            if k > j:  # at least one head noun -> a real chunk
                span = lemmas[i:k]
                for s in range(len(span)):
                    sub = span[s:]
                    if 1 <= len(sub) <= max_len:
                        out.append(" ".join(sub))
                i = k
                continue
            i = max(j, i + 1)
        else:
            i += 1
    return out


def _rake_phrases(segs: list[list[str]], max_len: int) -> list[str]:
    """RAKE: content-word runs delimited by stopwords and punctuation."""
    out = []
    for seg in segs:
        run = []
        for lem in seg:
            if lem in _STOP:
                if 1 <= len(run) <= max_len:
                    out.append(" ".join(run))
                run = []
            else:
                run.append(lem)
        if 1 <= len(run) <= max_len:
            out.append(" ".join(run))
    return out


def _has_alpha(s: str) -> bool:
    return any(c.isalpha() for c in s)


def _pass_a(args):
    """Extract. Returns (postings for len 2-3, nominations for len 1 and 4+)."""
    lemmas, pos, year, date, pid = args
    segs = _segments(lemmas, pos)

    ngrams = set()
    for seg in segs:
        L = len(seg)
        for n in (2, 3):
            for s in range(L - n + 1):
                ngrams.add(" ".join(seg[s:s + n]))

    nominated = defaultdict(int)
    for ph in _rake_phrases(segs, _MAXLEN):
        nominated[ph] |= SRC_RAKE
    for ph in _chunk_spans(lemmas, pos, _MAXLEN):
        nominated[ph] |= SRC_CHUNK

    # Length 2-3: extraction is already the complete re-scan.
    postings = {}
    for g in ngrams:
        if _has_alpha(g):
            postings[g] = SRC_NGRAM | nominated.get(g, 0)
    # Length 1 and 4+: nominations only. Pass B counts them exhaustively.
    noms = {c: m for c, m in nominated.items()
            if _has_alpha(c) and (c.count(" ") == 0 or c.count(" ") >= 3)}
    return (year, date, pid, postings, noms)


def _pass_b(args):
    """Re-scan for lengths 1 and 4-5 against the nominated candidate sets."""
    lemmas, pos, year, date, pid = args
    segs = _segments(lemmas, pos)
    hits = set()
    for seg in segs:
        L = len(seg)
        for tok in seg:
            if tok in _UNI:
                hits.add(tok)
        for n in range(4, _MAXLEN + 1):
            for s in range(L - n + 1):
                g = " ".join(seg[s:s + n])
                if g in _LONG:
                    hits.add(g)
    return (year, date, pid, hits)


def _set_scan_sets(max_len: int, uni: set, long_: set):
    """Install the re-scan sets as module globals in the PARENT process.

    Pool workers are forked on Linux, so they inherit these by copy-on-write.
    Passing them through `initargs` instead would pickle a full copy into every
    worker — with a multi-million-entry long-candidate set that is several GB of
    duplicated memory on a 14 GB machine.
    """
    global _MAXLEN, _UNI, _LONG
    _MAXLEN, _UNI, _LONG = max_len, uni, long_


# ---------------------------------------------------------------------------

def _flush_postings(acc, writer):
    if not acc:
        return
    cols = {"cand": [], "year": [], "n_papers": [], "min_date": [],
            "min_paper": [], "src_mask": []}
    for (cand, year), (n, mind, minp, mask) in acc.items():
        cols["cand"].append(cand); cols["year"].append(year)
        cols["n_papers"].append(n); cols["min_date"].append(mind)
        cols["min_paper"].append(minp); cols["src_mask"].append(mask)
    writer.write_table(pa.Table.from_pydict(cols, schema=PART_SCHEMA))
    acc.clear()


def _accumulate(acc, cand_masks, year, date, pid):
    """One paper contributes at most 1 to any candidate's document frequency."""
    for cand, mask in cand_masks:
        key = (cand, year)
        cur = acc.get(key)
        if cur is None:
            acc[key] = [1, date, pid, mask]
        else:
            cur[0] += 1
            cur[3] |= mask
            if date < cur[1]:
                cur[1], cur[2] = date, pid


WORKDIR = Path("data/interim/phase2")
TOK = Path("data/interim/tokens.parquet")
PA_PATH = WORKDIR / "postings_a.parquet"
PB_PATH = WORKDIR / "postings_b.parquet"
NOM_PATH = WORKDIR / "nominations.parquet"


def stage_a(cfg: dict) -> dict:
    """Extract. Emits complete postings for lengths 2-3 and nominations elsewhere."""
    max_len = cfg["candidates"]["max_candidate_tokens"]
    n_proc = cfg["runtime"]["n_workers"]
    WORKDIR.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(TOK)
    total = pf.metadata.num_rows

    print(f"pass A: extract over {total:,} papers", flush=True)
    wa = pq.ParquetWriter(PA_PATH, PART_SCHEMA, compression="zstd")
    wn = pq.ParquetWriter(NOM_PATH, NOM_SCHEMA, compression="zstd")
    acc, nom_seen, done = {}, {}, 0
    with Pool(n_proc, initializer=_init_worker, initargs=(max_len,)) as pool:
        for batch in pf.iter_batches(batch_size=BATCH_ROWS):
            d = batch.to_pydict()
            work = list(zip(d["lemmas"], d["pos"], d["v1_year"], d["v1_date"], d["id"]))
            for year, date, pid, postings, noms in pool.imap_unordered(_pass_a, work, chunksize=100):
                _accumulate(acc, postings.items(), year, date, pid)
                for c, m in noms.items():
                    nom_seen[c] = nom_seen.get(c, 0) | m
            done += len(work)
            if len(acc) >= ACC_FLUSH_ROWS:
                _flush_postings(acc, wa)
            print(f"  A {done:>7,}/{total:,}  acc {len(acc):>9,}  noms {len(nom_seen):>9,}", flush=True)
    _flush_postings(acc, wa)
    wa.close()
    wn.write_table(pa.Table.from_pydict(
        {"cand": list(nom_seen.keys()), "src_mask": list(nom_seen.values())}, schema=NOM_SCHEMA))
    wn.close()

    uni = sum(1 for c in nom_seen if " " not in c)
    lng = sum(1 for c in nom_seen if c.count(" ") >= 3)
    print(f"pass A done: {uni:,} unigram + {lng:,} long candidates", flush=True)
    return {"unigram_nominations": uni, "long_nominations": lng, "papers": done}


def stage_b(cfg: dict) -> dict:
    """Dictionary re-scan for lengths 1 and 4+, against pass A's nominations."""
    max_len = cfg["candidates"]["max_candidate_tokens"]
    tbl = pq.read_table(NOM_PATH, columns=["cand"])["cand"].to_pylist()
    uni = {c for c in tbl if " " not in c}
    long_ = {c for c in tbl if c.count(" ") >= 3}
    del tbl
    print(f"re-scan sets: {len(uni):,} unigram, {len(long_):,} long", flush=True)

    # Workers fork after these are installed, so the sets are shared, not copied.
    _set_scan_sets(max_len, uni, long_)
    n_proc = cfg["runtime"]["n_workers"]

    pf = pq.ParquetFile(TOK)
    total = pf.metadata.num_rows
    print("pass B: dictionary re-scan (lengths 1 and 4+)", flush=True)
    wb = pq.ParquetWriter(PB_PATH, PART_SCHEMA, compression="zstd")
    acc, done = {}, 0
    with Pool(n_proc) as pool:
        for batch in pf.iter_batches(batch_size=BATCH_ROWS):
            d = batch.to_pydict()
            work = list(zip(d["lemmas"], d["pos"], d["v1_year"], d["v1_date"], d["id"]))
            for year, date, pid, hits in pool.imap_unordered(_pass_b, work, chunksize=100):
                _accumulate(acc, ((h, 0) for h in hits), year, date, pid)
            done += len(work)
            if len(acc) >= ACC_FLUSH_ROWS:
                _flush_postings(acc, wb)
            print(f"  B {done:>7,}/{total:,}  acc {len(acc):>9,}", flush=True)
    _flush_postings(acc, wb)
    wb.close()
    return {"papers": done}


def stage_agg(cfg: dict) -> dict:
    tok, workdir = TOK, WORKDIR
    pa_path, pb_path, nom_path = PA_PATH, PB_PATH, NOM_PATH
    print("aggregating ledger", flush=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{cfg['runtime']['mem_budget_gb'] - 3}GB'")
    con.execute(f"PRAGMA temp_directory='{workdir / 'duckdb_tmp'}'")
    con.execute(f"""
      CREATE VIEW parts AS
        SELECT * FROM read_parquet('{pa_path}')
        UNION ALL SELECT * FROM read_parquet('{pb_path}');
      CREATE VIEW noms AS SELECT * FROM read_parquet('{nom_path}');
    """)
    con.execute(f"""
      COPY (
        SELECT cand, year, sum(n_papers)::INTEGER AS n_papers
        FROM parts GROUP BY 1, 2 ORDER BY 1, 2
      ) TO 'data/interim/ledger_yearly.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)
    con.execute(f"""
      COPY (
        SELECT p.cand,
               sum(p.n_papers)::INTEGER                     AS df,
               min(p.min_date)                              AS first_date,
               arg_min(p.min_paper, p.min_date)             AS first_paper_id,
               (length(p.cand) - length(replace(p.cand,' ','')) + 1)::TINYINT AS n_tokens,
               (bit_or(p.src_mask) | coalesce(max(n.src_mask), 0))::TINYINT   AS src_mask
        FROM parts p LEFT JOIN noms n USING (cand)
        GROUP BY p.cand ORDER BY df DESC
      ) TO 'data/interim/ledger.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)
    stats = con.execute("""
      SELECT count(*), sum(df), min(first_date), max(first_date)
      FROM read_parquet('data/interim/ledger.parquet')""").fetchone()
    con_rows = con.execute(
        "SELECT count(*) FROM read_parquet('data/interim/ledger_yearly.parquet')").fetchone()[0]
    by_len = con.execute("""
      SELECT n_tokens, count(*) FROM read_parquet('data/interim/ledger.parquet')
      GROUP BY 1 ORDER BY 1""").fetchall()
    con.close()

    out = {"distinct_candidates": stats[0], "total_postings": stats[1],
           "first_date": str(stats[2]), "last_first_date": str(stats[3]),
           "candidates_by_token_length": {int(k): int(v) for k, v in by_len}}
    yearly_stats = {"rows": int(con_rows), "distinct_candidates": stats[0]}
    Manifest.build("data/interim/ledger_yearly.parquet", phase="2b",
                   inputs=[tok, pa_path, pb_path], cfg=cfg,
                   params={"postings_mode": cfg["candidates"]["postings_mode"]},
                   stats=yearly_stats).write("data/interim/ledger_yearly.parquet")
    Manifest.build("data/interim/ledger.parquet", phase="2b",
                   inputs=[tok, pa_path, pb_path, nom_path], cfg=cfg,
                   params={k: cfg["candidates"][k] for k in
                           ("extractors", "ngram_range", "postings_mode",
                            "max_candidate_tokens", "normalization")},
                   stats=out).write("data/interim/ledger.parquet")
    return out


if __name__ == "__main__":
    import json
    cfg = load_config()
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = {}
    if stage in ("a", "all"):
        res["stage_a"] = stage_a(cfg)
    if stage in ("b", "all"):
        res["stage_b"] = stage_b(cfg)
    if stage in ("agg", "all"):
        res["ledger"] = stage_agg(cfg)
    print(json.dumps(res, indent=2, default=str))
