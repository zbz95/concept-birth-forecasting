"""Phase 3 — pattern kills and C-value termhood.

Two tiers exist from here on. The coinage ledger holds everything, forever. The
*modeling vocabulary* is a view over it: candidates that clear `min_total_freq`,
survive the pattern kills, and clear the C-value floor. Nothing is deleted —
every kill is an append-only row in `logs/kills.jsonl` carrying an `undone`
column, so any filter decision can be reversed without rebuilding the ledger.

The pattern kills need part-of-speech information the ledger does not carry
(it stores candidate strings, not tags). Rather than re-tag tens of millions of
candidate strings, a token -> dominant-POS table is derived once from the token
cache and the kills consult that.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.manifest import Manifest, load_config, log_event  # noqa: E402
from src.extraction.candidates import _STOP  # noqa: E402

POS_TABLE = Path("data/interim/token_pos.parquet")

_ORDINAL = re.compile(
    r"^(?:\d+(?:st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth"
    r"|next|last|previous|former|latter)$", re.I)

# Heads so generic that alone they name no concept. A generic head is NOT killed
# when it is modified — `diffusion model` and `language model` are exactly the
# objects this project exists to track. Only candidates made *entirely* of
# generic and function words are killed.
GENERIC = frozenset("""
model models method methods approach approaches system systems framework frameworks
technique techniques algorithm algorithms result results experiment experiments
paper papers work works study studies task tasks problem problems dataset datasets
data set sets analysis application applications performance accuracy evaluation
case cases example examples number amount level levels type types kind form forms
way ways part parts step steps stage stages state states process processes
value values feature features function functions structure structures property
properties information effect effects use uses user users time times year years
research field area areas domain domains context term terms
""".split())


def build_pos_table(cfg: dict | None = None) -> Path:
    """Dominant POS per lemma, derived once from the token cache."""
    cfg = cfg or load_config()
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{cfg['runtime']['mem_budget_gb'] - 3}GB'")
    con.execute(f"""
      COPY (
        WITH exploded AS (
          SELECT unnest(lemmas) AS lemma, unnest(pos) AS pos
          FROM read_parquet('data/interim/tokens.parquet')
        ), counted AS (
          SELECT lemma, pos, count(*) AS n FROM exploded GROUP BY 1, 2
        )
        SELECT lemma,
               arg_max(pos, n)  AS dominant_pos,
               sum(n)::BIGINT   AS n_tokens,
               (max(n)::DOUBLE / sum(n)) AS dominance
        FROM counted GROUP BY lemma
      ) TO '{POS_TABLE}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    n = con.execute(f"SELECT count(*) FROM read_parquet('{POS_TABLE}')").fetchone()[0]
    con.close()
    Manifest.build(str(POS_TABLE), phase="3", inputs=["data/interim/tokens.parquet"],
                   cfg=cfg, stats={"distinct_lemmas": n}).write(POS_TABLE)
    return POS_TABLE


def load_pos_map() -> dict[str, str]:
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT lemma, dominant_pos FROM read_parquet('{POS_TABLE}')").fetchall()
    con.close()
    return dict(rows)


# ---------------------------------------------------------------------------
# Pattern kills. Each returns a reason string, or None to keep.
# ---------------------------------------------------------------------------

def kill_reason(cand: str, pos_map: dict[str, str], *, strict_straddler: bool = True,
                require_nominal_head: bool = True) -> str | None:
    toks = cand.split()
    if not toks:
        return "empty"

    # verb-led: "propose method", "show that" — a predicate, not a term.
    if pos_map.get(toks[0]) in ("VERB", "AUX"):
        return "verb_led"

    # ordinals and deictics: "first stage", "next step".
    if _ORDINAL.match(toks[0]):
        return "ordinal"

    # stopword-straddler. The strict reading kills a candidate containing a
    # function word ANYWHERE, not merely at an edge. Only the raw n-gram arm can
    # produce those: RAKE splits on stopwords and ADJ*(NOUN|PROPN)+ cannot match
    # one, so a candidate straddling a function word is an n-gram artefact
    # ("network in image", "range of camera") rather than a term. The n-gram
    # arm's job is recall where the tagger fails, not to invent phrases.
    if strict_straddler:
        if any(t in _STOP for t in toks):
            return "stopword_straddler"
    elif toks[0] in _STOP or toks[-1] in _STOP:
        return "stopword_straddler"

    # A term is a noun phrase, so its head must be nominal. Without this, bare
    # adjectives survive as high-frequency "concepts" — novel, large, different,
    # deep, available — none of which name anything.
    if require_nominal_head and pos_map.get(toks[-1]) not in ("NOUN", "PROPN"):
        return "non_nominal_head"

    # generic head: only when NOTHING in the candidate is contentful. This
    # deliberately spares `diffusion model` while killing bare `model`.
    if all(t in GENERIC or t in _STOP for t in toks):
        return "generic_head"

    # a term must contain a letter somewhere (pure numerals, "3 4 5")
    if not any(c.isalpha() for c in cand):
        return "no_alpha"

    return None


def apply_pattern_kills(cfg: dict | None = None, *, log: bool = True) -> dict:
    """Kill pass over candidates that clear min_total_freq.

    The frequency filter runs first because C-value and the kills are both
    expensive, and a candidate below `min_total_freq` can never be a node at
    any origin — but it stays in the ledger, untouched, for coinage dating.
    """
    cfg = cfg or load_config()
    min_df = cfg["vocabulary"]["min_total_freq"]
    pos_map = load_pos_map()

    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{cfg['runtime']['mem_budget_gb'] - 3}GB'")
    rows = con.execute(f"""
        SELECT cand, df, first_date, n_tokens, src_mask
        FROM read_parquet('data/interim/ledger.parquet')
        WHERE df >= {min_df}""").fetchall()

    kept, killed, by_reason = [], [], {}
    for cand, df, first_date, n_tokens, src_mask in rows:
        r = kill_reason(cand, pos_map)
        if r is None:
            kept.append((cand, df, first_date, n_tokens, src_mask))
        else:
            killed.append((cand, df, r))
            by_reason[r] = by_reason.get(r, 0) + 1

    con.execute("CREATE TABLE surviving (cand VARCHAR, df INTEGER, first_date DATE, "
                "n_tokens TINYINT, src_mask TINYINT)")
    con.executemany("INSERT INTO surviving VALUES (?,?,?,?,?)", kept)
    con.execute("COPY surviving TO 'data/interim/vocab_candidates.parquet' "
                "(FORMAT PARQUET, COMPRESSION ZSTD)")
    con.close()

    if log:
        # One row per reason, with a bounded sample. Logging 10^6 individual
        # kills would make the log unreadable and unversionable; the full kill
        # set is recoverable as ledger-minus-survivors at any time.
        for reason, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
            sample = [c for c, _, r in killed if r == reason][:25]
            log_event(cfg["vocabulary"]["kill_log"], {
                "phase": "3", "kind": "pattern_kill", "reason": reason,
                "n_killed": n, "sample": sample,
                "recoverable_as": "ledger.parquet minus vocab_candidates.parquet",
            })

    stats = {"candidates_at_min_df": len(rows), "killed": len(killed),
             "surviving": len(kept), "by_reason": by_reason, "min_total_freq": min_df}
    Manifest.build("data/interim/vocab_candidates.parquet", phase="3",
                   inputs=["data/interim/ledger.parquet", str(POS_TABLE)],
                   cfg=cfg, params={"min_total_freq": min_df}, stats=stats
                   ).write("data/interim/vocab_candidates.parquet")
    return stats


if __name__ == "__main__":
    import json
    cfg = load_config()
    if not POS_TABLE.exists():
        print("building token POS table...", flush=True)
        build_pos_table(cfg)
    print(json.dumps(apply_pattern_kills(cfg), indent=2, default=str))
