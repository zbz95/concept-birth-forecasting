"""Phase 3 — the merge map: global identity, evidence-dated membership.

A merge asserts that two surface forms name the same concept. Every merge
carries an *effective date*, and a cluster's count series at year y sums only
those members whose merge is effective by y. This is what keeps merging inside
the causality gate: knowing in 2024 that `gans` and `gan` are the same thing
must not retroactively change the 2016 count series unless the evidence for that
identity existed in 2016.

Three sources, and only three:

  1. Deterministic string variants — timeless. Morphological and orthographic
     identity is a property of English, not of the corpus, so these are active
     at every origin.
  2. Schwartz-Hearst acronym / long-form pairs — effective from the v1 date of
     the first paper that attests the pair. The corpus itself supplies both the
     evidence and its date.
  3. Human-approved merges from the review queue — effective from the date of
     the dated corpus evidence cited in the approval.

Embedding similarity NEVER creates a merge. It may only nominate candidates into
the human queue. A 2025-vintage embedder knows which concepts turned out to be
the same, and letting that write merges would leak the future into every origin.

Nothing is destructive: merges are append-only rows in `logs/merges.jsonl` with
an `undone` column, and cluster membership is always recomputable from the log.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.manifest import Manifest, load_config, log_event  # noqa: E402

OUT = "data/interim/merge_map.parquet"
TIMELESS = date(1900, 1, 1)   # sentinel: active at every origin

# Schwartz-Hearst: "<long form> (<SHORT>)". The short form must look like an
# abbreviation - mostly upper case, 2-10 chars, starting alphanumeric.
_SH = re.compile(r"([A-Za-z][A-Za-z0-9\-\s]{4,80}?)\s*\(\s*([A-Z][A-Za-z0-9\-]{1,9})s?\s*\)")

_BRIT = [
    (re.compile(r"isation\b"), "ization"), (re.compile(r"isating\b"), "izating"),
    (re.compile(r"ised\b"), "ized"), (re.compile(r"ising\b"), "izing"),
    (re.compile(r"iser\b"), "izer"), (re.compile(r"ise\b"), "ize"),
    (re.compile(r"yse\b"), "yze"), (re.compile(r"ysed\b"), "yzed"),
    (re.compile(r"our\b"), "or"), (re.compile(r"ours\b"), "ors"),
    (re.compile(r"centre\b"), "center"), (re.compile(r"metre\b"), "meter"),
    (re.compile(r"modelling\b"), "modeling"), (re.compile(r"modelled\b"), "modeled"),
    (re.compile(r"labelling\b"), "labeling"), (re.compile(r"labelled\b"), "labeled"),
]


def _normal_form(cand: str) -> str:
    """Orthographic normal form used to group deterministic string variants.

    Collapses hyphenation and spacing (`pre-train` / `pre train` / `pretrain`),
    then British-to-American spelling. Deliberately does NOT touch morphology
    beyond what the lemmatizer already did.
    """
    s = cand.replace("-", "").replace(" ", "")
    for rx, rep in _BRIT:
        s = rx.sub(rep, s)
    return s


def _depluralize(tok: str) -> str | None:
    """Residual plural stripping for forms the lemmatizer left alone.

    spaCy's rule lemmatizer does not singularize PROPN, so `GANs` lemmatizes to
    `gans` while `GAN` gives `gan`, and they fork into separate ledger rows.
    """
    if len(tok) < 4:
        return None
    if tok.endswith("ies"):
        return tok[:-3] + "y"
    if tok.endswith(("ses", "xes", "zes", "ches", "shes")):
        return tok[:-2]
    if tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return None


def build_string_variants(cands: set[str]) -> list[tuple[str, str, date, str]]:
    """(member, canonical, effective_date, rule) for deterministic variants."""
    edges = []
    # Group by orthographic normal form; canonical = most frequent surface form,
    # resolved by the caller's df ordering (cands is pre-sorted by df desc).
    by_norm: dict[str, list[str]] = defaultdict(list)
    for c in cands:
        by_norm[_normal_form(c)].append(c)
    for norm, members in by_norm.items():
        if len(members) < 2:
            continue
        canon = members[0]
        for m in members[1:]:
            edges.append((m, canon, TIMELESS, "string_variant/orthographic"))

    # Residual plurals, last token only (the head carries number).
    present = set(cands)
    for c in cands:
        toks = c.split()
        sing = _depluralize(toks[-1])
        if sing is None:
            continue
        target = " ".join(toks[:-1] + [sing])
        if target != c and target in present:
            edges.append((c, target, TIMELESS, "string_variant/plural"))
    return edges


MIN_ACRONYM_LEN = 3
ACRONYM_DOMINANCE = 0.70
ACRONYM_MIN_PAPERS = 3


def build_schwartz_hearst(cfg: dict, cands: set[str]) -> list[tuple[str, str, date, str]]:
    """Acronym / long-form pairs, dated by the first paper attesting each pair.

    An acronym is merged only when it is UNAMBIGUOUS in this corpus. `SR` attests
    to speech recognition, super-resolution, success rate, surface reconstruction
    and spatial reasoning; merging on the shared short form would chain all five
    into one concept through union-find. So each short form's attestations are
    counted per long form, and the pair is kept only if one long form dominates.
    Ambiguous acronyms are not evidence of identity — they go to the human review
    queue instead, exactly like any other contested merge.
    """
    con = duckdb.connect()
    rows = con.execute("""SELECT title || '. ' || abstract, v1_date
                          FROM read_parquet('data/interim/papers.parquet')
                          ORDER BY v1_date""").fetchall()
    con.close()

    first_seen: dict[tuple[str, str], date] = {}
    attest: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for text, d in rows:
        for long_raw, short_raw in _SH.findall(text):
            short = short_raw.lower().rstrip("s")
            if len(short) < MIN_ACRONYM_LEN:
                continue   # two-letter acronyms are hopelessly ambiguous
            long_toks = long_raw.lower().split()
            if not (2 <= len(long_toks) <= 6):
                continue
            long_form = " ".join(long_toks[-len(short):]) if len(long_toks) >= len(short) else None
            # Accept only when the initials of the trailing words spell the short
            # form -- the core Schwartz-Hearst constraint.
            for span in range(len(short), min(len(long_toks), len(short) * 2) + 1):
                cand_long = " ".join(long_toks[-span:])
                initials = "".join(w[0] for w in cand_long.split() if w)
                if initials == short:
                    long_form = cand_long
                    break
            else:
                continue
            if long_form in cands and short in cands and long_form != short:
                key = (short, long_form)
                attest[short][long_form] += 1
                if key not in first_seen or d < first_seen[key]:
                    first_seen[key] = d

    # Ambiguity must be judged on NORMALIZED long forms. `CNN` attests 2236 times
    # to "convolutional neural networks" and 1066 to "convolutional neural
    # network" — one long form, split by number. Counting surface forms would
    # score that 65% and refuse a merge that is not remotely ambiguous.
    def _long_key(lf: str) -> str:
        toks = lf.split()
        return _normal_form(" ".join(toks[:-1] + [_depluralize(toks[-1]) or toks[-1]]))

    out, ambiguous = [], []
    for short, forms in attest.items():
        grouped: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for lf, n in forms.items():
            grouped[_long_key(lf)][lf] += n
        totals = {k: sum(v.values()) for k, v in grouped.items()}
        total = sum(totals.values())
        best_key, n_best = max(totals.items(), key=lambda kv: kv[1])
        if n_best >= ACRONYM_MIN_PAPERS and n_best / total >= ACRONYM_DOMINANCE:
            # Merge onto the most frequently attested surface form of the winner.
            surface = max(grouped[best_key].items(), key=lambda kv: kv[1])[0]
            out.append((short, surface, first_seen[(short, surface)], "schwartz_hearst"))
        else:
            top = sorted(((k, v) for k, v in totals.items()), key=lambda kv: -kv[1])[:5]
            ambiguous.append((short, [(k, v) for k, v in top], total))

    for short, forms, total in sorted(ambiguous, key=lambda x: -x[2])[:300]:
        log_event("logs/merges.jsonl", {
            "phase": "3", "kind": "acronym_ambiguous_not_merged", "short_form": short,
            "attested_long_forms": [{"long": lf, "n_papers": n} for lf, n in forms],
            "total_attestations": total,
            "action": "no merge created; queued for human review (ambiguous acronym)"})

    return out


def build(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    con = duckdb.connect()
    rows = con.execute("""SELECT cand, df, first_date FROM
        read_parquet('data/interim/termhood.parquet') ORDER BY df DESC""").fetchall()
    con.close()
    order = [c for c, _, _ in rows]
    df = {c: d for c, d, _ in rows}
    coinage = {c: fd for c, _, fd in rows}
    cands = set(order)

    edges = build_string_variants(order)
    n_variant = len(edges)
    edges += build_schwartz_hearst(cfg, cands)
    n_sh = len(edges) - n_variant

    # Union-find over the edges, canonical = highest-df member of each cluster.
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if df.get(ra, 0) >= df.get(rb, 0):
            parent[rb] = ra
        else:
            parent[ra] = rb

    for m, c, _, _ in edges:
        union(m, c)

    # A member's effective date is the earliest date at which its own merge edge
    # became active. Timeless rules are active everywhere.
    eff: dict[str, date] = {}
    rule: dict[str, str] = {}
    for m, c, d, r in edges:
        if m not in eff or d < eff[m]:
            eff[m], rule[m] = d, r

    gap_flagged = []
    out = []
    for c in order:
        root = find(c) if c in parent else c
        if root == c:
            out.append((c, c, TIMELESS, "canonical", df[c]))
            continue
        d = eff.get(c, TIMELESS)
        out.append((c, root, d, rule.get(c, "transitive"), df[c]))
        # Merge-gap flag: members whose coinage dates differ by more than
        # merge_gap_flag_years need a human to look at them.
        gap = abs((coinage[c] - coinage[root]).days) / 365.25
        if gap > cfg["merges"]["merge_gap_flag_years"]:
            gap_flagged.append((c, root, round(gap, 1)))

    schema = pa.schema([("cand", pa.string()), ("cluster", pa.string()),
                        ("effective_date", pa.date32()), ("rule", pa.string()),
                        ("df", pa.int32())])
    pq.write_table(pa.Table.from_pylist(
        [dict(zip([f.name for f in schema], r)) for r in out], schema=schema),
        OUT, compression="zstd")

    n_clusters = len({find(c) if c in parent else c for c in order})

    # Safety net. A concept cluster is a handful of surface forms of ONE thing.
    # A large cluster means the union-find chained through an ambiguous bridge,
    # which is the failure mode that merged speech recognition with
    # super-resolution via the shared acronym "sr". Flag, never silently accept.
    sizes: dict[str, int] = defaultdict(int)
    for c in order:
        sizes[find(c) if c in parent else c] += 1
    oversized = sorted(((k, v) for k, v in sizes.items() if v > 6), key=lambda kv: -kv[1])
    for root, n in oversized[:200]:
        members = [c for c in order if (find(c) if c in parent else c) == root][:12]
        log_event("logs/merges.jsonl", {
            "phase": "3", "kind": "oversized_cluster", "cluster": root, "n_members": n,
            "members_sample": members,
            "action": "mandatory human review — likely transitive over-merge"})
    for m, c, g in gap_flagged[:500]:
        log_event("logs/merges.jsonl", {
            "phase": "3", "kind": "merge_gap_flag", "member": m, "cluster": c,
            "coinage_gap_years": g, "threshold": cfg["merges"]["merge_gap_flag_years"],
            "action": "mandatory human review before this merge is trusted"})

    stats = {"terms": len(order), "clusters": n_clusters,
             "merged_terms": len(order) - n_clusters,
             "string_variant_edges": n_variant, "schwartz_hearst_edges": n_sh,
             "merge_gap_flagged": len(gap_flagged),
             "oversized_clusters": len(oversized),
             "largest_cluster": max(sizes.values()) if sizes else 0,
             "embedding_created_merges": 0}
    Manifest.build(OUT, phase="3", inputs=["data/interim/termhood.parquet",
                                          "data/interim/papers.parquet"],
                   cfg=cfg, params={k: cfg["merges"][k] for k in
                                    ("merge_gap_flag_years", "paper_dedupe_within_cluster")},
                   stats=stats).write(OUT)
    return stats


if __name__ == "__main__":
    import json
    print(json.dumps(build(), indent=2, default=str))
