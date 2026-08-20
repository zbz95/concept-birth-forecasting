# Working in this repository

Guidance for anyone — human or agent — picking this up. Read `report.md` for what
the project found; this file is about how it is built and what will break if you
are careless.

## The one rule that matters

**No artifact indexed by year *T* may contain information dated after *T*.**

This is not a stylistic preference. The entire question — can you predict where a
concept will appear before it exists — is void if any part of the pipeline has
seen the answer. Violations are subtle and do not announce themselves.

The rule is enforced in code, not by convention. `src/manifest.py` provides
`assert_causal(artifact, as_of, max_observed_date)`, which every stage producing a
time-indexed artifact calls at build time. If you add a stage, call it.

Concretely, this means:

- Vocabulary, merge activation, embeddings, region membership, and every feature
  are **rebuilt from scratch at each origin year**. A global filter applied to all
  years is a leak, even when it looks innocent — computing C-value over the whole
  corpus admits a term into `vocab_2019` partly because it became frequent in 2023.
- **No pretrained encoder, anywhere.** `configs/config.yaml` sets
  `embedder.pretrained_allowed: false` and the embedder asserts on it. A
  pretrained model has read the future of every origin.
- **Merges carry evidence dates.** Knowing in 2024 that two surface forms denote
  one concept must not retroactively alter a 2016 count series. Embedding
  similarity may *nominate* merge candidates but is forbidden from *creating*
  merges (`merges.sources.embedding_similarity.creates_merges: false`).
- **Birth profiles are outcome data.** They may be used for attachment,
  evaluation, and the confirmed-birth back-fill. They may not become model
  features. This distinction is written into `PLAN.md`'s leakage checklist as
  item 9.

## Conventions

**Every threshold lives in `configs/config.yaml`**, with a comment recording why
it holds its value and, where it was changed, what evidence changed it. Changing a
value changes the config hash, which invalidates every downstream manifest. Do not
hard-code a threshold in a module.

**Every artifact gets a manifest.** `<path>.manifest.json` beside it, carrying the
config hash, code commit, hashes of every input, and the stage's own statistics.
This is what makes a result auditable six months later.

**Logs are append-only with an undo column.** `logs/kills.jsonl`,
`logs/merges.jsonl`, `logs/flags.jsonl`. Nothing is deleted; a reversal is a new
row referencing the original's `event_id`. `logs/flags.jsonl` is the decision
record — every threshold change, out-of-band measurement, gate outcome and one
process error is in there, written when it happened rather than reconstructed
afterwards.

**Failures are kept.** The discarded unit-rate model and its report remain in
`reports/phase8_models.md`. A documented negative result is part of the finding.

## Pipeline order

Stages read each other's artifacts, so order matters:

```
extraction/ingest.py       corpus -> papers.parquet
extraction/nlp_cache.py    spaCy lemma/POS cache (slowest stage, ~30 min)
extraction/candidates.py   coinage ledger, in three stages: a | b | agg
extraction/filters.py      pattern kills + POS table
extraction/termhood.py     C-value
extraction/merges.py       merge map (evidence-dated)
extraction/postings.py     term-level postings
extraction/vocabulary.py   per-origin vocab_T
extraction/registry.py     birth registry
graph/build.py             event store + per-origin projections
regions/embedder.py        per-origin word2vec
regions/cpm.py             clique percolation + lineage
regions/features.py        region features
models/attachment.py       births -> regions, and the C2 gate
models/backfill.py         confirmed-birth history (needs attachment first)
models/layer2_split.py     localisation under the three-way split
```

`data/` is gitignored and fully regenerable. Nothing under it is precious.

## Things that will bite you

These were all discovered the hard way; each cost real time.

**spaCy splits hyphenated compounds.** `few-shot` becomes `few` / `-` / `shot`,
and the hyphen tags as PUNCT. Left alone, `few-shot learning`, `zero-shot
learning` and `one-shot learning` collapse into one concept. `nlp_cache.py`
rejoins them; do not remove that.

**The noun-chunk pattern is `ADJ*(NOUN|PROPN)+`, not `ADJ*NOUN+`.** spaCy tags
`BERT` and `NeRF` as PROPN. A NOUN-only reading silently drops them.

**Pure lift ranking floods profiles with hapax terms.** At `min_total_freq = 9`, a
concept appearing once among twenty papers scores an identical lift to hundreds of
others, so the top-k becomes an arbitrary draw from the rarest terms in the
corpus. `profile_min_count: 3` is an eligibility floor, not a ranking change, and
removing it degrades attachment badly.

**Clique percolation degenerates on dense cores.** Generic head nouns are hubs
(`image` was adjacent to 56% of the 2014 graph). Without the lift filter
(`edge_min_lift`), percolation returns 2.4 million maximal cliques on a
1,571-node graph and does not finish. The degeneracy guard escalates
geometrically; a linear ladder stalls by 2021.

**Acronyms are ambiguous.** `SR` attests to speech recognition, super-resolution,
success rate and surface reconstruction. Without the dominance test, union-find
chains all four into one cluster. Ambiguity must be judged over *normalised* long
forms, or singular/plural variants split the vote and reject valid merges (`CNN`
scored 65% before normalisation).

**Floating-point invariants need relative tolerance.** The edge-mass invariant
sums millions of `1/C(k,2)` terms; at *n* ≈ 10⁵ the drift is ~10⁻⁵ absolute. An
absolute `1e-6` check produces false failures. This mistake was made twice.

**Fitting can lose to sorting.** On this data a raw sort by a single feature
repeatedly beat gradient boosting over the same feature. With ~56 validation
regions there is not enough signal to fit much. Check the trivial baseline before
concluding a model helps.

## Evaluation discipline

Partitions are declared in `configs/config.yaml` under `localization`:

```
train      2016-2018   fit coefficients
validation 2019-2021   ALL model decisions happen here
test       2022-2023   scored once, never iterated on
```

If you change the model, re-select on validation. Do not re-score test to see
whether the change helped — that is the failure this partition exists to prevent,
and it already happened once in this project (recorded in `logs/flags.jsonl` as
`process_error`).

**A specification is declared and must not be quietly amended.** `report.md` §5.4
states what is to be graded prospectively on origin 2024 once 2026 papers exist:
*k*=4, the own-series feature family, within-region normalisation, deterministic
top-degree as the baseline, recall@*K* as the metric, with *K*=20 indicated by the
validation sweep. Amending that after seeing more data destroys its value.

## Known defects, documented not hidden

`reports/datasheet.md` is the authoritative list. The most consequential:

**The corpus violates the causality rule and this is unfixed.** The arXiv snapshot
stores only current abstracts, so 19.1% of papers carry text finalised in a later
year than their first-version date. This biases coinage dates *earlier* — the
direction that flatters a forecaster. It was measured, escalated, and accepted as
a documented limitation rather than corrected. Any result narrative must say so;
`PLAN.md`'s Phase 9 criterion that "the leakage checklist passes on every
artifact" cannot honestly be signed off.

Do not remove this from the datasheet to make the results look cleaner.
