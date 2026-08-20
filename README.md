# Concept-Birth Forecasting

Where do new scientific concepts appear? This project builds a concept
co-occurrence graph from 269,814 arXiv abstracts in NLP and computer vision
(1994–2025), partitions it into overlapping regions, and asks a concrete
forecasting question:

> At origin year *T*, rank the members of a region by their estimated propensity to
> be a parent of a concept born there in a later window, select the top *K*, and
> evaluate the selection against the parents actually observed.

*K* is a parameter of the evaluation, not of the method; it is swept over
{1, 2, 3, 5, 10, 20, 50}, with lift over random maximised at *K*=20.

**Full report: [`report.md`](report.md)**

## Headline result

Held-out test years (2022–2023), *K*=10 (the value fixed before the sweep):

| horizon | random | degree baseline | model | lift | 95% CI |
|---:|---:|---:|---:|---:|---|
| 1 | 0.144 | 0.196 | **0.250** | 1.73× | [1.46, 2.07] |
| 2 | 0.152 | 0.213 | **0.321** | 2.10× | [1.48, 2.83] |

Lift increases with region size, reaching **4.78×** for regions above 300
members, where a fixed selection covers a small fraction of the candidate set.

The mechanism is a ceiling result. An oracle with complete hindsight attains
3.12×, of which *magnitude* (paper count and degree) alone captures **76%**.
Across ~40 structural features, those measuring magnitude attain 2.1–2.6× and
those measuring shape or temporal dynamics attain 1.0–1.2×. Burt's structural
holes are supported in isolation (1.60× after magnitude is partialled out) but
contribute no measurable improvement in combination.

## What's here

```
report.md              the write-up
PLAN.md                the pre-registered design (v2.1), unmodified
configs/config.yaml    every threshold, with the reason it holds that value
src/
  extraction/          corpus, candidates, vocabulary, merges, registry
  graph/               event store, projections
  regions/             CPM regions, lineage, features, parent-sets, embedder
  models/              attachment, localisation, evaluation
reports/               per-phase reports and figures
logs/                  append-only audit trail: kills, merges, flags, decisions
judge/                 the LLM-judge method: prompt, protocol, hindsight audit
```

**No data is shipped.** Everything under `data/` and the LLM-judge payload under
`judge/` is derived and rebuildable from the arXiv snapshot by running the
pipeline below. What is tracked is the code that produces it, the configuration
that parameterises it, the reports that describe it, and the append-only logs that
record every decision.

## The birth registry

`data/registry/births.parquet` (produced by `src/extraction/registry.py`) —
100,295 concepts with dated coinage and
crystallisation, author-group counts, fates, and censoring flags. Built causally:
a concept crystallises in the first year *t* with ≥5 papers in each of *t..t+1*
across ≥2 disjoint author groups, computed from data ≤ *t*+1 only.

Spot check: NeRF 2020, BERT 2018, GAN 2015, Stable Diffusion 2022. `diffusion
model` shows coinage 2012 and crystallisation 2020 — the naming-to-consolidation
gap the design was built to expose.

Read [`reports/datasheet.md`](reports/datasheet.md) before using it. It documents
every known bias, including a measured causality-gate violation affecting 19.1% of
papers.

## Reproducing

The arXiv snapshot is CC0 and downloads without credentials:

```bash
curl -L -o data/raw/arxiv-snapshot-v299.zip \
  "https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv"
```

Then, in order — each stage writes a manifest and the next reads it:

```bash
uv sync
uv run python src/extraction/ingest.py          # corpus
uv run python src/extraction/nlp_cache.py       # tokenise
uv run python src/extraction/candidates.py      # coinage ledger
uv run python src/extraction/vocabulary.py      # per-origin vocab
uv run python src/extraction/registry.py        # birth registry
uv run python src/graph/build.py                # event store + graphs
uv run python src/regions/cpm.py                # regions + lineage
uv run python src/models/attachment.py          # attach births
uv run python src/models/layer2_split.py        # localisation, train/val/test
```

Runtime is a few hours on 8 cores / 12 GB. The two slow stages are the spaCy
token cache (~30 min) and the dictionary re-scan (~10 min); everything else is
DuckDB and finishes in minutes.

The LLM-judge stage is optional and off by default in a fresh clone. To reproduce
it, regenerate the term export as described in `judge/README.md`, judge the
batches, and run `src/extraction/judge_ingest.py` — which also runs the hindsight
audit that must pass before the verdicts are applied.

## Contributing / picking this up

[`CLAUDE.md`](CLAUDE.md) documents the causality rule the pipeline enforces, the
manifest and append-only-log conventions, pipeline order, the evaluation
partitions, and a list of the non-obvious failure modes encountered during
construction. Read it before changing anything under `src/`.

## How this was built

Every artifact carries a manifest (config hash, code commit, input hashes), and
every stage indexed by origin *T* asserts at build time that no input postdates
*T*. Decisions, threshold changes, out-of-band measurements and one process error
are recorded in `logs/flags.jsonl` rather than reconstructed after the fact.

Findings that did not work are kept, not deleted: the discarded unit-rate model
and its report are in `reports/phase8_models.md`.

## Status

Retrospective. The registry is complete through 2024, so origin 2024 at horizon 1
becomes gradeable once 2026's papers exist. The specification to be graded then is
declared in [`reports/layer2_localization.md`](reports/layer2_localization.md) and
should not be modified before it is.
