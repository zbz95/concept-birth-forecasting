# Concept-Birth Forecasting

Can we forecast where new concepts crystallize in NLP/CV, with skill beyond a
size-and-heat null? See `PLAN.md` (v2.1) — it is binding.

## Layout
    configs/config.yaml   every threshold; changing one changes the config hash
    src/manifest.py       run manifests, causality gate, append-only logs
    src/extraction/       Phase 1-4: corpus, candidates, vocabulary, registry
    src/graph/            Phase 5: event store + projections
    src/regions/          Phase 6: CPM regions, lineage, parent-sets
    src/models/           Phase 8: null + challengers
    src/eval/             Phase 9: rolling-origin evaluation
    data/interim/         papers.parquet and friends (gitignored)
    logs/                 kills.jsonl, merges.jsonl, flags.jsonl — append-only
    reports/              phase accept reports and figures

## Run
    uv run python src/extraction/ingest.py     # Phase 1

Every artifact writes `<path>.manifest.json`: config hash, code commit, input
hashes, `as_of`, and the max observed input date. `assert_causal` refuses to
build any artifact indexed by T that absorbed data after T.

## Status
Phase 1 complete — 269,814 papers. See `reports/phase1_corpus.md`; a Principle-6
tripwire is open and awaiting PI decision.
