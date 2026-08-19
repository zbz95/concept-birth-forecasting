# External LLM judging — what to do

**106,960 terms, 214 batches of 500.** These are the clusters that appear in any
`vocab_T` for T in 2014–2024 — everything that can become a graph node or a
registry entry.

## Run

1. Paste `PROMPT.md` (everything below the `---`), then one file from `batches/`.
2. Save the reply verbatim as `verdicts/batch_NNNN.txt`, matching the input
   filename.
3. Repeat. You can do a subset — the ingest reports which batches are missing and
   only applies verdicts for batches it received.
4. `uv run python src/extraction/judge_ingest.py`

That reattaches verdicts to terms, logs every deletion reversibly to
`logs/kills.jsonl`, and runs the hindsight audit.

## Expected reply format

One line per input, same order, nothing else:

```
1	KEEP
2	DROP
```

The parser tolerates `,`, `:` or spaces as the separator and is case-insensitive.
It reports malformed lines rather than guessing.

## Why the terms are shuffled and bare

Each batch shows the phrase and nothing else — no counts, no dates, no example
papers — and the list is shuffled with a fixed seed rather than ordered by
frequency. Both are deliberate. Showing a judge that a term appears in 20,000
papers hands it the popularity signal the plan forbids it to use, and sorting by
frequency does the same thing more subtly.

## The constraint that actually matters

The plan's `forbidden_criteria` are `familiarity`, `novelty`, `importance`. This
is not stylistic. A 2026 model knows which concepts became famous. If it keeps
`diffusion model` because it recognises the name, and drops an obscure but
perfectly real 2019 term, it has written 2026 knowledge into `vocab_2019` — and
`vocab_2019` is an input to a model that forecasts 2020. That is the exact
failure the causality gate exists to prevent, arriving through a side door.

So the prompt tells the judge, repeatedly, to ignore recognition, and to KEEP
anything it has never heard of whose form denotes a kind of thing.

## And then it is checked anyway

`judge_ingest.py` runs a hindsight audit that does not take the judge's word for
it. Concepts that crystallized and then **declined** are real concepts that never
became famous; concepts that **persisted** did. Within a size stratum, a
form-blind judge keeps both at the same rate. A judge using familiarity keeps the
persisted ones more often.

The audit reports the keep-rate gap per stratum. A gap ≥ 5 percentage points is
flagged to `logs/flags.jsonl` as suspected hindsight and should stop the run —
it means the filter is not admissible and the verdicts should be discarded
rather than patched.

A useful sanity habit while judging: if a batch reply contains a term you had to
think about because you recognised it, that is the signal to re-read constraint 2.
