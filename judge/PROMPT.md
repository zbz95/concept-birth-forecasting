# Judging prompt

Paste this, then one `batches/batch_NNNN.txt` file. Use the **same prompt every
time** — a prompt that drifts between batches makes the verdicts
non-reproducible, and the run manifest records this file's hash.

---

You are filtering a terminology list automatically extracted from computer
science paper abstracts (natural language processing and computer vision).

For each numbered phrase, decide whether it **names a kind of thing** or merely
**describes** something.

**KEEP** if the phrase denotes a category, task, method, model, architecture,
artifact, dataset, phenomenon, or measurable quantity — something a paper could
be *about*, that a researcher could build, study, or evaluate.

**DROP** if the phrase is:

- an evaluative or degree description — *thorough analysis*, *competitive
  accuracy*, *rigorous experiment*, *remarkable efficacy*
- a generic container noun with a non-naming modifier — *conventional image*,
  *different architecture*, *inherent weakness*, *new word*
- a fragment of a longer phrase that does not stand alone — *carlo tree* (from
  *Monte Carlo tree search*), *fold cross* (from *10-fold cross validation*)
- a general-purpose noun phrase with no technical denotation — *product
  quality*, *cognitive task*, *thorough study*

## Constraints — these are not optional

This list feeds a forecasting study whose validity depends on the filter being
blind to hindsight. Violating these invalidates the study rather than merely
degrading it.

1. Judge **only** the linguistic form and denotation of the phrase itself.
2. Do **not** consider whether you recognise the term, or whether it is famous,
   important, influential, recent, or historically significant. A term you have
   never seen and a term you know well must be treated identically.
   *liveness detection* and *diffusion model* are both KEEP, and for the same
   reason: each names a kind of thing.
3. Do **not** consider whether the term is currently popular, or when it became
   prominent. Do not reason about dates at all.
4. A phrase you have never encountered is **KEEP** if its form denotes a kind of
   thing. Unfamiliarity is never a reason to DROP.
5. Do not reason about whether the term "matters". Only whether it *names*.

Ambiguous cases go to KEEP. This filter is meant to remove phrases that are
clearly not names, not to make fine judgements about significance.

## Output format

Exactly one line per input phrase, in the same order, nothing else:

```
1	KEEP
2	DROP
3	KEEP
```

No preamble, no commentary, no explanation, no markdown fences.
