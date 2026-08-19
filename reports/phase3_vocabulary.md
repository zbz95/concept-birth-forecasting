# Phase 3 — Filtering + normalization: gate report

**STOP-AND-ASK. Three decisions needed before Phase 4.**

Artifacts: `data/interim/vocab/vocab_{2014..2024}.parquet`, `merge_map.parquet`,
`concept_postings.parquet` (77,487,127 rows), `termhood.parquet`.
Review materials: `reports/phase3/{vocab_sample,kill_log_slice,merge_audit}.md`.

---

## A leak I introduced and fixed

My first implementation computed the POS table and C-value **globally**, over all
years, then applied one filter at every origin. That admits a term into
`vocab_2019` partly *because* it became frequent in 2023 — leakage-checklist
item 1. Every filter is now recomputed from data dated ≤ T:

| step | how it is made causal |
|---|---|
| frequency | `df_T` from `ledger_yearly` restricted to `year <= T` |
| pattern kills | POS table derived from abstracts dated ≤ T only |
| C-value | computed from `df_T` and containment among the ≤ T survivors |
| merges | only edges whose evidence date is ≤ T are active |
| cluster counts | a paper naming several members of one cluster counts **once** |

`concept_postings.parquet` exists to make this possible: term-level postings for
every candidate with global `df >= min_total_freq`. That is a strict superset of
every possible `vocab_T`, because document frequency only grows with time, so it
can be built once and every per-origin filter recomputed from it without leaking.

## Vocabulary by origin

| T | terms df≥9 | after kills | clusters | merged | cv>0 | cv>15 | cv>25 | cv>50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2014 | 21,291 | 3,887 | 3,698 | 174 | 3,670 | 2,748 | 1,637 | 800 |
| 2016 | 48,123 | 8,514 | 8,028 | 441 | 7,955 | 6,278 | 3,822 | 1,844 |
| 2018 | 111,417 | 19,819 | 18,639 | 1,075 | 18,444 | 14,945 | 9,073 | 4,438 |
| 2019 | 162,482 | 28,797 | 27,063 | 1,572 | 26,785 | 22,074 | 13,447 | 6,510 |
| 2020 | 228,931 | 40,582 | 38,100 | 2,236 | 37,706 | 31,374 | 19,330 | 9,390 |
| 2021 | 302,528 | 53,723 | 50,424 | 2,966 | 49,897 | 41,774 | 26,021 | 12,617 |
| 2022 | 384,180 | 68,248 | 64,057 | 3,739 | 63,436 | 53,470 | 33,332 | 16,165 |
| 2023 | 491,139 | 87,940 | 82,490 | 4,835 | 81,710 | 69,285 | 43,255 | 21,153 |
| 2024 | 629,777 | 113,795 | 106,911 | 6,105 | 105,958 | 90,444 | 56,955 | 27,755 |

Pattern kills at T=2024: stopword_straddler 371,498 · verb_led 101,095 ·
non_nominal_head 41,025 · ordinal 1,795 · generic_head 569. All logged to
`logs/kills.jsonl` with an `undone` column; the full kill set is recoverable at
any time as ledger-minus-survivors. Nothing is deleted.

**Two readings of the plan's stated filters were needed, both adopted on
linguistic grounds and both verified not to lose any Phase-2 spot-list term:**

- *stopword-straddler* is read strictly — a candidate containing a function word
  **anywhere** dies, not just at an edge. Only the raw n-gram arm can produce
  such a candidate (RAKE splits on stopwords, and `ADJ*(NOUN|PROPN)+` cannot
  match one), so `network in image` and `range of camera` are n-gram artefacts,
  not terms. Known cost: `bag of word` is killed.
- *a term is a noun phrase*, so its head must be nominal. Without this, bare
  adjectives survive as high-frequency "concepts" — `novel` (75,178 papers),
  `large`, `different`, `available`, `deep`.

## Decision 1 — `c_value_min`

Config leaves this null by design. The empirical answer is unusually clear.

| floor | clusters at T=2022 | deletes | what it deletes |
|---:|---:|---:|---|
| **0** | 63,436 | 621 (1%) | `fold cross`, `carlo tree`, `dyer et`, `photometric bundle`, `probability hypothesis`, `absolute word` |
| 15 | 53,470 | 10,587 (17%) | the above **plus** `wav2vec2`, `sent2vec`, `vit-l`, `tvqa`, `xtreme`, `gpt-4` |
| 25 | 33,332 | 30,725 (48%) | the above **plus** `stylegan inversion`, `covid-19 lesion`, `sar target` |

Floor 0 deletes exactly what C-value exists to detect: terms whose frequency is
fully explained by the longer terms containing them. `fold cross` is a fragment
of *10-fold cross validation*; `carlo tree` of *Monte Carlo tree search*;
`probability hypothesis` of *probability hypothesis density*. Every higher floor
is a frequency threshold wearing C-value's clothes, and `min_total_freq` already
occupies that role.

**This matters more here than in a normal terminology project.** A concept
crystallizes at `k_year=5, m=2` — roughly ten papers. A C-value floor prunes by
frequency, so it prunes precisely the newly observable concepts that are this
project's forecasting targets:

| floor | share of just-observable clusters (df_T 9–25) retained, T=2019 / 2022 / 2024 |
|---:|---|
| 0 | 99% / 99% / 99% |
| 10 | 91% / 92% / 93% |
| 15 | 73% / 76% / 77% |
| 25 | 25% / 29% / 30% |
| 50 | **0%** / **0%** / **0%** |

**Recommendation: `c_value_min = 0`.**

## Decision 2 — the expected band no longer fits

The plan's Phase 3 band is a single range, 20–60k clusters. But `vocab_T` is a
function of T: it grows 29× from 3,698 (2014) to 106,911 (2024), tracking a
corpus that grows 19× over the same period. No fixed C-value floor is in band at
every origin — floor 0 is in band for 2019–2022 only; floor 25 for 2021–2024
only; and reaching band at 2024 requires floor ≈ 50, which retains 0% of
just-observable concepts.

The band and the crystallization threshold are not mutually satisfiable. One has
to be recalibrated, and Principle 6 forbids picking the threshold to fit the band.

## Decision 3 — LLM judge

Deferred from the earlier gate to here, as agreed. The judge-off vocabulary now
exists and can be inspected in `reports/phase3/vocab_sample.md`.

## Merge map

145,772 terms → 137,093 clusters (8,679 merged). Sources: 8,010 timeless
string-variant edges, 712 evidence-dated Schwartz-Hearst edges, **0 merges from
embeddings** (they may only nominate). 3,819 merges flagged on a coinage gap
> 3 years; 7 clusters flagged oversized; 300 ambiguous acronyms refused.

Two defects were found and fixed during construction:

- **Transitive over-merge through ambiguous acronyms.** `SR` attests to speech
  recognition, super-resolution, success rate, surface reconstruction and
  spatial reasoning; union-find chained all five into one 33-member cluster.
  Acronyms now require ≥3 characters and a single long form accounting for ≥70%
  of attestations; ambiguous ones are refused and queued for human review.
  `bnn` (binary vs bayesian neural network), `lda` (latent dirichlet allocation
  vs linear discriminant analysis) and `mrf` (markov random field vs magnetic
  resonance fingerprinting) are correctly refused.
- **Dominance measured on surface forms.** `CNN` attests 2,236 times to
  "convolutional neural networks" and 1,066 to "convolutional neural network" —
  one long form split by number, scoring 65% and wrongly refused. Ambiguity is
  now judged on normalized long forms.

Evidence-dating verified: `peft` merges into *parameter-efficient fine-tuning*
only from 2023-07-11, `vlm` from 2022-09-30, so at origin 2020 both are
correctly separate terms.

## Leakage spot-check

First origin at which each concept enters `vocab_T`:

| concept | public debut | first in vocab_T | verdict |
|---|---:|---:|---|
| bert | 2018 | 2019 | correct |
| nerf | 2020 | 2020 | correct |
| vit | 2020 | 2021 | correct |
| capsule network | 2017 | 2018 | correct |
| transformer | 2017 | 2016 | wrong-sense (electrical transformers) |
| diffusion model | 2020 | 2017 | wrong-sense — and the coinage/crystallization gap the plan predicts |
| **gpt-4** | **2023** | **2022** | **true leak** — the documented revision contamination (datasheet §1) |

`gpt-4` is the only genuine violation, and it is the known, PI-accepted defect.
