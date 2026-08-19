# Phase 4 — Birth registry: gate report

**STOP-AND-ASK. The plan requires PI review of the registry audit before modeling.**

Artifacts: `data/registry/births.parquet` (136,195 concepts), `concept_year.parquet`.
Accept materials: `reports/phase4/birth_audit_sheet.csv` (200 births, blank verdict
column), `reports/phase4_births_per_year.png`.

---

## Result

| fate | concepts |
|---|---:|
| persisted | 47,957 |
| crystallized-then-declined | 4,209 |
| coinage-only | 63,418 |
| censored (indeterminate at the 2025 boundary) | 20,611 |

Registry complete through **2024** = `last_complete_year` − (m−1). Coinage→
crystallization lag: median 5 years, p90 15, max 30, min 0.

**Phase 4 power gate: PASS, with enormous margin.** `power_floor` is 50
crystallizations/yr in test origins; the measured range is 4,429–11,983. The
design does not starve. If anything the opposite.

## Spot check — 13/18 within ±1 year

| concept | coinage | crystallization | expected | verdict |
|---|---:|---:|---:|---|
| nerf | 2020-03-19 | 2020 | 2020 | exact |
| bert | 2017-10-12 | 2018 | 2018 | exact |
| gan | 2015-06-18 | 2015 | 2015 | exact |
| stable diffusion | 2022-06-20 | 2022 | 2022 | exact |
| chain-of-thought | 2018-10-16 | 2022 | 2022 | exact |
| knowledge distillation | 2016-04-01 | 2016 | 2016 | exact |
| **diffusion model** | **2012-01-07** | **2020** | 2020 | **exact — 8-year gap** |
| vit | 2019-06-04 | 2021 | 2020 | +1 |
| capsule network | 2018-01-02 | 2018 | 2017 | +1 |
| transformer | 2009-07-09 | 2015 | 2017 | −2, wrong-sense |
| attention mechanism | 2011-04-13 | 2015 | 2017 | −2, *see below* |
| semantic segmentation | 2012-02-10 | 2013 | 2015 | −2 |
| contrastive learning | 2017-10-06 | 2018 | 2020 | −2 |

**The plan's central test passes.** `diffusion model` shows coinage 2012 and
crystallization 2020 — the coinage/crystallization gap the plan requires to
appear, at 8 years.

**Two of the five "misses" may be the expectation being wrong, not the registry.**
Attention mechanisms predate the Transformer — Bahdanau et al. is 2014 — so a
2015 crystallization for `attention mechanism` is defensible and the plan's 2017
figure appears to conflate attention with the Transformer. Likewise
`contrastive learning` predates SimCLR. `transformer` at 2015 is genuine
wrong-sense contamination (electrical transformers; coinage 1994–2009).

## Tripwire: crystallizations/yr is ~10× the expected band

Plan band: low 10²–10³ in the late period. Measured:

| 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---:|---:|---:|---:|---:|---:|
| 4,429 | 4,999 | 4,109 | 5,024 | 9,411 | 11,983 |

**Diagnosis: the excess is non-concept terminology, not a counting bug.** A
random draw from the audit sample:

> liveness detection · **rigorous experiment** · optical flow estimation ·
> **thorough analysis** · molecular property prediction · **inherent weakness** ·
> abstractive dialogue summarization · **competitive accuracy** ·
> complex logical reasoning · **product quality** · popular text-to-image ·
> **conventional image**

Roughly half of the sample are real concepts; the rest are generic
adjective–noun compounds that satisfy every filter — nominal head, no function
word, not verb-led, df ≥ 9 — without naming anything.

**This cannot be fixed deterministically without collateral damage.** Two
candidate rules were measured against the crystallized set (50,733 concepts,
2014–2024):

| rule | would remove | but also kills |
|---|---:|---|
| head noun is generic | 10,046 (20%) | **`diffusion model`** — a plan spot-list term |
| starts with an evaluative adjective | 5,668 (11%) | `weak supervision`, `general data protection` |
| either | 14,569 (29%) | both of the above |

`diffusion model` and `thorough analysis` are structurally identical —
modifier + noun, with a head that is generic in isolation. No surface rule
separates them. This is exactly the discrimination the LLM judge's stated remit
covers ("not a noun phrase; generic"), and the judge was switched off at the
Phase 3 gate.

## Two defects found and fixed during construction

- **Censoring never fired.** A concept whose first qualifying window starts in
  2025 needs 2026 data; the per-year frequency test simply failed on the missing
  year, so it was silently labelled coinage-only rather than censored. Boundary
  windows are now judged on the years actually observable and marked
  indeterminate. 20,611 concepts are affected — they were being reported as
  having failed to crystallize when the truth is that we cannot yet know.
- **Clusters could crystallize before their own coinage** (observed lag of
  −1 years). Coinage was read from the cluster label's own first occurrence
  rather than the earliest among its members. Now zero concepts crystallize
  before coinage.

## What the audit sheet needs from you

200 randomly sampled births, 2014–2024, with a blank verdict column. The plan's
verdict set is {correct, wrong-date, wrong-sense, ambiguous}; the sample above
suggests a fifth is needed — **not-a-concept** — since `rigorous experiment` has
neither a wrong date nor a wrong sense. The agreement rate becomes the registry's
published error bar.

## Reading the metabolism figure — one caveat

`reports/phase4_births_per_year.png`, right panel, shows coinages peaking around
2017–2018 and falling steeply through 2025 while crystallizations keep rising,
the two curves crossing near 2023. **The coinage decline is a right-censoring
artefact, not a slowdown in the field.** A term only enters the registry universe
once it clears `min_total_freq`, so a concept coined in 2024 has had barely a year
to accumulate the nine papers that make it visible, and one coined in 2025 has had
none. The true recent coinage rate is unobservable until those terms mature. Do
not read the crossing as the field coining less and consolidating more.
