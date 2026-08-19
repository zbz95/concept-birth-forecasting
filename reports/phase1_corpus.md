# Phase 1 — Corpus: accept report

**Status: accept criteria PASS. Two issues raised and disposed of by the PI on 2026-08-19 —
a Principle-6 band miss (accepted, band recalibrated) and a causality-gate breach
(accepted and documented; see `reports/datasheet.md` §1).**

Config hash `1b4c86efac62e9a3` · snapshot `Cornell-University/arxiv` v299 (updated 2026-08-15, CC0)
· artifact `data/interim/papers.parquet` · figure `reports/phase1_papers_per_year.png`

## Result

| | count |
|---|---:|
| records scanned | 3,134,984 |
| matched cs.CL or cs.CV (primary or cross-listed) | 310,607 |
| dropped — v1 date after `corpus_cutoff` 2025-12-31 | 40,121 |
| dropped — withdrawn stubs | 672 |
| dropped — duplicate id / no v1 date / empty abstract | 0 |
| **kept** | **269,814** |

Composition: cs.CV only 170,748 · cs.CL only 92,171 · both 6,895.
Cross-listed-only papers (primary category elsewhere) 58,452 — top primaries
cs.LG 16,099, eess.IV 13,768, cs.AI 4,585, cs.RO 4,254.

## Accept criteria

- **Papers-per-year plot per category** — produced.
- **Totals sane against known arXiv growth** — PASS. 2015→2025 growth factor 18.9×
  (34.2% CAGR); log-scale curve is smooth with no kinks. 96.4% of the corpus is 2016+.
- Date field is `versions[0].created` (v1 submission) only; `update_date` is carried
  nowhere. Max observed v1 date is exactly 2025-12-31, zero records past cutoff.

## Tripwire

`papers_kept = 269,814` falls **below** the expected band 3–6×10⁵ (−10.1% vs the
floor). Logged to `logs/flags.jsonl`. The plan's prescribed response is *check
filter / date field; consult PI.*

**Filter and date field both check out.** The category filter matches cs.CL/cs.CV
anywhere in the category list, so cross-listed papers are included (58,452 of them —
this is not a primary-only filter bug). No records were lost to malformed dates,
empty abstracts, or duplicate ids. The withdrawal filter removed 672 papers (0.25%),
consistent with arXiv's real withdrawal rate. **The shortfall is a property of the
corpus, not a defect in the ingest.** cs.CL ∪ cs.CV through 2025-12-31 simply
contains 269,814 papers.

PI accepted the measured value on 2026-08-19; band recalibrated to [255000, 285000].
Note that widening the category set to reach the band would be
a Principle-6 violation — the plan reserves cs.LG widening as the remedy for the
*Phase 4 power gate*, not for a Phase 1 band miss.

## Post-hoc finding — revision contamination (2026-08-19)

An adversarial audit after this report was written established that the snapshot
stores only the CURRENT title/abstract, which Phase 1 stamps with the v1 date.
19.1% of the corpus carries text finalized in a later year than its v1 date, and
this produces at least one phantom crystallization (`gpt-4` dated 2022 for a
concept announced 2023-03-14). The PI accepted this and elected to document
rather than mitigate. Full measurement, demonstrated harm, and the standing
consequence for Phase 9 sign-off are in `reports/datasheet.md` §1.

## For the datasheet

- **Pre-2016 cs.CL undercount (ACL Anthology era).** Visible in the figure: cs.CL is
  a thin sliver before ~2016 and roughly 39% of the corpus after. Pre-2016 NLP is
  systematically under-represented relative to CV. Regime splits at 2017 and 2020
  partially absorb this; it must not be read as an NLP metabolism signal.
- **2026 truncation.** 40,121 matched papers dated 2026-01-01 onward were dropped by
  the hard cutoff. The snapshot is current to 2026-08-15, so the 2025 year is complete
  and unaffected by snapshot lag.
- **Withdrawal detection** is comment-based (`/withdraw/i` on the comments field) plus
  a narrow abstract-prefix rule. Deliberately narrow, to avoid killing papers that
  discuss withdrawal topically.
