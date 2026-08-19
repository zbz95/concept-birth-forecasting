# Datasheet — Concept-Birth Forecasting corpus and registry

Living document. Every known bias, limitation, and accepted defect is recorded
here as it is discovered, with the measurement that established it and the PI
decision that disposed of it. Required for the Phase 10 release.

Corpus artifact: `data/interim/papers.parquet` — 269,814 papers, arXiv Kaggle
snapshot `Cornell-University/arxiv` v299 (2026-08-15), cs.CL ∪ cs.CV by primary
or cross-list, v1 submission date, hard cutoff 2025-12-31.

---

## 1. Revision contamination — v1 dates on latest-revision text

**KNOWN VIOLATION OF THE CAUSALITY GATE. Accepted by PI decision 2026-08-19.**

The arXiv Kaggle snapshot stores exactly one `title` and one `abstract` per
paper: the **current** metadata, as of the snapshot date. The `versions` array
carries version numbers and creation dates but no per-version text. Phase 1
therefore attributes latest-revision text to `versions[0].created`, the v1
submission date.

Measured on the 269,814-paper corpus:

| quantity | value |
|---|---:|
| multi-version papers | 112,325 (41.6%) |
| **text finalized in a later year than v1** | **51,509 (19.1%)** |
| latest version dated after `corpus_cutoff` 2025-12-31 | 10,426 |
| median v1→latest lag (multi-version) | 141 days |
| revised more than one year after v1 | 16,160 (6.0%) |

**Demonstrated, not hypothetical.**

- `gpt-4` (announced 2023-03-14) has yearly counts 2022:13 / 2023:814. Under
  `k_year=5, m=2` that yields a crystallization year of **2022** — a phantom
  birth for a concept that did not exist, and it falls inside `origins_test`.
  All 13 of the 2022-dated papers were individually checked against the
  snapshot: 13/13 have a latest version dated 2023 or later.
- `bert`: paper 1710.04334 (v1 2017-10-12, latest 2019-06-04) has a stored
  abstract naming BERT. Its true v1 abstract was fetched from
  `arxiv.org/abs/1710.04334v1` and does **not** mention BERT. Uncorrected, the
  ledger dates the coinage of `bert` twelve months before BERT existed.
- Paper 1805.11546 carries a v1 date of 2018-05-29 and a latest version of
  **2026-07-24** — 2026 text inside an artifact whose manifest certifies
  `max_observed_date = 2025-12-31`.

**Character of the bias.** Aggregate contamination is small — across eight
probe concepts with hard public debut dates, 67 papers of 269,814 (~0.02%) name
a concept that postdates their v1 date. But it is not randomly distributed. It
concentrates on the newest and fastest-rising concepts, which are precisely the
births the model is asked to forecast, and it moves coinage and crystallization
dates **earlier** — the direction that makes a forecaster look prescient.
Registry dates for recent, fast-growing concepts should be read as having a
one-year early-bias tail.

**Breaches.** Principle 1, and leakage-checklist items 1 (`vocab_T` contains no
term whose qualifying occurrences postdate T), 3 (registry entries computed from
data ≤ t+m−1), and 5 (graph/region/feature artifacts built from data ≤ T).

**Consequence that must be carried into the write-up.** Phase 9's accept
criterion — "leakage checklist passes on every artifact" — cannot be signed off
as written. Any results narrative must state that items 1, 3 and 5 are
known-violated for the 19.1% of papers whose text was finalized in a later year
than their v1 date.

**Alternatives considered and declined by the PI** (2026-08-19): a two-sided
registry with targeted v1 verification of only the terms whose crystallization
year moves; excluding the 51,509 cross-year-revised papers from concept counting
while keeping them in graphs; a full v1 backfill via arXiv's requester-pays S3
bucket. Logged in `logs/flags.jsonl`.

---

## 2. Pre-2016 cs.CL undercount (ACL Anthology era)

NLP work before roughly 2016 was published to the ACL Anthology rather than
arXiv. cs.CL is a thin sliver of the corpus before 2016 and roughly 39% after.
Pre-2016 NLP is systematically under-represented relative to CV, and a rise in
cs.CL volume around 2016–2018 is substantially a *posting-behaviour* change, not
a change in field metabolism. Regime splits at 2017 and 2020 partially absorb
this. 96.4% of the corpus is dated 2016 or later.

## 3. Corpus size below the original expected band

The plan's Phase 1 band was 3–6×10⁵ papers; the measured corpus is 269,814. The
category filter (primary or cross-list, 58,452 cross-listed-only papers
included) and the date field were both verified correct, so the shortfall is a
property of cs.CL ∪ cs.CV, not a defect. PI accepted the measured value on
2026-08-19 and the band was recalibrated to [255000, 285000] to guard against
snapshot drift and filter regression rather than against the original estimate.
Widening to cs.LG was declined as a Principle-6 violation; the plan reserves it
as the remedy for the Phase 4 power gate.

## 4. 2026 truncation

40,121 category-matching papers dated 2026-01-01 or later were dropped by the
hard cutoff. The snapshot is current to 2026-08-15, so calendar year 2025 is
complete and unaffected by snapshot lag.

## 5. Withdrawal detection

672 papers (0.25%) were dropped as withdrawn, detected by `/withdraw/i` on the
comments field plus a narrow abstract-prefix rule. Deliberately narrow, so that
papers discussing withdrawal topically are not killed. Note that withdrawal
status is read from snapshot-current metadata, so a paper withdrawn in 2025 is
absent from every origin including 2014 — a second, much smaller instance of the
same post-hoc-metadata issue as §1.

## 6. Tokenization decisions affecting concept identity

- **Hyphenated compounds are rejoined into single tokens.** spaCy splits
  `few-shot` into `few` / `-` / `shot` and tags the hyphen PUNCT. Left alone,
  `few-shot learning` (6,717 papers), `zero-shot learning` (9,754) and
  `one-shot learning` (1,208) all collapse into the single candidate
  `shot learning`. 96.2% of the corpus contains an intra-word hyphen.
- **The noun-chunk pattern is `ADJ*(NOUN|PROPN)+`, not the plan's literal
  `ADJ*NOUN+`.** spaCy tags `BERT` and `NeRF` as PROPN; a NOUN-only reading
  drops two of the plan's ten Phase 2 spot-list terms from that arm. PROPN is a
  noun in UPOS, so this is read as faithful to intent.
- **The chunk arm emits suffixes as well as maximal spans**, so
  `deep convolutional neural network` also yields `convolutional neural
  network`, `neural network`, `network`. Suffixes preserve the head noun. This
  is the only source of unigram candidates, and 4 of the 10 spot-list terms
  (`transformer`, `BERT`, `NeRF`, `GAN`) are unigrams.
- **spaCy's rule lemmatizer does not singularize PROPN**, so `GANs` lemmatizes
  to `gans` while `GAN` gives `gan`. These fork into separate ledger rows and
  must be reunited by Phase 3's deterministic string-variant merge rules, which
  are timeless and therefore causally safe.
