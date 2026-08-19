# Phase 7 — Attachment and targets: gate report

**STOP-AND-ASK: the plan requires PI review of 20 birth→attachment mappings
before modeling.** They are in `reports/phase7/attachment_eyeball.md`.

Artifacts: `data/registry/attachment/attachments.parquet` (per-birth localization
records) and `targets.parquet` (5,007 fractional target rows).

---

## C2 decision gate: PASS, by a wide margin

| | |
|---|---:|
| mean dual-attached births per tune origin (k=4, h=1) | **416.7** |
| `dual_attachment_floor` | 10 |
| per tune origin | 315, 476, 459 |

**I predicted this would be close and I was wrong.** The prediction came from the
pair-unit finding that only 15% of region pairs share a concept. That turns out
to be the wrong quantity: dual attachment does not require the two regions to
overlap. A birth's 10-concept profile can overlap two entirely disjoint regions
by 3 concepts each — and that is the *more* interesting case for the bridge
hypothesis, not the excluded one.

The relaxed pair rule the plan pre-declared (≥3 with one region, ≥2 with the
other) is therefore not needed. `attachment.relaxed_pair_rule` stays disabled,
and the bridge hypothesis can be registered under the standard rule.

## The invariant holds

Total target mass equals the number of attached births at all 45
(origin, k, horizon) combinations, exactly.

## Which arm carried the attachments

Of 80,113 accepted attachments at k=4:

| | count | share |
|---|---:|---:|
| both arms | 39,876 | 50% |
| overlap ≥ 3 only | 3,227 | 4% |
| **hypergeometric surprise only** | **37,010** | **46%** |

**The v2.1 amendment was decisive.** Replacing the Jaccard arm with a
hypergeometric tail is carrying nearly half of all attachments on its own — those
are births whose profile overlaps a small region by 2 concepts, which is
extraordinary for a 12-member region and unremarkable for a 900-member one. A
size-blind arm cannot make that distinction, and without this arm the orphan rate
would roughly double.

## Orphan rate

| T | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| orphan rate | 47% | 48% | 47% | 33% | 40% | 44% | 30% | 25% |

Falling over time, as regions cover more of a growing vocabulary. Between a
quarter and a half of births attach to nothing at all — reported per the plan,
and a real limit on what fraction of the birth process the region system can
even express.

## The triptych — all three deviate, and the reason is the same

The plan expects: transformer → an NLP region; capsule network → a CV region;
diffusion model → orphan.

| concept | expected | observed | why |
|---|---|---|---|
| `transformer` | NLP region | not in the window | crystallized 2015, before the attachable window; and its profile is `spatial transformer module, spatial transformer network` |
| `capsule network` | CV region | **orphan** at T=2017 | profile is right (`capsnet, dynamic routing, routing, digit`) but none of those were in vocab_2017 regions yet |
| `diffusion model` | orphan | attached to a **denoising** region | profile is `nonlinear diffusion, reaction-diffusion, linear filter, poisson` |

**Two of the three are the wrong-sense problem the Phase 4 audit already
flagged, arriving downstream.** The registry's `transformer` is the *spatial*
transformer of 2015, not the Transformer architecture; its profile says so
explicitly. The registry's `diffusion model` is the PDE sense — and given that
sense, attaching to a region of `denoiser, denoising, gaussian, image denoising`
is *correct*, since anisotropic diffusion is a classical denoising method. The
attachment machinery is behaving properly on a concept whose identity is wrong.

`capsule network` is a different and more interesting failure: the attachment is
orphaned because its parent concepts (`capsnet`, `dynamic routing`) had not yet
cleared the vocabulary floor at T=2017. A birth whose parentage is itself brand
new has nothing established to attach to. That is a real property of the design,
not a defect, but it means the region system structurally cannot localize the
most novel births — the ones a forecaster would most want.

## Attachment strength

Accepted attachments have median overlap 3 (p90 6, max 10 — the profile length)
and median surprise 6.6×10⁻⁶ against a 10⁻³ threshold, so the typical accepted
attachment clears the bar by two orders of magnitude rather than sitting on it.
The plan asks for a sensitivity grid only if margins are fragile; on this
evidence they are not, though the 4% carried by the overlap arm alone are the
rows that would move first.

## What this unblocks

Feature 8 (confirmed-birth persistence) can now be back-filled, per the plan's
6 → 7 → back-fill → 8 ordering, using only births with crystallization
≤ T−(m−1).
