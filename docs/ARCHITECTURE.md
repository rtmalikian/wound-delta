# Architecture

## The one idea

The product ships its own OpenAI-compatible inference gateway, and every
AI feature in the product calls that gateway. Nothing calls a vendor API
directly. On top of that sits the Week 3 idea: **the measurement is a
subtraction, done in code**. The model is the last and smallest step.

```
            ┌──────────────────────────────────────────────────────────┐
            │                      FastAPI app                          │
            │                                                          │
  HTTP ───► │  /api/*  cases · captures · compare · trend · demo       │
            │     │                                                    │
            │     ▼                                                    │
            │  app/wounddelta/  domain module                          │
            │   imaging (align · segment · diff · flags)               │
            │   synth (ground-truth fixtures) · models · service       │
            │     │  phrasing only, numbers already measured           │
            │     ▼                                                    │
            │  app/gateway.py  embedded OpenAI-compatible gateway      │
            │   /v1/models · /v1/chat/completions · /v1/embeddings     │
            │   tiers: wound-delta-frugal | wound-delta-full           │
            │     │                                                    │
            │     ├─ stub     deterministic phrasing, no model         │
            │     ├─ local    Ollama / llama.cpp / vLLM on-site        │
            │     └─ upstream any OpenAI-compatible provider           │
            │                                                          │
            │  app/safety.py  PHI redaction gate on labels             │
            └──────────────────────────────────────────────────────────┘
```

## The shared-skeleton answer, Week 3

Weeks 1 and 2 established the pattern: one platform skeleton (gateway +
redaction gate + verified generation + eval harness), a new domain module
per week. Week 3 reuses `app/gateway.py` and `app/safety.py` with two
changes, both forced by the vision domain:

1. **Multimodal dispatch and a token ledger.** Chat messages can carry
   image parts, and the gateway estimates vision tokens with the published
   tile maths (low detail flat 85; high detail 85 + 170 per 512px tile).
   Budgets become assertable: the eval harness checks the frugal median
   against 300 tokens per pair. `CALL_LEDGER` counts every dispatch, which
   is how "unchanged pair = zero model calls" is a test, not a slogan.
2. **Tiers as routing profiles.** `gateway.TIERS` holds the frugal and
   full profiles. `resolve_tier` reads the model alias or the `X-Tier`
   header. The domain module asks for a profile and gets budgets, detail,
   and verifier behaviour from it. Nothing else in the codebase knows
   tiers exist.

What changed is only the domain module: discharge plans and PA packets
out, imaging in.

## The compare pipeline (app/wounddelta/service.py)

1. **Hash gate.** Load both photos; identical SHA-256 means a templated
   answer and zero dispatches.
2. **Align.** FFT phase correlation estimates the translation between
   frames; the new frame is shifted into the previous frame's coordinates.
   Translation only: rotation and scale are handled by the capture habit
   (ghost overlay), not by warping, because warping would bend the
   measurement.
3. **Segment.** Two channels: strong redness (index above 105) or much
   darker than skin (gray below 140). Constants in `imaging.py`, with the
   reasoning in the docstring, including why the inflamed-margin channel
   (redness above 85) is measured separately instead of segmented.
4. **Diff.** Area change %, grew/healed pixels, redness spread outside
   both masks, darkening of the wound bed, and the bounding box of the
   change mask (the ROI the frugal tier sends).
5. **Flags.** Four deterministic pattern checks with stated thresholds.
6. **Phrase.** One gateway call with the measurements JSON and, in the
   frugal tier, only the ROI crop at `detail:low`. Negligible change with
   no flags short-circuits to a template and zero calls. In the full tier
   the call carries the full frame at `detail:high`, and a second call
   verifies that every number in the summary appears in the measurements.

## Verified generation, applied to phrasing

Weeks 1 and 2 verified *claims against sources*. Week 3's version is
smaller and stricter: the stub (and the prompt contract for real backends)
can only phrase numbers present in the measurements block, and the full
tier's verifier pass extracts every number from the summary and checks it
against the block, reporting strays. A model that invents a number fails
the check in the response payload, where a human or a test can see it.
