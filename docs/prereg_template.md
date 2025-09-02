# Pre-registration (Profile: CORE | QC | MDL-X; Track: CC | TC)

**Experiment family:** (Geometry A/B | Switch-rate A/B | TC gate suite)

## Hypotheses (directional)
- CC: S(High) > S(Low); Δ ≤ 0.01.
- TC: A_refinement decreases; Born L¹/∞ within bands.

## Active knobs & Priors Tier
- Active knobs (≤ 4): list and values.
- Priors tier: 1 | 2 | 3 (see Parameter Policy).

## Design
- CC: ABBA blocks (equal lengths), randomized order; instrument = geometry or rate.
- TC: specify graphs, window rules, detector micro-coupling ranges.

## Endpoints & Analysis
- CC: Bayes factor BF_10 for β_S>0; no-signaling Δ; (secondary) regression on overlap lower bound.
- TC: U drift, interference L¹, Born L¹/∞, anisotropy; thresholds per `theory.md`.

## Blinding & QC
- Mask condition labels until pipeline frozen (hash code + config).
- Dual timing references and drift checks; negative geometry controls.

## Power & N
- Variance estimates; N_max; futility bound for CC; minimal relevant effects.

## Stopping & Deviations
- Optional stopping via BF; deviations ⇒ version bump + Gate-0 rerun.
