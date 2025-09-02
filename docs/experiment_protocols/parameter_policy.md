# CWT Parameter Policy (Profiles, Knob Budgets, Bounds & Change Control)

**Scope.** Constrain degrees of freedom so CWT remains falsifiable.

## Profiles & ≤4 Active Knobs
- **CORE (no Bell):** { η·W0, α_d/α_ℓ, γ/d0, a/b }.
- **QC (Bell, quantum-compatible):** { κ_a, κ_ξ, α_ℓ, η·W0 }.
- **MDL-X (Bell, exploratory MDL):** { κ_a, κ_ξ, ν_χ, T_χ/W0 }.

All non-active parameters are frozen to §13 defaults of `theory.md`.

## Invariants & Hard Constraints
- Stability: 0 ≤ α_d ≤ 1, 0 < α_ℓ ≤ 1, α_d + α_ℓ ≤ 1.
- Window bounds: 1 ≤ W_min ≤ W(v) ≤ W_max < ∞; default inject_mode = incident.
- RNG lanes separated (settings / noise / ε-seeds / χ).
- Gauge test (Gate-G) must pass when gauge is on.

## External Evidence (Bounds, not Handcuffs)
- Three prior tiers for {ν_χ, T_χ, κ_a}:
  - **Tier-1 (Conservative):** sharp near Null-MD; use for negative-control replications of published layouts.
  - **Tier-2 (Reference):** weakly informative; used for tightening bounds.
  - **Tier-3 (Exploratory):** broader priors for new layouts; discovery only.
- Any positive CC claim must pass **Null-compatibility** for ≥1 benchmark layout within reported error bands.

## Non-Zero-Volume Requirement
- Minimal gate sets must pass on a set of active knobs with hypervolume ≥ V_min (report V_min).

## Multi-Gate Coherence
- CC and TC graduate independently on their **minimal sets**; enhanced gates tracked but do not back-invalidate graduation.
- Joint CC+TC synthesis attempted only after both graduate.

## Change Control & Versioning
- **Patch:** docs/plots/controls that don’t alter pipelines or active knobs.
- **Minor/Major:** any change to active knobs, pipelines, or thresholds ⇒ Gate-0 rerun and new pre-reg.
- **Retune budget:** at most one small pre-unblinding adjustment per minor version (documented).
