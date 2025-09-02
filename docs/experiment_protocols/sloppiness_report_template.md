# Gate-0 Sloppiness & Identifiability Report

**Profile & Track:** (QC/CC, CORE/TC, etc.)
**Active knobs (dimensionless):** e.g., [κ_a, κ_ξ, α_ℓ, η·W0]; baseline values.
**Metrics M(θ):** pick 6–10: (U drift, interference L¹, Born L¹/∞, L anisotropy, CHSH S, Δ, Gate-χ pass fraction).

**Procedure**
1) Finite-difference Jacobian J via symmetric perturbations.
2) Fisher F = Jᵀ W J (W = inverse variance weights).
3) Eigenanalysis: stiffness indices s_k = λ_k / Σ λ_j.

**Decisions**
- Keep K directions with Σ s_k ≥ 0.8 (stiff subspace).
- Freeze any movement along sloppy directions for this profile.

**Deliverables**
- Jacobian table; stiffness spectrum; stiff combos with plain-language interpretation.
- Final **Active Knob Set** (≤ 4).
