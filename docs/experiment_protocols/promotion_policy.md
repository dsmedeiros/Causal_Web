# Promotion Policy — Bayesian Sequential Design (CC)

**Hypothesis:** Directional (“High-overlap S > Low-overlap S”).

**Prior:** Heavy-tailed on effect size β_S (e.g., Cauchy with two pre-registered scales). Report sensitivity.

**Design:** Block-randomized ABBA instrument (geometry or switch-rate), equal block lengths.

**Decision thresholds (after each block):**
- **Promote:** BF_10 ≥ 10 and no-signaling (Δ ≤ 0.01).
- **Abandon:** BF_01 ≥ 10 (strong evidence for null).
- **Continue:** otherwise, until **N_max** or **futility** (posterior P(|β_S| > β_min) < 0.1).

**Tier-2 “Grade-A” Evidence:**
- Fresh run BF_10 ≥ 10 (or two independent runs BF_10 ≥ 3 each) + no-signaling.

**Adaptive binning (one-time):**
- If 1/3 < BF_10 < 3, split High/Low once by pre-registered rule; enforce monotonicity and test with isotonic/ordered-means; promotion only with strong evidence and Δ ≤ 0.01.
