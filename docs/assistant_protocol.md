# Assistant Protocol — CC/TC Automation & Escalation

**Loop**
1) Load profile; ensure ≤4 active knobs.
2) Run Gate-0; freeze sloppy directions; emit report.
3) Plan experiment:
   - CC: choose instrument (geometry or rate), ABBA blocks.
   - TC: choose gate suite.
4) Generate pre-reg; lock code hash; mask labels.
5) Run/ingest data; compute metrics (S, Δ, Born, anisotropy, U drift, etc.).
6) CC decisions:
   - Compute BF_10 for β_S>0; if BF_10≥10 and Δ≤0.01 → Promote; if BF_01≥10 → Abandon; else if 1.5σ–2.5σ from threshold → extend +50% N and re-check; else apply one-time adaptive binning.
7) TC decisions: compare to thresholds; if ambiguous (1.5σ–2.5σ), extend +50% N; else escalate human review.
8) Report: evidence grade (A/B/C), non-zero-volume check (grid over active knobs), priors tier, pass/fail by gate.
9) Version guard: any pipeline/knob change ⇒ minor/major bump + Gate-0 rerun.

**Hard stops**
- No-signaling breach (Δ>0.01) in CC → pause and trigger diagnostic template.
