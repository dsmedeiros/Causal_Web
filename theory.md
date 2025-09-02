Causal Web Theory & Layered Causal Coherence Model (CWT/LCCM)

Version 1.5 — Testable Program (Strict-Local, Sub-Quantum)
Status: Research draft (2025-08-26)
Principle: Every rule reads/writes only local state and messages on finite causal paths. No global tick; no nonlocal action.

What’s new in 1.5 (testable program)
• Two-Track program: Correlation Core (CC) and Transport Core (TC) graduate independently; joint synthesis attempted later.
• Robust pass: non-zero-volume requirement (no razor-thin needle solutions).
• Promotion via Bayesian sequential design (directional hypotheses; Bayes factors; optional stopping).
• Overlap instrumentation via block-randomized A/B (difference-in-differences); continuous overlap only secondary.
• Evidence grades (A/B/C) and adaptive binning with monotonicity guard.
• Assistant-friendly escalation protocols and decision rules.

⸻

0. Assumptions & Invariants

Assumptions. Finite graph \mathcal G; d_0(e)∈ℤ_{≥1}; d_eff(e)∈ℤ_{≥1}; all Q‑packets emitted are unit‑norm (‖ψ‖_2=1); parameters α_d,α_ℓ,η,γ,ρ_0>0; per‑destination tiebreak seq is stable and monotone.

Stability constraint. 0≤α_d≤1, 0<α_ℓ≤1, and  α_d+α_ℓ≤1  (required).

Determinism & RNG contract. The per‑event RNG seed is derived only from local data: 
  seed = splitmix64(run_seed ⊕ d_arr ⊕ (v_dst≪1) ⊕ (e_id≪2) ⊕ (seq≪3)),
with disjoint lane tags OR’d for (i) detector setting, (ii) readout noise, (iii) seed emission. Changing the queue/tie‑break invalidates replay.

No‑signaling diagnostic (Bell). For each local setting a_D, the remote‑conditioned marginals are flat in the remote setting:
  max_{a'_D} | P(b=+1 | a_D,a'_D) − 0.5 | ≤ 0.01  (pre‑registered run length).

Tsirelson guard (MDL mode). In MI_conditioned runs, require CHSH S ≤ 2.86 (buffer above 2√2≈2.828); parameter regions exceeding this are flagged inadmissible.

Window bounds. 1 ≤ W_min ≤ W(v) ≤ W_max < ∞.

0C. External Bounds & Null-Compatibility (MDL)
We treat high-quality Bell tests as external benchmarks that define prior tiers for MD parameters (ν_χ, T_χ, κ_a). A Null-MD profile (ν_χ→0 or T_χ below relevant separations) reproduces flat S vs overlap at those layouts. Discovery is pursued in new layouts under broader priors; any positive signal must also be consistent with at least one benchmark layout within reported error bands (Null-compatibility check).

0D. Research Tracks & Core-Claim Graduation
We separate claims to avoid all-or-nothing bottlenecks.
 • Correlation Core (CC) minimal set: Gate-6 (CHSH) + no-signaling + Gate-χ (genericity) + Null-compatibility (≥1 published layout). 
 • Transport Core (TC) minimal set: Gate-U (unitarity proxy), Gate-B/B′ (Born + micro), Gate-L1 (basic isotropy), Gate-G (gauge).
A track “graduates” when its minimal set passes with non-zero-volume in the active-knob subspace (see §12). Enhanced gates (e.g., TC L2/L3; CC Tsirelson bands) do not invalidate a track already graduated. Only after both tracks graduate do we attempt a CC+TC synthesis; if the joint pass region collapses to measure-zero, we conclude CWT cannot realize both cores simultaneously under current ontology.

⸻

1. Ontology & State

1.1 Graph substrate

A finite directed multigraph \mathcal G=(V,E). Multiple edges between the same ordered vertex pair are allowed.

1.2 Locality

All state is stored on the owning vertex or edge. Rules may read:
•the local object’s state;
•payloads delivered along incident edges;
•static metadata of incident edges/vertices.

No global variables, clock, or cross-graph scans.

1.3 Events & causal order

An event is delivery of a packet across an edge. Causal precedence e_1\prec e_2 holds if a directed path of events leads from e_1 to e_2. This relation is a partial order. No simultaneity structure is assumed.

Local total order & determinism. For a fixed destination v, arrivals are totally ordered by the tuple
\big(d_{\mathrm{arr}},\, v,\, e_{\mathrm{id}},\, \mathrm{seq}\big),
where e_{\mathrm{id}} is a stable edge identifier and \mathrm{seq} is a per-destination tiebreak counter.

Lemma (Deterministic processing & partial-order preservation).
On a finite \mathcal{G} with finite base delays and \max_e d_{\mathrm{eff}}(e)<\infty, the scheduler that processes events in ascending \big(d_{\mathrm{arr}}, v, e_{\mathrm{id}}, \mathrm{seq}\big) (per destination) (i) processes every enqueued event exactly once, and (ii) preserves the causal partial order: if e_1\prec e_2, then e_1 is processed before e_2.
Sketch. Downstream arrivals satisfy d_{\mathrm{arr}}(e)=d(u)+d_{\mathrm{eff}}(e)\ge d(u)+1, so path-causal events appear in non-decreasing d_{\mathrm{arr}} and the per-destination tie-break makes order deterministic.

Addendum (linear extension). Because d_eff≥1 on every hop, along any causal path d_arr increases by at least 1; thus the per‑destination total order is a linear extension of the causal partial order.

1.4 Oscillators (optional)

Each vertex v may host a phase \theta_v\in [0,2\pi) with intrinsic \omega_v. An oscillator “fires” when \theta_v\to 2\pi, emitting on each outgoing edge. In the event-driven kernel we simply consume emissions already queued by prior arrivals; the oscillator is an interpretation layer.

1.5 State variables

Vertex v
•Depth: d(v)\in\mathbb Z_{\ge0} (last processed arrival-depth).
•Window index: w(v)=\big\lfloor d(v)/W(v)\big\rfloor.
•Layer: \ell(v)\in\{\text{Q},\Theta,\text{C}\}.
•Q-state: \psi_v \in \mathbb C^D (normalized at window close).
•Q-accumulator: \psi^\text{acc}_v\in \mathbb C^D (reset each window).
•Theta-probabilities: p_v \in \Delta^{K-1} (simplex).
•C-bit & confidence: (\text{bit}_v\in\{0,1\},\, \text{conf}_v\in[0,1]) via majority buffer.
•Fan-in this window: \Lambda_v\in\mathbb Z_{\ge0}.
•Q-arrivals this window: \Lambda_v^{Q}\in\mathbb Z_{\ge0}.
•Meters: E_Q(v),\,E_\Theta(v),\,E_C(v),\,E_\rho(v) at window close (Sec. 7).
•Ancestry fields (Bell): rolling hash h_v and phase-moment m_v\in\mathbb R^3.

Edge e:u\to v
•Base delay: d_0(e)\in\mathbb Z_{\ge1}.
•Effective delay: d_\text{eff}(e)\in\mathbb Z_{\ge1}.
•Coupling: \alpha_e\in\mathbb R_+.
•Phase & gauge: \phi_e, A_e \in \mathbb R.
•Unitary: U_e\in\mathbb C^{D\times D} (Q-layer).
•Stress-energy: \rho_e\in\mathbb R_{\ge0}.
•Bridge strength (epsilon-pairs): \sigma_e\in\mathbb R_{\ge0} (0 if not a bridge).

Packet payload
•Layer payload: one of \psi (Q), p (Theta), or bit (C).
•Intensity (derived at delivery): I\in[0,1] (Sec. 4.4).
•Arrival-depth: d_\text{arr}\in\mathbb Z_{\ge0}.
•Bell hidden var (optional): \lambda=(u,\zeta).
•Optional epsilon-seed fields (Sec. 6).

⸻

2. Temporal structure: arrival-depth & windows

Arrival-depth is the only operational time. The kernel keeps a priority queue keyed by
(d_\text{arr},\, v_\text{dst},\, e,\, \text{seq})
and repeatedly delivers the minimum. Deterministic RNG uses only local fields and these keys with the seed contract in §0 (“Determinism & RNG contract”): splitmix64 over (run_seed,d_arr,v_dst,e_id,seq) and disjoint RNG lanes for settings, noise, and seed emission. On delivery to v:
 d(v)\;\leftarrow\;\max\{d(v),\, d_\text{arr}\}.

Each vertex has a local window length W(v) (Sec. 5.1). The window index is w(v)=\lfloor d(v)/W(v)\rfloor. A window closes at v when w(v) increases; then:
•compute E_Q and normalize \psi_v;
•reset \psi^\text{acc}_v and \Lambda_v;
•evaluate meters / transitions;
•while in C, retain (\text{bit}_v, \text{conf}_v) across windows, clearing only the majority buffer. These fields reset on C→Θ transitions.

⸻

3. Event life-cycle (flow & pseudocode)

flowchart TD
  A[Emit on e: u→v] --> B[Compute d_arr = d(u)+d_eff(e)]
  B --> C{Enqueue at v by (d_arr, v, e_id, seq)}
  C --> D[Deliver to v]
  D --> E{Layer L(v)}
  E -->|Q| F[Accumulate ψ_acc(v) with α_e·phase·U_e ψ_pkt]
  E -->|Θ| G[Accumulate p(v) with α_e·p_pkt]
  E -->|C| H[Update majority buffer & confidence]
  F & G & H --> I[Update intensity I; update ρ_e; maybe emit ε-seeds]
  I --> J{Window close?}
  J -->|No| K[Continue deliveries]
  J -->|Yes| L[Normalize (ψ or p), compute meters (E_Q,E_Θ,E_C,E_ρ)]
  L --> M[Apply LCCM transitions (Q/Θ/C) with hysteresis/holds]
  M --> N[Reopen window; update W via controller]

Emission / scheduling: when u emits on edge e:u→v,
d_\text{arr}(e)= d(u) + d_\text{eff}(e),\quad \text{enqueue}(d_\text{arr}, v, e).

Delivery: (to v, via e, payload X)

 d(v) = max(d(v), d_arr)
 Lambda_v += 1

 if layer(v) == "Q":
     Lambda_v_Q += 1
     psi_acc[v] += alpha_e * exp(i * (phi_e + A_e)) * (U_e @ psi_packet)
 elif layer(v) == "Theta":
     p_v = normalize(p_v + alpha_e * p_packet)
 else:  # C
     (bit_v, conf_v) = majority_update(bit_v, conf_v, packet.bit)
 
 # compute intensity I for rho/delay update (Sec. 4.4)
 I = intensity_from_layer(layer(v), e, payload)
 
 # stress-energy and delay update on edge e
 rho_e, d_eff[e] = rho_update_and_delay(e, neighbors(e), I)
 
 # epsilon-seed emit on Q delivery (Sec. 6)
 if layer(v) == "Q":
     emit_epsilon_seed(v, e, d_arr)
 
 # update ancestry (Bell), then LCCM transitions (Sec. 5)
 update_ancestry(v, payload)
 maybe_transition_layer(v)

Window close at v:

E_Q(v) = ||psi_acc[v]||^2
if E_Q(v) > 0: psi[v] = psi_acc[v] / sqrt(E_Q(v))
psi_acc[v] = 0
E_Theta(v) = kappa_Theta * (1 - H(p_v))     # H = Shannon entropy
E_C(v) = kappa_C * conf_v
Lambda_v = 0
if Lambda_v_Q == 0:
    m_v = normalize((1 - delta_m) * m_v)
Lambda_v_Q = 0


⸻

4. Stress-energy & delays

4.1 Edge neighborhood (diffusion graph)

Edges are neighbors if they share a vertex (edge-adjacency). Let \langle\rho\rangle_{\text{nbrs}(e)} be the mean over neighbors of e.

Edge-adjacency averaging operator. With |E| edges, define M\in\mathbb{R}^{|E|\times|E|}:
(M\mathbf{x})e \;\triangleq\; \frac{1}{|\mathrm{Nbr}(e)|}\sum{e’\in \mathrm{Nbr}(e)} x_{e’},
\quad
\mathrm{Nbr}(e)\equiv\{e’\in E:\; e’\text{ shares an endpoint with }e\}.
Then M is row-stochastic (M\mathbf{1}=\mathbf{1}) with spectral radius \le 1.

4.2 ρ-update (per delivery and system form)

Local update (per delivered edge):
\rho_e\ \leftarrow\ (1-\alpha_d-\alpha_{\text{leak}})\,\rho_e
\;+\; \alpha_d\,\big\langle\rho\big\rangle_{\text{nbrs}(e)}
\;+\; \eta\,I,
\quad \rho_e\ge 0.

Vector form & stability. Let \boldsymbol{\rho}^t\in\mathbb{R}{\ge 0}^{|E|}, intensity \boldsymbol{I}^t\ge 0. With diffusion \alpha_d\in[0,1] and leak \alpha\ell\in(0,1]:
\boxed{
\boldsymbol{\rho}^{t+1}
= \Big[(1-\alpha_d-\alpha_\ell)I + \alpha_d\,M\Big]\boldsymbol{\rho}^{t}
\ \mathbf{+}\ \eta\,\boldsymbol{I}^t
}
Stability & contraction. Require α_d+α_ℓ ≤ 1. For any eigenvalue |λ(M)|≤1, the homogeneous multiplier magnitude is ≤ (1−α_ℓ); with α_ℓ>0 the homogeneous mode contracts.

Steady state (homogeneous input). For \boldsymbol{I}^t\equiv \bar I\,\mathbf{1},
\[
\boldsymbol{\rho}^{\*}
\;\approx\;
\frac{\eta\,\bar I}{\alpha_\ell}\,\mathbf{1}
\quad\text{(after mixing)}.
\]

4.3 Effective delay (saturating)

\boxed{
 d_\text{eff}(e)
 = \max\!\Big\{1,\; d_0(e) + \big\lfloor \gamma \,\ln\!\big(1+\rho_e/\rho_0\big)\big\rfloor\Big\}
}
•\gamma>0, \rho_0>0 shape the saturation. Ensures integrality, positivity, and no singularities.

4.4 Layer intensities (bounded)

Intensity I is taken from the current layer \ell(v) at delivery.
•Q: I = \|\,U_e\,\psi\,\|_2^2 \le 1, with all emitted Q‑packets unit‑norm (‖ψ‖_2=1); accumulation scaling uses α_e in the accumulator only.
•Theta: I = \|p\|_1 (with \sum p\le 1 after mixing). Default inject_mode = incident (mean per‑window), to avoid ρ inflation under high fan‑in; ‘incoming’ remains available for experiments.
inject_mode="incoming" applies this intensity per delivered edge.
Non-incoming modes (incident,outgoing) inject using the mean per-edge \|p\|_1 over the window/batch to avoid saturation under high fan-in.
•C: I = \text{bit}\in\{0,1\}.

4.7 Causal DAG & Microcausality (Bell context)
Let the micro-beables B = {ψ, p, bit, ρ, χ, σ,…} evolve only along graph adjacency with depth-bounded rules. Denote Past(A) as all beables in A’s past lightcone under the discrete causal graph.
Microcausality assertions:
 A_out ⟂ B_set | Past(A) and B_out ⟂ A_set | Past(B).
Thus the model forbids signaling even with MDL; any observed signaling is a model or implementation error.

⸻

5. LCCM (Layered Causal Coherence Model)

5.1 Local window size (target + controller)

Let \deg_\text{inc}(v)=\deg_\text{in}(v)+\deg_\text{out}(v)+\deg_\text{bridge}(v). Let \bar\rho_v be the current mean of \rho_e over edges incident to v (recomputed when the window closes). Define a target window:
W_{\mathrm{tgt}}(v)=W_0+\Big\lfloor \zeta_1\ln\big(1+\deg_\text{inc}(v)\big)+\zeta_2\ln\big(1+\bar\rho_v/\rho_0\big)\Big\rfloor,\quad W_{\mathrm{tgt}}\ge 1.

Then update the actual window once per window close via a clipped EWMA:
\boxed{
W_{t+1}(v) \;\leftarrow\; \mathrm{clip}\!\Big(
W_t(v)\;+\;\mathrm{sat}{\Delta}\big[\beta\,(W{\mathrm{tgt}}(v)-W_t(v))\big],
\; W_{\min},\; W_{\max}\Big)
}
with 0<\beta\le 1 and per-step cap \Delta>0.
•Stability. Unclipped, this is a contraction with factor 1-\beta; clipping preserves boundedness and prevents overshoot.
•Settling time. T_{95}\approx 3/\beta window updates (rule of thumb).
•Theta reset policy: \theta_\text{reset}\in\{\text{uniform},\text{renorm},\text{hold}\} chooses how p_v is reset when the window closes (default renorm).
Define primitives used above:
  sat_Δ(x) ≡ sign(x)·min(|x|,Δ),   clip(x,a,b) ≡ min(max(x,a),b).
Bounds: enforce 1 ≤ W_min ≤ W(v) ≤ W_max.

5.2 Thresholds & timers (nondimensional & hysteresis)

Define nondimensional load \hat\lambda_v\triangleq \min\{1,\Lambda_v/W(v)\} and choose 0<b<a<1:
N_\text{decoh}(v)= a\,W(v),\qquad
N_\text{recoh}(v)= b\,W(v).
•Q→Θ (“decoh_threshold”): when \Lambda_v \ge aW(v) within the current window. \psi_v becomes frozen (read-only); p_v activates.
•Θ→Q (“recoh_threshold”): when \Lambda_v \le bW(v) for T_\text{hold} consecutive windows and E_Q(v)\ge E_{Q,\min}.
•Θ→C (“classical_dominance”): when H(p_v)\le H_\text{max}, bitfrac ≥ f_min, and conf_v ≥ conf_min for T_\text{class} windows.
Define bitfrac precisely as the fraction of ones in a fixed‑length majority buffer of length L_C (default L_C=16).
•(Optional C→Θ can be added; not required for v1.3.)

Hysteresis rationale. A simple 2-state Markov approximation under noise \epsilon shows mean dwell time increases with the band \Delta\lambda=a-b and with the hold timers. Recommended: a\approx 0.8, b\approx 0.5, T_{\mathrm{hold}}\in[2,4].

5.6 Instruments for MDL/Bell (operational overlap)
Primary instrumentation uses binary conditions that modulate causal connectivity without changing hardware:
 • Geometry instrument: same detectors/RNG; place the setting generator to realize High-overlap vs Low-overlap by design (block-randomized ABBA).
 • Switch-rate instrument: alternate fast vs slow setting blocks at spacelike separation.
Primary estimator: difference in block means of S with block fixed effects (ABBA), or a difference-in-differences layout to remove slow drift. Secondary analysis may use a conservative lower bound for continuous overlap (interval arithmetic with redundant timing), but promotion decisions rely on the binary instrument.

⸻

6. Epsilon-Pairs: local correlation channels

6.1 Seeds (Q-layer only)

On Q-delivery at v, emit seeds along outgoing edges.

Default: one seed per window. Enabling emit_per_delivery switches to a per-arrival emission mode.
•Ancestry prefix: match key from h_v (first L_bits bits).
•Angle tag: local phase proxy \theta_v (e.g., \operatorname{atan2}(m_{v,y},m_{v,x})).
•Expiry by depth: d_\text{exp} = d_\text{emit} + T_\text{TTL}.
•A forwarded seed uses the current edge delay: d_\text{next}=d_\text{curr}+d_\text{eff}; continue only if d_\text{next}\le d_\text{exp}.

Capacity & eviction. Cap the per‑vertex seed pool at N_\text{seed} (default 64) to avoid unbounded growth; on overflow, evict FIFO by depth or by smallest σ.

6.2 Binding & bridges

Two seeds collide at a vertex and bind iff: (i) both unexpired; (ii) ancestry prefixes match (length L); (iii) |\theta_1-\theta_2|\le \theta_\text{max}.

Binding creates a transient bridge edge with:
•initial \sigma=\sigma_0;
•local effective delay
d_\text{bridge}(u,v)=\max\!\left\{1,\left\lfloor\operatorname{median}\{d_\text{eff}(e): e\text{ incident to }u\text{ or }v\}\right\rfloor\right\};
•a stable synthetic id (negative id space) for determinism/logs.
Bridges are scheduled exactly like edges.

6.3 \sigma-dynamics & binding math
•On traversal: \sigma\leftarrow (1-\lambda_\text{decay})\sigma + \sigma_\text{reinforce}.
•Each window (idle): \sigma\leftarrow (1-\lambda_\text{decay})\sigma.
•Remove bridge when \sigma<\sigma_\text{min}.

Accidental bind probability (per encounter). For uniformly random relative angles,
\Pr\big(|\Delta\theta|\le \theta_{\max}\big)=\theta_{\max}/\pi.
If seeds arrive at rate \lambda per window and TTL spans T_TTL windows, and a match also requires L_bits ancestry prefix equality, then the expected false‑binds per window at a vertex are
\[
\boxed{
\mathbb{E}[\mathrm{false\binds}] \;\approx\; \lambda^2\,T_\mathrm{TTL}\,2^{-L_\mathrm{bits}}\,\frac{\theta_{\max}}{\pi}.
}
\]
If you intend the rate conditional on a prefix match, state that explicitly and omit the 2^{-L_bits} factor. Choose \theta_{\max} to meet a false-positive target p_{\mathrm{fp}}.

Bridge lifetime (mean). With Poisson traversal rate r:
\[
\mathbb{E}[\sigma^\] \approx \frac{\sigma_{\mathrm{reinforce}}\,r}{\lambda_{\mathrm{decay}}},\qquad
\mathbb{E}[T_{\mathrm{drop}}] \approx \frac{1}{\lambda_{\mathrm{decay}}}\,
\ln\!\frac{\mathbb{E}[\sigma^\]}{\sigma_{\min}}.
\]

⸻

7. Conservation (meters, calibration & residual)

At each window close:
•Q meter: E_Q(v) = \|\psi^\text{acc}_v\|_2^2.
•Theta meter: E_\Theta(v) = \kappa_\Theta\,(1 - H(p_v)).
•C meter: E_C(v) = \kappa_C\cdot \text{conf}_v.
•ρ meter: E_\rho(v) = \kappa_\rho\, \bar\rho_v.

Calibration (nondimensionalization). On a baseline run, choose \kappa so that
\mathbb{E}[E_Q] = \mathbb{E}[E_\Theta] = \mathbb{E}[E_C] = \mathbb{E}[E_\rho] = 1,
making residual unitless and balanced.

Leak. At steady state with homogeneous input, expected leak per window is approximately
\mathrm{leak} \;\approx\; \alpha_{\text{leak}} \sum_{e\in E}\rho_e.

Residual (per window; region/global). Positive residual indicates net creation after accounting for leak and boundary ρ‑flux; negative indicates net loss. For any finite processed region \mathcal R,
\boxed{
\mathrm{Res} \;\equiv\;
\sum_{v\in\mathcal R}\big(\Delta E_Q+\Delta E_\Theta+\Delta E_C\big)
\;+\;\kappa_\rho\sum_{e\in\partial \mathcal{R}}\Delta\rho_e
\;+\;\mathrm{leak}.
}
We report an EWMA of \mathrm{Res} over windows.
Vertex‑star option (debug). For per‑vertex diagnostics, approximate ∂\mathcal R by the incident edges ("star") of v and sum κ_ρ·Δρ_e over the star.

⸻

8. Bell / Shared-Ancestry Selector (SAS)

8.1 Local ancestry fields (Q-arrivals only)

Each vertex keeps a rolling hash h_v=(h_0,h_1,h_2,h_3)\in(\mathbb{Z}_{2^{64}})^4 and a unit moment m_v\in\mathbb R^3.

From a delivered packet \psi on edge e with phases \phi_k and weights w_k=|\psi_k|^2/(\sum|\psi|^2+\varepsilon), define \tilde\phi_k=\phi_k+\phi_e+A_e, z=\sum_k w_k\,e^{i\tilde\phi_k}, mean direction \mu=\arg z, concentration \kappa=|z|\in[0,1], and u_\text{local}=[\cos\mu,\sin\mu,\kappa].

Moment EMA: m_v\leftarrow \mathrm{normalize}\big((1-\beta_m)\,m_v+\beta_m\,u_\text{local}\big).
Window decay: if the prior window had \Lambda_v^{Q}=0, decay m_v\leftarrow \mathrm{normalize}\big((1-\delta_m)\,m_v\big).

Rolling hash (splitmix64 lanes, strictly local):
h_0\leftarrow \mathrm{smix}\big(h_0\oplus v \oplus (d_\text{arr}\ll 1)\big),
h_1\leftarrow \mathrm{smix}\big(h_1\oplus e \oplus (\text{seq}\ll 1)\big),
h_2\leftarrow \mathrm{smix}\big(h_2\oplus \mathrm{bits}(\mu)\big),
h_3\leftarrow \mathrm{smix}\big(h_3\oplus \mathrm{bits}(\kappa)\big).
Seed prefix: first L MSBs of h_0.

8.2 Source hidden variable

At a pair source S, compute \lambda=(u,\zeta) from (h_S,m_S): blend m_S with a hash-derived direction (weights \beta_m,\beta_h), normalize to unit u; \zeta\in[0,1) is a local hash-blended scalar used later in a deterministic rotation. Both halves of the pair carry the same \lambda.

8.3 Detector setting (toggle MI)

At detector D with ancestry (h_D,m_D):
•MI_\text{strict}: draw a_D from a hash of h_D (independent of \lambda).
•MI_\text{conditioned}: draw a_D from a vMF-like distribution centered on m_D blended with h_D, with concentration \kappa_a (strictly local but statistically correlated via shared ancestry).

8.4 Local readout

Outcome b\in\{+1,-1\}:
b=\operatorname{sgn}\!\big(\langle a_D,\ R(h_D,\zeta)\,u\rangle + \xi\big),
with local noise \xi\sim\mathcal N(0,\sigma(\kappa_\xi)). R is a deterministic local rotation about a hash-derived axis with angle 2\pi\zeta\,\alpha_R.

Predictions & tests.
•RNG lanes: use disjoint RNG lanes for (i) detector setting, (ii) readout noise, (iii) seed emission (see §0).
•MI_\text{strict}: CHSH S\le 2 by construction.
•MI_\text{conditioned} (\kappa_a>0): S(\kappa_a) increases monotonically in simulation while **no‑signaling holds per remote setting**; operational test is max_{a'_D}\!|P(b=+1|a_D,a'_D)−0.5|≤0.01. A Tsirelson guard flags any S>2.86 as inadmissible. See Validation Appendix.

8.8 Bayesian Sequential Design (CC promotion)
Directional hypothesis: “S is larger in High-overlap than Low-overlap condition” (or slope β_S>0). Use a heavy-tailed prior for β_S (e.g., Cauchy), pre-registered scales. After each AB block, compute Bayes factor BF_10.
Decision thresholds:
 • Promote Tier-3→Tier-2 if BF_10≥10 and no-signaling holds;
 • Abandon if BF_01≥10;
 • Otherwise continue with more blocks until N_max or a futility bound is met (posterior P(|β_S|>β_min)<0.1).
Tier-2 “Grade-A” requires BF_10≥10 in a fresh run or BF_10≥3 in two independent runs; always check no-signaling.

8.9 Adaptive Binning (guarded)
If BF is inconclusive (1/3<BF_10<3), split each instrument bin once by a pre-specified rule (distance threshold or rate quantiles), impose a monotonicity constraint (“bin means non-decreasing”), and test with isotonic regression or ordered-means ANOVA. At most one refinement per side. Promotion only if the monotonic trend has strong evidence and no-signaling holds.

⸻

9. Adaptive parameters & dimensionless groups
•Windows: W_0,\zeta_1,\zeta_2 (local topology & \bar\rho).
•Decoherence: a,b,T_\text{hold},E_{Q,\min}.
•rho/delay: \alpha_d,\alpha_{\text{leak}},\eta,\gamma,\rho_0.
•epsilon-pairs: \Delta, L, \theta_\text{max}, \sigma_0,\lambda_\text{decay},\sigma_\text{reinforce},\sigma_\text{min}.
•Bell: \beta_m,\beta_h,\kappa_a,\kappa_\xi.

Useful dimensionless ratios for DOE:
\frac{\Delta}{W_0},\
\frac{\alpha_d}{\alpha_{\text{leak}}},\
\gamma\ \text{vs}\ d_0,\
\eta W_0,\
\frac{a}{b},\
\frac{\sigma_\text{reinforce}}{\lambda_\text{decay}},\
\kappa_a,\ \kappa_\xi.

Notes on renames & clamps (v1.3.1).
• C_min → E_{Q,\min} (Θ→Q gate key is the Q‑meter).
• λ_v used in thresholds replaced by \hat{λ}_v=\min\{1,Λ_v/W(v)\}.
• L→L_bits and TTL→T_TTL in ε‑pairs.
• Default inject_mode set to “incident”.

⸻

10. Emergence sketch (physics at scale)
•Wave transport: Q-layer unitary accumulation over windows yields interference/diffraction; Θ/C layers provide local decoherence & classicalization under fan-in pressure.
•Geometry: sustained intensities raise \rho, which delays edges via d_\text{eff}, bending causal paths (a discrete lensing analog).
•No-signaling: all correlations arise via ancestry and depth-bounded epsilon-channels; no superluminal dependencies.

⸻

11. Minimal simulator interface (for reference)
•Scheduler: PQ keyed by (d_\text{arr},v_\text{dst},e,\text{seq}).
•Window rule: close when w(v) increments; compute meters; reset accumulators.
•Intensity: derived per layer at delivery; drives \rho and d_\text{eff}.
•epsilon-pairs: seeds with expiry by depth; local bind; transient bridges with sigma-dynamics.
•Bell: ancestry updates; \lambda at source; local setting & readout at detectors.
•Interface knobs: emit_per_delivery, inject_mode.

(These are implementation notes, not additional physics.)

⸻

## 12. Validation Suite (Gates & Decisions)
**Gate-0 (Sloppiness & Identifiability).** Around the chosen baseline, compute a finite-difference Jacobian for key metrics vs the **≤4** active dimensionless knobs; form Fisher information, keep stiff directions that explain ≥80% of variance. Sloppy directions are frozen in this profile.

**Non-Zero-Volume pass (robustness).** A configuration passes only if the set of parameters that satisfy the minimal gate set has **non-zero hypervolume** in the active-knob subspace (≥ V_min). Razor-edge solutions do not count.

**Track minima (graduation).**
 • **CC minimum:** Gate-6 (S with CIs), no-signaling (Δ≤0.01), Gate-χ (genericity ensemble: majority of seeds produce S>2 within bands), Null-compatibility for ≥1 benchmark layout.
 • **TC minimum:** Gate-U (global Q-norm drift ≤1%/1e4 windows; interference ≤2% L1), Gate-B (Born L1/L∞), Gate-B′ (weak-coupling invariance), Gate-L1 (isotropy), Gate-G (gauge).
Enhanced gates (L2/L3; QC Tsirelson bands) are tracked but do not invalidate graduation.

**Promotion decisions (Bayesian sequential).** For CC directional claims under instrumentation (High vs Low):
 • After each AB block, compute BF_10 for β_S>0; if BF_10≥10 → Promote; if BF_01≥10 → Abandon; else Continue (or stop for futility).
Tier-2 Grade-A requires a fresh run BF_10≥10 (or two runs BF_10≥3) with no-signaling; report C_MI and sensitivity to prior scale.

**Adaptive binning (one-time).** If BF inconclusive, perform one pre-registered split per bin and test monotonicity; promotion requires strong evidence and Δ≤0.01.

**Pareto & Reporting.** For each track, plot the Pareto front across core metrics (CC: S, Δ, Gate-χ pass fraction; TC: U-drift, Born L1, anisotropy). Publish evidence grade (A/B/C), V_min check, priors tier, and whether graduation achieved.

⸻

13. Defaults (illustrative; tune per graph)
•W_0=4,\ \zeta_1=\zeta_2=0.3.
•W_min=2,\ W_max=64.
•a=0.7,\ b=0.4,\ T_\text{hold}=2,\ E_{Q,\min}=0.1.
•\alpha_d=0.1,\ \alpha_{\text{leak}}=0.01,\ \eta=0.2,\ \gamma=0.8,\ \rho_0=1.0.
•\Delta\approx 2W_0,\ T_\text{TTL}=\Delta,\ L_\text{bits}=16,\ \theta_\text{max}\approx \pi/12,\ \sigma_0=0.3,\ \lambda_\text{decay}=0.05,\ \sigma_\text{reinforce}=0.1,\ \sigma_\text{min}=10^{-3},\ N_\text{seed}=64.
•\kappa_a\in\{0,2,5,10\},\ \kappa_\xi=0.5.
•H_\text{max}=0.2,\ f_\text{min}=0.6,\ \text{conf}_\text{min}=0.7,\ T_\text{class}=2.
•inject_mode = incident.
•Ancestry: \beta_m=0.1,\ \beta_h=0.3,\ \delta_m=0.02.

⸻

Notes on strict locality
•No rule reads non-incident state or any global aggregate.
•Seeds use only their own expiry_depth and local edge d_\text{eff}.
•Bell draws depend only on local ancestry fields and local RNG seeded from local data.
•Scheduler order is purely (d_\text{arr}, v_\text{dst}, e, \text{seq}).

⸻

Appendix A — Validation Figures (reproducible)

Place rendered plots in docs/figs/ and record run references.

FigTitleWhat to plot
A.1Delay saturation & ρ convergenced_{\mathrm{eff}} vs \rho/\rho_0; \rho_t with/without diffusion
A.2ε-pairs: binds & lifetimeFalse-bind rate vs \theta_{\max}; lifetime vs traversal rate r
A.3Bell/SASCHSH S vs \kappa_a; no-signaling marginals
A.4InterferenceVisibility vs phase
A.5LCCM stabilityLayer occupancy vs load; hysteresis with holds
A.6ConservationResidual EWMA over a long run

Reproduction note. Each figure caption should include {git_sha, theory_version, graph_hash, optimizer, cfg_hash} and a pointer to the log directory.

Appendix B — Instrument Designs (CC)
B.1 Geometry instrument (ABBA): same hardware; two placements engineered to minimize vs maximize causal connectivity by design. Randomize block order; equal block lengths; document distances/timing.
B.2 Switch-rate instrument: slow vs fast setting blocks with spacelike separation; randomize order and durations; confirm detector independence.
B.3 Overlap bounds (secondary): derive conservative lower bounds via interval arithmetic with redundant clocks; consistency check only—primary decisions rely on AB differences.

Appendix C — Power & Sensitivity
Provide variance estimates for S per setting, sample-size calculators for a minimal relevant effect, and futility bounds used in Bayesian sequential design.
