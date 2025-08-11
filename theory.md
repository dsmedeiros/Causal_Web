# Causal Web Theory & Layered Causal Coherence Model (CWT/LCCM)

**Version 1.2 — Strict-Local**
**Status:** Research draft (2025-08-11).
**Principle:** Every rule reads/writes only local state and messages on finite causal paths. No global tick; no nonlocal action.

## What's new since 1.1

* Replaced global tick with **arrival-depth** scheduling (event-driven).
* Resolved Q-layer normalization: **store energy, normalize state** at window close.
* **LCCM hysteresis** with local fan-in counters & hold timers (Q↔Θ, Θ→C).
* **Stress-energy ρ** and **delay map** $d_\text{eff}$ via a saturating log law.
* **Dynamic ε-pairs** using **depth-based TTL** seeds and transient **bridges** with σ-decay/reinforcement.
* **Bell / SAS**: strictly local, toggled **measurement-independence** (MI\_strict vs MI\_conditioned).
* **Adaptive windows** $W(v)$ from local degree and local mean ρ.
* **Conservation meters** $E_Q,E_\Theta,E_C,E_\rho$ with leak accounting.
* Deterministic scheduler tie-breakers; seeded RNG everywhere.

---

# 1. Ontology & State

## 1.1 Graph substrate

A finite directed multigraph $\mathcal G=(V,E)$. Multiple edges between the same ordered vertex pair are allowed.

## 1.2 Locality

All state is stored on the **owning vertex or edge**. Rules may read:

* the local object's state;
* payloads delivered along incident edges;
* static metadata of incident edges/vertices.

No global variables, clock, or cross-graph scans.

## 1.3 Events & causal order

An **event** is delivery of a packet across an edge. Causal precedence $e_1\prec e_2$ holds if a directed path of events leads from $e_1$ to $e_2$. This relation is a partial order. No simultaneity structure is assumed.

## 1.4 Oscillators (optional)

Each vertex $v$ may host a phase $\theta_v\in [0,2\pi)$ with intrinsic $\omega_v$. An oscillator “fires” when $\theta_v\to 2\pi$, emitting on each outgoing edge. In the event-driven kernel we simply consume emissions already queued by prior arrivals; the oscillator is an interpretation layer.

## 1.5 State variables

### Vertex $v$

* **Depth:** $d(v)\in\mathbb Z_{\ge0}$ (last processed arrival-depth).
* **Window index:** $w(v)=\big\lfloor d(v)/W(v)\big\rfloor$.
* **Layer:** $\ell(v)\in\{\text{Q},\Theta,\text{C}\}$.
* **Q-state:** $\psi_v \in \mathbb C^D$ (normalized at window close).
* **Q-accumulator:** $\psi^\text{acc}_v\in \mathbb C^D$ (reset each window).
* **Θ-probabilities:** $p_v \in \Delta^{K-1}$ (simplex).
* **C-bit & confidence:** $(\text{bit}_v\in\{0,1\},\, \text{conf}_v\in[0,1])$ via majority buffer.
* **Fan-in this window:** $\Lambda_v\in\mathbb Z_{\ge0}$.
* **Meters:** $E_Q(v),\,E_\Theta(v),\,E_C(v)$ at window close (Sec. 7).
* **Ancestry fields (Bell):** rolling hash $h_v$ and phase-moment $m_v\in\mathbb R^3$.

### Edge $e:u\to v$

* **Base delay:** $d_0(e)\in\mathbb Z_{\ge1}$.
* **Effective delay:** $d_\text{eff}(e)\in\mathbb Z_{\ge1}$.
* **Coupling:** $\alpha_e\in\mathbb R_+$.
* **Phase & gauge:** $\phi_e, A_e \in \mathbb R$.
* **Unitary:** $U_e\in\mathbb C^{D\times D}$ (Q-layer).
* **Stress-energy:** $\rho_e\in\mathbb R_{\ge0}$.
* **Bridge strength (ε-pairs):** $\sigma_e\in\mathbb R_{\ge0}$ (0 if not a bridge).

### Packet payload

* **Layer payload:** one of $\psi$ (Q), $p$ (Θ), or bit (C).
* **Intensity (derived at delivery):** $I\in[0,1]$ (Sec. 4.4).
* **Arrival-depth:** $d_\text{arr}\in\mathbb Z_{\ge0}$.
* **Bell hidden var (optional):** $\lambda=(u,\zeta)$.
* **Optional ε-seed fields** (Sec. 6).

---

# 2. Temporal structure: arrival-depth & windows

**Arrival-depth** is the only operational time. The kernel keeps a priority queue keyed by

$$
(d_\text{arr},\, v_\text{dst},\, e,\, \text{seq})
$$

and repeatedly delivers the minimum. On delivery to $v$:

$$
d(v)\;\leftarrow\;\max\{d(v),\, d_\text{arr}\}.
$$

Each vertex has a **local window** length $W(v)$ (Sec. 5.1). The **window index** is $w(v)=\lfloor d(v)/W(v)\rfloor$. A **window closes** at $v$ when $w(v)$ increases; then:

* compute $E_Q$ and normalize $\psi_v$;
* reset $\psi^\text{acc}_v$ and $\Lambda_v$;
* evaluate meters / transitions.

---

# 3. Event life-cycle (pseudocode)

**Emission / scheduling:** when $u$ emits on edge $e:u\to v$,

$$
d_\text{arr}(e)= d(u) + d_\text{eff}(e),\quad \text{enqueue}(d_\text{arr}, v, e).
$$

**Delivery:** (to $v$, via $e$, payload $X$)

```
d(v) = max(d(v), d_arr)
Λ_v += 1

if ℓ(v) == Q:
    ψ_acc[v] += α_e · exp(i(φ_e + A_e)) · (U_e @ ψ_packet)
elif ℓ(v) == Θ:
    p_v ← normalize( p_v + α_e · p_packet )
else:  # C
    (bit_v, conf_v) ← majority_update(bit_v, conf_v, packet.bit)

# compute intensity I for ρ/delay update (Sec. 4.4)
I = intensity_from_layer(ℓ(v), e, payload)

# stress-energy and delay update on edge e
ρ_e, d_eff(e) ← ρ_update_and_delay(e, neighbors(e), I)

# ε-seed emit on Q delivery (Sec. 6)
if ℓ(v) == Q:
    emit_epsilon_seed(v, e, d_arr)

# update ancestry (Bell), then LCCM transitions (Sec. 5)
update_ancestry(v, payload)
maybe_transition_layer(v)
```

**Window close at $v$:**

```
E_Q(v) = ||ψ_acc[v]||^2
if E_Q(v) > 0: ψ[v] = ψ_acc[v] / sqrt(E_Q(v))
ψ_acc[v] = 0
E_Θ(v) = κ_Θ · (1 - H(p_v))     # H = Shannon entropy
E_C(v) = κ_C · conf_v
Λ_v = 0
```

---

# 4. Stress-energy & delays

## 4.1 Edge neighborhood (diffusion graph)

Edges are neighbors if they share a vertex (edge-adjacency). Let $\langle\rho\rangle_{\text{nbrs}(e)}$ be the mean over neighbors of $e$.

## 4.2 ρ-update (per delivery through $e$)

$$
\rho_e\ \leftarrow\ (1-\alpha_d-\alpha_{\text{leak}})\,\rho_e
\;+\; \alpha_d\,\big\langle\rho\big\rangle_{\text{nbrs}(e)}
\;+\; \eta\,I,
\quad \rho_e\ge 0.
$$

* $\alpha_d\in[0,1)$: edge-adjacent diffusion.
* $\alpha_{\text{leak}}\in[0,1)$: sink term.
* $\eta$: injection scale from **intensity** $I$.

## 4.3 Effective delay (saturating)

$$
d_\text{eff}(e)
= \max\!\Big\{1,\; d_0(e) + \big\lfloor \gamma \,\ln\!\big(1+\rho_e/\rho_0\big)\big\rfloor\Big\}.
$$

* $\gamma>0$, $\rho_0>0$ shape the saturation.
* Ensures integrality, positivity, and no singularities.

## 4.4 Layer intensities (bounded)
Intensity $I$ is taken from the current layer $\ell(v)$ at delivery.

* **Q:** $I = \|\,U_e\,\psi\,\|_2^2 \le 1$.
* **Θ:** $I = \|p\|_1$ (with $\sum p\le 1$ after mixing).
* **C:** $I = \text{bit}\in\{0,1\}$.

---

# 5. LCCM (Layered Causal Coherence Model)

## 5.1 Local window size

$$
W(v) = W_0
+ \big\lfloor \zeta_1 \,\ln\big(1+\deg(v)\big)
+ \zeta_2 \,\ln\big(1+\bar\rho_v/\rho_0\big) \big\rfloor,
$$

where $\bar\rho_v$ is the mean $\rho$ of edges incident to $v$. $W(v)\ge 1$.

* **Θ reset policy:** $\theta_\text{reset}\in\{\text{uniform},\text{renorm},\text{hold}\}$ chooses how $p_v$ is reset when the window closes.

## 5.2 Thresholds & timers

$$
N_\text{decoh}(v)= a\,W(v),\qquad
N_\text{recoh}(v)= b\,W(v),\qquad
0<b<a<1.
$$

* **Hold** $T_\text{hold}$ windows for Θ→Q recovery.
* **Dominance** for Θ→C uses entropy & confidence thresholds.

## 5.3 Transitions (all strictly local)

* **Q → Θ** (“**decoh_threshold**”): when $\Lambda_v \ge N_\text{decoh}(v)$ within the current window. $\psi_v$ becomes **frozen** (read-only); $p_v$ activates.
* **Θ → Q** (“**recoh_threshold**”): when $\Lambda_v \le N_\text{recoh}(v)$ for $T_\text{hold}$ consecutive windows **and** $E_Q(v)\ge C_\text{min}$.
* **Θ → C** (“**classical_dominance**”): when $H(p_v)\le H_\text{max}$ **and** bit dominance/confidence exceed thresholds for $T_\text{class}$ windows.
* (Optional C→Θ can be added; not required for v1.2.)

---

# 6. ε-Pairs: local correlation channels

## 6.1 Seeds (Q-layer only)

On Q-delivery at $v$, emit **seeds** along outgoing edges with:

* **Ancestry prefix**: match key from $h_v$ (first $L$ bits).
* **Angle tag**: local phase proxy $\theta_v$.
* **Expiry by depth:** $d_\text{exp} = d_\text{emit} + \Delta$.

A seed forwarded across an edge with $d_\text{eff}$ computes $d_\text{next}=d_\text{curr}+d_\text{eff}$ and **continues only if** $d_\text{next}\le d_\text{exp}$. Otherwise it **drops** (strict locality in arrival-depth).

## 6.2 Binding & bridges

Two seeds **collide** at a vertex and **bind** iff:

* both unexpired;
* ancestry prefixes match (length $L$);
* $|\theta_1-\theta_2|\le \theta_\text{max}$.

Binding creates a **transient bridge edge** with:

* initial $\sigma=\sigma_0$;
* local effective delay
  $d_\text{bridge}(u,v)=\max\!\Big\{1,\Big\lfloor\operatorname{median}\{d_\text{eff}(e): e\text{ incident to }u\text{ or }v\}\Big\rfloor\Big\}$;
* **stable synthetic id** (negative id space) for determinism/logs.

Bridges are scheduled **exactly like edges**.

## 6.3 σ-dynamics (use-dependent)

* On traversal: $\sigma\leftarrow (1-\lambda_\text{decay})\sigma + \sigma_\text{reinforce}$.
* Each window (idle): $\sigma\leftarrow (1-\lambda_\text{decay})\sigma$.
* Remove bridge when $\sigma<\sigma_\text{min}$.

All rules are local to the bridge endpoints.

---

# 7. Conservation (meters & balance)

At each window close:

* **Q meter:** $E_Q(v) = \|\psi^\text{acc}_v\|_2^2$.
* **Θ meter:** $E_\Theta(v) = \kappa_\Theta\,(1 - H(p_v))$.
* **C meter:** $E_C(v) = \kappa_C\cdot \text{conf}_v$.

Global/region balance (over any finite processed region $\mathcal R$):

$$
\Delta \!\!\sum_{v\in\mathcal R}\!\!(E_Q+E_\Theta+E_C)
\;+\;
\kappa_\rho\,\Delta\!\!\sum_{e\in\partial\mathcal R}\!\!\rho_e
\;\approx\; -\,\text{leak},
$$

where **leak** is controlled by $\alpha_{\text{leak}}$. This is a meter-level check, not an ontic identity.

---

# 8. Bell / Shared-Ancestry Selector (SAS)

## 8.1 Local ancestry fields

Each vertex maintains:

* **Hash** $h_v$ (rolling 256-bit; implementation uses 4×64).
* **Moment** $m_v\in\mathbb R^3$ (phase-direction statistics).

On each Q-layer delivery to $v$ with phase $\theta$ and arrival-depth $d_\text{arr}$:

$$
\begin{aligned}
\text{seed} &= (v \ll 32) \oplus d_\text{arr} \oplus \operatorname{round}(\theta\cdot 1000), \\
h_v &\leftarrow \operatorname{roll}(h_v) \oplus \operatorname{splitmix64}(\text{seed}), \\
m_v &\leftarrow 0.9\, m_v + 0.1\,(\cos\theta,\sin\theta,0).
\end{aligned}
$$

No updates occur on Θ or C deliveries.

## 8.2 Source hidden variable

At a pair source (vertex $S$), compute $\lambda=(u,\zeta)$ from $(h_S,m_S)$:

* Blend $m_S$ with a hash-derived direction, weights $\beta_m,\beta_h$, normalize to get **unit** $u$.
* $\zeta$ is a $[0,1)$ scalar from a split-mix of $h_S$.
  Both halves of the pair carry the same $\lambda$.

## 8.3 Detector setting (toggle MI)

At detector $D$ with ancestry $(h_D,m_D)$:

* **MI\_strict:** draw $a_D$ from a hash of $h_D$ (independent of $\lambda$).
* **MI\_conditioned:** draw $a_D$ from a vMF-like distribution centered on $m_D$ blended with $h_D$, with **concentration** $\kappa_a$ (strictly local but statistically correlated via shared ancestry).

## 8.4 Local readout

Outcome $b\in\{+1,-1\}$:

$$
b=\operatorname{sgn}\!\big(\langle a_D,\ R(h_D,\zeta)\,u\rangle + \xi\big),
$$

with local noise $\xi\sim\mathcal N(0,\sigma(\kappa_\xi))$. $R$ is a hash-controlled local rotation, no signaling.

**Prediction:**

* MI\_strict ⇒ CHSH $\le 2$.
* MI\_conditioned ⇒ tunable violations with $\kappa_a$, still no superluminal signaling.

---

# 9. Adaptive parameters & dimensionless groups

* **Windows:** $W_0,\zeta_1,\zeta_2$ (local topology & $\bar\rho$).
* **Decoherence:** $a,b,T_\text{hold},C_\text{min}$.
* **ρ/delay:** $\alpha_d,\alpha_{\text{leak}},\eta,\gamma,\rho_0$.
* **ε-pairs:** $\Delta, L, \theta_\text{max}, \sigma_0,\lambda_\text{decay},\sigma_\text{reinforce},\sigma_\text{min}$.
* **Bell:** $\beta_m,\beta_h,\kappa_a,\kappa_\xi$.

Useful **dimensionless ratios** for DOE:

$$
\frac{\Delta}{W_0},\ 
\frac{\alpha_d}{\alpha_{\text{leak}}},\ 
\gamma\ \text{vs}\ d_0,\ 
\eta W_0,\ 
\frac{a}{b},\ 
\frac{\sigma_\text{reinforce}}{\lambda_\text{decay}},\ 
\kappa_a,\ \kappa_\xi.
$$

---

# 10. Emergence sketch (physics at scale)

* **Wave transport:** Q-layer unitary accumulation over windows yields interference/diffraction; Θ/C layers provide local decoherence & classicalization under fan-in pressure.
* **Geometry:** sustained intensities raise $\rho$, which **delays** edges via $d_\text{eff}$, bending causal paths (a discrete lensing analog).
* **No-signaling:** all correlations arise via ancestry and depth-bounded ε-channels; no superluminal dependencies.

---

# 11. Minimal simulator interface (for reference)

* **Scheduler:** PQ keyed by $(d_\text{arr},v_\text{dst},e,\text{seq})$.
* **Window rule:** close when $w(v)$ increments; compute meters; reset accumulators.
* **Intensity:** derived per layer at delivery; drives $ρ$ and $d_\text{eff}$.
* **ε-pairs:** seeds with **expiry by depth**; local bind; transient bridges with σ-dynamics.
* **Bell:** ancestry updates; $\lambda$ at source; local setting & readout at detectors.

(These are implementation notes, not additional physics.)

---

# 12. Validation suite (expected outcomes)

1. **Two-path interference (Gate 1):**
   Visibility depends on relative phases; **intra-window arrival order does not** affect $E_Q$ at close.

2. **ρ→delay saturation (Gate 2):**
   Under sustained traffic, $d_\text{eff}$ rises smoothly (log-like), then relaxes when traffic stops.

3. **LCCM hysteresis (Gate 3):**
   Q→Θ at $\Lambda_v \ge aW$; Θ→Q below $bW$ sustained for $T_\text{hold}$; Θ→C under dominance criteria.

4. **ε-pairs locality (Gate 4):**
   Bridges form **only** within $\Delta$ (depth-TTL), and decay when unused (σ below $\sigma_\text{min}$).

5. **Conservation (Gate 5):**
   $E_Q+E_\Theta+E_C+\kappa_\rho\sum\rho$ remains within leak-tolerance; residual tracks $\alpha_{\text{leak}}$.

6. **Bell toggles (Gate 6):**
   MI\_strict ⇒ CHSH $\le 2$. MI\_conditioned ⇒ CHSH increases with $\kappa_a$; no signaling.

---

# 13. Defaults (illustrative, tune per graph)

* $W_0=4,\ \zeta_1=\zeta_2=0.3$.
* $a=0.7,\ b=0.4,\ T_\text{hold}=2,\ C_\text{min}=0.1$.
* $\alpha_d=0.1,\ \alpha_{\text{leak}}=0.01,\ \eta=0.2,\ \gamma=0.8,\ \rho_0=1.0$.
* $\Delta\approx 2W_0,\ L=16,\ \theta_\text{max}\approx \pi/12,\ \sigma_0=0.3,\ \lambda_\text{decay}=0.05,\ \sigma_\text{reinforce}=0.1,\ \sigma_\text{min}=10^{-3}$.
* $\beta_m=0.7,\ \beta_h=0.3,\ \kappa_a\in\{0,2,5,10\},\ \kappa_\xi=0.5$.

---

## Notes on strict locality

* No rule reads non-incident state or any global aggregate.
* Seeds use **only** their own `expiry_depth` and local edge $d_\text{eff}$.
* Bell draws depend **only** on local ancestry fields and local RNG seeded from local data.
* Scheduler order is purely $(d_\text{arr}, v_\text{dst}, e, \text{seq})$.

