# Causal Web Theory & Layered Causal Coherence Model (CWT/LCCM)

**Version 1.2 - Strict-Local**
**Status:** Research draft (2025-08-11).
**Principle:** Every rule reads/writes only local state and messages on finite causal paths. No global tick; no nonlocal action.

## What's new since 1.1

* Replaced global tick with **arrival-depth** scheduling (event-driven).
* Resolved Q-layer normalization: **store energy, normalize state** at window close.
* **LCCM hysteresis** with local fan-in counters & hold timers (Q<->Theta, Theta->C).
* **Stress-energy $\rho$** and **delay map** $d_\text{eff}$ via a saturating log law.
* **Dynamic epsilon-pairs** using **depth-based TTL** seeds and transient **bridges** with sigma-decay/reinforcement.
* **Bell / SAS**: strictly local, toggled **measurement-independence** (MI\_strict vs MI\_conditioned).
* **Adaptive windows** $W(v)$ from local degree and local mean $\rho$.
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

Each vertex $v$ may host a phase $\theta_v\in [0,2\pi)$ with intrinsic $\omega_v$. An oscillator "fires" when $\theta_v\to 2\pi$, emitting on each outgoing edge. In the event-driven kernel we simply consume emissions already queued by prior arrivals; the oscillator is an interpretation layer.

## 1.5 State variables

### Vertex $v$

* **Depth:** $d(v)\in\mathbb Z_{\ge0}$ (last processed arrival-depth).
* **Window index:** $w(v)=\big\lfloor d(v)/W(v)\big\rfloor$.
* **Layer:** $\ell(v)\in\{\text{Q},\Theta,\text{C}\}$.
* **Q-state:** $\psi_v \in \mathbb C^D$ (normalized at window close).
* **Q-accumulator:** $\psi^\text{acc}_v\in \mathbb C^D$ (reset each window).
* **Theta-probabilities:** $p_v \in \Delta^{K-1}$ (simplex).
* **C-bit & confidence:** $(\text{bit}_v\in\{0,1\},\, \text{conf}_v\in[0,1])$ via majority buffer.
* **Fan-in this window:** $\Lambda_v\in\mathbb Z_{\ge0}$.
* **Q-arrivals this window:** $\Lambda_v^{Q}\in\mathbb Z_{\ge0}$.
* **Meters:** $E_Q(v),\,E_\Theta(v),\,E_C(v),\,E_\rho(v)$ at window close (Sec. 7).
* **Ancestry fields (Bell):** rolling hash $h_v$ and phase-moment $m_v\in\mathbb R^3$.

### Edge $e:u\to v$

* **Base delay:** $d_0(e)\in\mathbb Z_{\ge1}$.
* **Effective delay:** $d_\text{eff}(e)\in\mathbb Z_{\ge1}$.
* **Coupling:** $\alpha_e\in\mathbb R_+$.
* **Phase & gauge:** $\phi_e, A_e \in \mathbb R$.
* **Unitary:** $U_e\in\mathbb C^{D\times D}$ (Q-layer).
* **Stress-energy:** $\rho_e\in\mathbb R_{\ge0}$.
* **Bridge strength (epsilon-pairs):** $\sigma_e\in\mathbb R_{\ge0}$ (0 if not a bridge).

### Packet payload

* **Layer payload:** one of $\psi$ (Q), $p$ (Theta), or bit (C).
* **Intensity (derived at delivery):** $I\in[0,1]$ (Sec. 4.4).
* **Arrival-depth:** $d_\text{arr}\in\mathbb Z_{\ge0}$.
* **Bell hidden var (optional):** $\lambda=(u,\zeta)$.
* **Optional epsilon-seed fields** (Sec. 6).

---

# 2. Temporal structure: arrival-depth & windows

**Arrival-depth** is the only operational time. The kernel keeps a priority queue keyed by

$$
(d_\text{arr},\, v_\text{dst},\, e,\, \text{seq})
$$

and repeatedly delivers the minimum. Deterministic seeded RNG uses only local fields and these keys. On delivery to $v$:

$$
d(v)\;\leftarrow\;\max\{d(v),\, d_\text{arr}\}.
$$

Each vertex has a **local window** length $W(v)$ (Sec. 5.1). The **window index** is $w(v)=\lfloor d(v)/W(v)\rfloor$. A **window closes** at $v$ when $w(v)$ increases; then:

* compute $E_Q$ and normalize $\psi_v$;
* reset $\psi^\text{acc}_v$ and $\Lambda_v$;
* evaluate meters / transitions;
* while in C, retain $(\text{bit}_v, \text{conf}_v)$ across windows, clearing only the majority buffer. These fields reset on C→Θ transitions.

---

# 3. Event life-cycle (pseudocode)

**Emission / scheduling:** when $u$ emits on edge $e:u\to v$,

$$
d_\text{arr}(e)= d(u) + d_\text{eff}(e),\quad \text{enqueue}(d_\text{arr}, v, e).
$$

**Delivery:** (to $v$, via $e$, payload $X$)

```
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
 ```

**Window close at $v$:**

```
E_Q(v) = ||psi_acc[v]||^2
if E_Q(v) > 0: psi[v] = psi_acc[v] / sqrt(E_Q(v))
psi_acc[v] = 0
E_Theta(v) = kappa_Theta * (1 - H(p_v))     # H = Shannon entropy
E_C(v) = kappa_C * conf_v
Lambda_v = 0
if Lambda_v_Q == 0:
    m_v = normalize((1 - delta_m) * m_v)
Lambda_v_Q = 0
```

---

# 4. Stress-energy & delays

## 4.1 Edge neighborhood (diffusion graph)

Edges are neighbors if they share a vertex (edge-adjacency). Let $\langle\rho\rangle_{\text{nbrs}(e)}$ be the mean over neighbors of $e$.

## 4.2 rho-update (per delivery through $e$)

$$
\rho_e\ \leftarrow\ (1-\alpha_d-\alpha_{\text{leak}})\,\rho_e
\;+\; \alpha_d\,\big\langle\rho\big\rangle_{\text{nbrs}(e)}
\;+\; \eta\,I,
\quad \rho_e\ge 0.
$$

* $\alpha_d\in[0,1)$: edge-adjacent diffusion.
* $\alpha_{\text{leak}}\in[0,1)$: sink term.
* $\eta$: injection scale from **intensity** $I$.

*Implementation knob*: choose the injection set with
`inject_mode \in \{\text{"incoming"},\text{"incident"},\text{"outgoing"}\}`
(default `incoming`). Default is `inject_mode="incoming"`, updating only the delivered edge using the per-delivery intensity from Sec. 4.4. Other modes (`incident`,`outgoing`) are implementation variants for ablation, not the default.
For `inject_mode \neq "incoming"` the intensity is the mean of per-packet $\|p\|_1$ over the window/batch.

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
* **Theta:** $I = \|p\|_1$ (with $\sum p\le 1$ after mixing).  For
  ``inject_mode="incoming"`` this per-packet value applies to each
  delivered edge individually.  Multi-packet modes (``incident`` or
  ``outgoing``) instead use the **mean** of per-packet $\|p\|_1$ over
  the window or batch to avoid saturation under high fan-in.
* **C:** $I = \text{bit}\in\{0,1\}$.

---

# 5. LCCM (Layered Causal Coherence Model)

## 5.1 Local window size

Let $\deg_\text{inc}(v)=\deg_\text{in}(v)+\deg_\text{out}(v)+\deg_\text{bridge}(v)$. Let $\bar\rho_v$ be the current mean of $\rho_e$ over edges incident to $v$, recomputed when the window closes. Then

$$
W(v)=W_0+\Big\lfloor \zeta_1\ln\big(1+\deg_\text{inc}(v)\big)+\zeta_2\ln\big(1+\bar\rho_v/\rho_0\big)\Big\rfloor,\quad W(v)\ge 1.
$$

* **Theta reset policy:** $\theta_\text{reset}\in\{\text{uniform},\text{renorm},\text{hold}\}$ chooses how $p_v$ is reset when the window closes (default $\text{renorm}$).

## 5.2 Thresholds & timers

$$
N_\text{decoh}(v)= a\,W(v),\qquad
N_\text{recoh}(v)= b\,W(v),\qquad
0<b<a<1.
$$

* **Hold** $T_\text{hold}$ windows for Theta->Q recovery.
* **Dominance** for Theta->C uses entropy & confidence thresholds.

## 5.3 Transitions (all strictly local)

* **Q -> Theta** ("**decoh_threshold**"): when $\Lambda_v \ge N_\text{decoh}(v)$ within the current window. $\psi_v$ becomes **frozen** (read-only); $p_v$ activates.
* **Theta -> Q** ("**recoh_threshold**"): when $\Lambda_v \le N_\text{recoh}(v)$ for $T_\text{hold}$ consecutive windows **and** $E_Q(v)\ge C_\text{min}$.
* **Theta -> C** ("**classical_dominance**"): when $H(p_v)\le H_\text{max}$ and $\text{bit\_frac}\ge f_\text{min}$ and $\text{conf}_v\ge \text{conf}_\text{min}$ for $T_\text{class}$ windows.
* (Optional C->Theta can be added; not required for v1.2.)
While in layer C the bit and confidence persist across window boundaries, with only the majority buffer cleared each window. Upon leaving C these fields are reset.

---

# 6. Epsilon-Pairs: local correlation channels

## 6.1 Seeds (Q-layer only)

On Q-delivery at $v$, emit **seeds** along outgoing edges with:

* **Ancestry prefix**: match key from $h_v$ (first $L$ bits).
* **Angle tag**: local phase proxy $\theta_v$.
* **Expiry by depth:** $d_\text{exp} = d_\text{emit} + \Delta$.

*Implementation note*: Default emission is **one seed per (v, window)** using $\theta_v=\operatorname{atan2}(m_{v,y},m_{v,x})$. The seed depth for this emission equals the maximum arrival depth seen in the window, i.e. the depth of the last processed delivery. An optional `emit_per_delivery` mode emits per Q-arrival. Implementations may cap the seed pool per vertex at $N_\text{seed}$ (e.g., 64) to avoid unbounded growth.

A seed forwarded across an edge uses that edge's current $d_\text{eff}$: $d_\text{next}=d_\text{curr}+d_\text{eff}$ and **continues only if** $d_\text{next}\le d_\text{exp}$. TTL advances by each traversed edge.

## 6.2 Binding & bridges

Two seeds **collide** at a vertex and **bind** iff:

* both unexpired;
* ancestry prefixes match (length $L$);
* $|\theta_1-\theta_2|\le \theta_\text{max}$.

Binding creates a **transient bridge edge** with:

* initial $\sigma=\sigma_0$;
* local effective delay
  $d_\text{bridge}(u,v)=\max\!\Big\{1,\Big\lfloor\operatorname{median}\{d_\text{eff}(e): e\text{ incident to }u\text{ or }v\}\Big\rfloor\Big\}$ (computed from current incident $d_\text{eff}$ at bind time);
* **stable synthetic id** (negative id space) for determinism/logs.

Bridges are scheduled **exactly like edges**.

## 6.3 sigma-dynamics (use-dependent)

* On traversal: $\sigma\leftarrow (1-\lambda_\text{decay})\sigma + \sigma_\text{reinforce}$.
* Each window (idle): $\sigma\leftarrow (1-\lambda_\text{decay})\sigma$.
* Remove bridge when $\sigma<\sigma_\text{min}$.

All rules are local to the bridge endpoints.

---

# 7. Conservation (meters & balance)

At each window close:

* **Q meter:** $E_Q(v) = \|\psi^\text{acc}_v\|_2^2$.
* **Theta meter:** $E_\Theta(v) = \kappa_\Theta\,(1 - H(p_v))$.
* **C meter:** $E_C(v) = \kappa_C\cdot \text{conf}_v$.
* **ρ meter:** $E_\rho(v) = \kappa_\rho\, \bar\rho_v$.

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

## 8.1 Local ancestry fields (Q-arrivals only)
Each vertex keeps a rolling hash $h_v=(h_0,h_1,h_2,h_3)\in(\mathbb{Z}_{2^{64}})^4$ and a unit moment $m_v\in\mathbb R^3$.

From the delivered packet $\psi$ on edge $e$ with phases $\phi_k$ and weights $w_k=|\psi_k|^2/(\sum|\psi|^2+\varepsilon)$, define $\tilde\phi_k=\phi_k+\phi_e+A_e$,
$z=\sum_k w_k\,e^{i\tilde\phi_k}$, mean direction $\mu=\arg z$, concentration $\kappa=|z|\in[0,1]$, and $u_\text{local}=[\cos\mu,\sin\mu,\kappa]$.

**Moment EMA:** $m_v\leftarrow \mathrm{normalize}\big((1-\beta_m)\,m_v+\beta_m\,u_\text{local}\big)$.
**Window decay:** if the prior window had $\Lambda_v^{Q}=0$, decay $m_v\leftarrow \mathrm{normalize}\big((1-\delta_m)\,m_v\big)$.

**Rolling hash (splitmix64 lanes, strictly local):**
$h_0\leftarrow \mathrm{smix}\big(h_0\oplus v \oplus (d_\text{arr}\ll 1)\big)$,
$h_1\leftarrow \mathrm{smix}\big(h_1\oplus e \oplus (\text{seq}\ll 1)\big)$,
$h_2\leftarrow \mathrm{smix}\big(h_2\oplus \mathrm{bits}(\mu)\big)$,
$h_3\leftarrow \mathrm{smix}\big(h_3\oplus \mathrm{bits}(\kappa)\big)$.
**Seed prefix:** first $L$ MSBs of $h_0$.

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

* MI\_strict => CHSH $\le 2$.
* MI\_conditioned => tunable violations with $\kappa_a$, still no superluminal signaling.

---

# 9. Adaptive parameters & dimensionless groups

* **Windows:** $W_0,\zeta_1,\zeta_2$ (local topology & $\bar\rho$).
* **Decoherence:** $a,b,T_\text{hold},C_\text{min}$.
* **rho/delay:** $\alpha_d,\alpha_{\text{leak}},\eta,\gamma,\rho_0$.
* **epsilon-pairs:** $\Delta, L, \theta_\text{max}, \sigma_0,\lambda_\text{decay},\sigma_\text{reinforce},\sigma_\text{min}$.
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

* **Wave transport:** Q-layer unitary accumulation over windows yields interference/diffraction; Theta/C layers provide local decoherence & classicalization under fan-in pressure.
* **Geometry:** sustained intensities raise $\rho$, which **delays** edges via $d_\text{eff}$, bending causal paths (a discrete lensing analog).
* **No-signaling:** all correlations arise via ancestry and depth-bounded epsilon-channels; no superluminal dependencies.

---

# 11. Minimal simulator interface (for reference)

* **Scheduler:** PQ keyed by $(d_\text{arr},v_\text{dst},e,\text{seq})$.
* **Window rule:** close when $w(v)$ increments; compute meters; reset accumulators.
* **Intensity:** derived per layer at delivery; drives $\rho$ and $d_\text{eff}$.
* **epsilon-pairs:** seeds with **expiry by depth**; local bind; transient bridges with sigma-dynamics.
* **Bell:** ancestry updates; $\lambda$ at source; local setting & readout at detectors.
* **Interface knobs:** `emit_per_delivery`, `inject_mode`.

(These are implementation notes, not additional physics.)

---

# 12. Validation suite (expected outcomes)

1. **Two-path interference (Gate 1):**
   Visibility depends on relative phases; **intra-window arrival order does not** affect $E_Q$ at close.

2. **rho->delay saturation (Gate 2):**
   Under sustained traffic, $d_\text{eff}$ rises smoothly (log-like), then relaxes when traffic stops.

3. **LCCM hysteresis (Gate 3):**
   Q->Theta at $\Lambda_v \ge aW$; Theta->Q below $bW$ sustained for $T_\text{hold}$; Theta->C under dominance criteria.

4. **epsilon-pairs locality (Gate 4):**
   Bridges form **only** within $\Delta$ (depth-TTL), and decay when unused (sigma below $\sigma_\text{min}$).

5. **Conservation (Gate 5):**
   $E_Q+E_\Theta+E_C+\kappa_\rho\sum\rho$ remains within leak-tolerance; residual tracks $\alpha_{\text{leak}}$.

6. **Bell toggles (Gate 6):**
   MI\_strict => CHSH $\le 2$. MI\_conditioned => CHSH increases with $\kappa_a$; no signaling.

7. **Ancestry determinism (Gate 7):**
   Identical local Q-delivery sequences at $v$ produce identical $(h_v,m_v)$. Shuffling remote events leaves $(h_v,m_v)$ unchanged.

---

# 13. Defaults (illustrative, tune per graph)

* $W_0=4,\ \zeta_1=\zeta_2=0.3$.
* $a=0.7,\ b=0.4,\ T_\text{hold}=2,\ C_\text{min}=0.1$.
* $\alpha_d=0.1,\ \alpha_{\text{leak}}=0.01,\ \eta=0.2,\ \gamma=0.8,\ \rho_0=1.0$.
* $\Delta\approx 2W_0,\ L=16,\ \theta_\text{max}\approx \pi/12,\ \sigma_0=0.3,\ \lambda_\text{decay}=0.05,\ \sigma_\text{reinforce}=0.1,\ \sigma_\text{min}=10^{-3}$.
* $\kappa_a\in\{0,2,5,10\},\ \kappa_\xi=0.5$.
* $H_\text{max}=0.2,\ f_\text{min}=0.6,\ \text{conf}_\text{min}=0.7,\ T_\text{class}=2$.
* Ancestry: $\beta_m=0.1,\ \beta_h=0.3,\ \delta_m=0.02$.

---

## Notes on strict locality

* No rule reads non-incident state or any global aggregate.
* Seeds use **only** their own `expiry_depth` and local edge $d_\text{eff}$.
* Bell draws depend **only** on local ancestry fields and local RNG seeded from local data.
* Scheduler order is purely $(d_\text{arr}, v_\text{dst}, e, \text{seq})$.

