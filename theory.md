## 1 Foundational Objects

| Symbol          | Meaning                        | Default / Domain  |
| --------------- | ------------------------------ | ----------------- |
| 𝒢 ≔ (V, E)     | Directed causal graph          | finite, loop-free |
| *v* ∈ V         | Node / oscillator / observer   | labelled `str`    |
| *e* = (u→v) ∈ E | Edge (causal channel)          | attributes below  |
| τ_v            | proper time accumulated at *v* | ℝ₊                |
| ψ_v ∈ ℂ²       | qubit state at *v*             | col-vector        |
| ρ_e            | stress–energy density on *e*   | ℝ₊                |
| Δt              | scheduler micro-tick           | user-set          |

Edge attributes

* **delay** d_e ∈ ℝ₊ (base lattice delay, ≥ 1 tick)
* **attenuation** α_e ∈ (0, 1]
* **phase_shift** φ_e ∈ [0, 2π) (static)
* **A_phase** A_e ∈ [0, 2π) (link gauge potential)
* **unitary type** u_id ∈ {0 ≡ 𝟙, 1 ≡ H}
* **ε-flag** ε_e ∈ {False, True} (entanglement pair)

---

## 2 Tick Dynamics

### 2.1 Emission rule

Every node *v* possesses an internal oscillator (frequency f_v) that fires when its phase crosses a threshold.
Upon firing at global tick *t*:

* Emit **one Tick object** down each outgoing edge e = (v→w).
* Each Tick carries:

  ```
  amplitude   a = 1
  phase       θ = φ_e + A_e        (± locked contribution if node is decoherent/classical)
  layer       ℓ  (from LCCM, Sec 4)
  trace_id    uuid4
  cumulative_delay = 0
  ```

### 2.2 Propagation delay

Effective delay on edge e:

<div align="center">

d_eff = ⟨delay scaling⟩ =
max (1, d_e · (1 + κ ρ_e))

</div>

κ is the stress–energy coupling (configurable).
`ρ_e` diffuses each scheduler step (Sec 3).

### 2.3 Complex amplitude update

For quantum-coherent nodes:

```
ψ_w  ←  ψ_w + α_e · e^{i θ} · U_e ψ_v
```

*U_e* ⟵ identity for `u_id = 0`, Hadamard H otherwise.

For **decoherent** nodes the complex vector is frozen; only the classical **probability vector** *p* propagates:

```
p_w ← p_w + α_e · p_v
```

For **classicalised** nodes a single eigenstate |0⟩ or |1⟩ (sampled by Born rule) is forwarded.

---

## 3 Stress–Energy Field

Amplitude energy on e is deposited each arrival:

```
ρ_e ← ρ_e + ‖amplitude‖²
```

At every macro tick the field diffuses:

```
ρ_e(t+1) = (1 − α) ρ_e(t) + α · mean_{e′∼e} ρ_{e′}(t)
```

α = `Config.density_diffusion_weight`.

---

## 4 Layered Causal Coherence Model (LCCM)

| Layer | Fan-in count | Behaviour                                         | NodeType      |
| ----- | ------------ | ------------------------------------------------- | ------------- |
| Q     | `< N_DECOH`  | unitary, coherent                                 | default       |
| Θ     | `= N_DECOH`  | **decoherent** (phase lock, probabilities frozen) | DECOHERENT    |
| C     | `≥ N_CLASS`  | **classical** (collapse to eigenstate)            | CLASSICALIZED |

`N_DECOH` (thermodynamic threshold) and `N_CLASS` (classical fallback) are tunable; default 3 / 6.

---

## 5 Proper-Time Law-Wave

Each node stores **τ_v**.  At scheduler step Δt:

<div align="center">

dτ_v = Δt (1 − κ ρ_local) √(1 − v²) ,

</div>

where *v*² = (Δx/Δt)² + (Δy/Δt)² in lattice units.
`ρ_local` can be sampled from incident edge densities or an external field.

---

## 6 Gauge Potential

An **A_phase** on edges realises a U(1) connection:

`θ_total = Σ (phase_shift + A_phase)` along a path.
Non-trivial loops ⇒ emergent holonomy reminiscent of electromagnetic flux.

---

## 7 Dynamic Entanglement (ε-pairs)

* A node flagged `cnot_source` marks its first two outbound edges as an ε-pair on every fire.
* Collapse rule: when node A collapses at tick *t*, its ε-partner B is instantly projected to the opposite eigenstate

```
ψ_B = [ψ_A1, ψ_A0]^T
```

Cross-edge Born-rule propagation preserves non-local correlations (CHSH S ≈ 2 √2).

---

## 8 Stress-Energy Horizons & Hawking Toy Model

Interior nodes are registered with energy *E*; each macro tick they emit an entangled Hawking pair with probability

```
P_emit = exp(−ΔE / T_H)
```

*ΔE* = `Config.hawking_delta_e`, *T_H* = `Config.hawking_temperature`.
Exterior entropy *S_out* obeys a qualitative Page curve:

```
S_out(t) = min(N_emit(t), N_total − N_emit(t))
```

---

## 9 Tensor Compression

Linear chains (> 4 edges) are represented as a **Matrix Product State**:

* Bond dimension χ ≤ `Config.chi_max` (default 16).
* On-chain unitary list {U₁…U_N} is contracted via SVD truncation; global norm maintained.

Error for 100-Hadamard chain with χ = 2 < 1 %.

---

## 10 Backend Acceleration

* CPU path: NumPy vectorised.
* GPU path: `complex_weighted_sum()` in `cupy_kernels.py` if `Config.backend=="cupy"` and CUDA present.
* Classical zones partitioned → Ray workers (fallback local).

---

## 11 Diagnostics & Sweeps

`tools/metrics.py` supplies:

* **Bell score** S(ε)
* **Interference visibility** V(fan_in)
* **Twin τ-ratio** R(v)

`tools/sweep.py` runs YAML-defined grids → CSV + PNG heat-maps.

---

## 12 Complete Parameter Table (excerpt)

| Name                       | Symbol | Default | Description             |
| -------------------------- | ------ | ------- | ----------------------- |
| `kappa`                    | κ      | 1.0 e-0 | Stress–delay coupling   |
| `N_DECOH`                  | —      | 3       | Decoherence fan-in      |
| `N_CLASS`                  | —      | 6       | Classicalisation fan-in |
| `chi_max`                  | χ_max | 16      | MPS bond dimension      |
| `backend`                  | —      | "cpu" | "cupy" ⇒ GPU kernels  |
| `density_diffusion_weight` | α      | 0.05    | ρ diffusion weight      |
| `hawking_temperature`      | T_H   | 2.0     | Horizon temperature     |
| `hawking_delta_e`          | ΔE     | 1.0     | Quantum energy          |

---

## 13 Emergent Phenomena Captured

| Phenomenon                               | Mechanism                                      |
| ---------------------------------------- | ---------------------------------------------- |
| **Relativistic time dilation**           | τ accumulation with velocity & density factors |
| **Gravitational lensing**                | Delay gradient from ρ-field wells              |
| **Wave-particle duality & interference** | ψ propagation + phase coherence                |
| **Decoherence & collapse**               | Layer thresholds (fan-in statistics)           |
| **Bell-inequality violation**            | ε-pair collapse + path-settings sampling       |
| **Hawking evaporation**                  | Horizon model emitting entangled pairs         |

---

*This document is the authoritative CWT + LCCM spec as of commit P-12.  Every subsequent code path or experiment should reference the definitions above.*

