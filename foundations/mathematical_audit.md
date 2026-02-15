# Technical Audit v1.5: Mathematical & Statistical Blueprint

**Document Date:** February 15, 2026  
**Audited Artifact:** `foundations/code/Proyecto_Ising_MonteCarlo.ipynb`  
**Audit Focus:** Physics correctness (Hamiltonian, PBC, Metropolis), numerical stability, and a C++-ready mathematical template.

---

## 0) Scope: Code Paths Audited

The notebook contains multiple, partially overlapping implementations of the Ising physics. This audit covers *all* relevant formula sites:

1. **Numba kernels (early section)**
   - `metropolis_step_numba`: ΔE and acceptance
   - `calculate_energy_numba`: total energy
   See `delta_E = 2.0 * spin * (J * neighbor_sum + H)` and the acceptance line in the early Numba cell.

2. **Pure-Python class (mid section)**
   - `IsingLattice.calculate_site_energy`: single-site energy contribution
   - `IsingLattice.calculate_total_energy`: total energy (with 1/2 correction)
   - `IsingLattice.metropolis_step`: Metropolis update by recomputing local energy before/after the flip

3. **Numba backend (optimized section)**
   - `_njit_metropolis_sweeps`: ΔE and acceptance
   - `_njit_total_energy`: total energy computed from neighbor table
   - Neighbor-table generators: `_neighbors_1d`, `_neighbors_square`, `_neighbors_hex`, `_neighbors_bcc`

4. **Auxiliary PBC demonstration**
   - `get_neighbors_2d`: PBC math in explicit coordinate form (square lattice)

---

## 1) Hamiltonian Consistency: $J$ and $H$ Convention

### 1.1 Standard reference Hamiltonian

The standard Ising Hamiltonian used in computational physics (with $k_B = 1$ unless stated otherwise) is:

$$
\mathcal{H}(\{s\}) = -J\sum_{\langle i,j\rangle} s_i s_j - H_0 \sum_i s_i
$$

where $s_i \in \{-1,+1\}$ (and in this notebook, *also* $s_i = 0$ is used to represent a diluted/vacant site).

- **Ferromagnet:** $J>0$ favors parallel alignment.
- **Antiferromagnet:** $J<0$ favors anti-parallel alignment.

### 1.2 What the notebook implements

Across the audited implementations, the energy and ΔE formulas match the reference Hamiltonian:

**(A) Local/site energy**

The pure-Python `calculate_site_energy` uses:

$$
E_i = -J\,s_i\sum_{j\in nn(i)} s_j - H_0 \,s_i
$$

with a guard that returns 0 if the site is vacant (`spin == 0`). This is a standard “diluted-site trick”: bonds connected to a vacancy contribute 0 because $s_j=0$.

**(B) Total energy**

The pure-Python `calculate_total_energy` sums `calculate_site_energy` over all sites and divides by 2:

$$
E = \frac{1}{2}\sum_i E_i
$$

This is mathematically equivalent to the pair sum $-J\sum_{\langle i,j\rangle} s_is_j$ because the neighbor-sum approach counts each undirected bond twice.

**(C) Metropolis energy change (ΔE)**

Both Numba kernels implement:

$$
\Delta E = 2 s_i\left(J\sum_{j\in nn(i)} s_j + H_0 \right)
$$

This is the correct ΔE for the Hamiltonian above when proposing a single-spin flip $s_i \to -s_i$.

**Conclusion:** **The $J$ and $H$ sign convention is consistent with the standard Ising Hamiltonian.**

---

## 2) ΔE Formula Verification (Metropolis physics)

### 2.1 Derivation (canonical reference)

Let $h_i$ be the local effective field:

$$
h_i = J\sum_{j\in nn(i)} s_j + H_0
$$

Then energy terms involving $s_i$ are $-s_i h_i$ (up to the bond double-counting convention). Flipping $s_i\to -s_i$ changes that contribution by:

$$
\Delta E = (-(-s_i)h_i) - (-(s_i)h_i) = 2 s_i h_i
$$

which matches the code’s ΔE.

### 2.2 Dilution ($s=0$)

The Numba kernels select **only occupied sites** for updates (they sample from `occ_idx` / `occupied_sites`), so they never attempt to flip vacancies. This preserves the intended diluted Ising dynamics.

---

## 3) Total Energy Formula Verification (double-counting and correctness)

### 3.1 Neighbor-sum energy and the factor 1/2

For an undirected lattice graph with symmetric neighbor lists, define:

$$
\text{pair\_sum} = \sum_i s_i\sum_{j\in nn(i)} s_j
$$

Each bond $\langle i,j\rangle$ appears twice in this expression (once as $i\to j$ and once as $j\to i$), therefore:

$$
\sum_{\langle i,j\rangle} s_is_j = \frac{1}{2}\,\text{pair\_sum}
$$

So total energy is:

$$
E = -\frac{J}{2}\,\text{pair\_sum} - H_0 \sum_i s_i
$$

This is exactly what `_njit_total_energy` returns.

### 3.2 **Critical requirement**: symmetric neighbor lists

The factor $1/2$ is correct **only if** the neighbor table is symmetric (undirected bonds represented twice).

- For the square lattice (`_neighbors_square`) and 1D (`_neighbors_1d`) and BCC (`_neighbors_bcc`), the neighbor construction is symmetric.
- For the hexagonal lattice (`_neighbors_hex`), the current implementation is labeled *“simplified”* and **does not obviously enforce symmetry** (see Section 4.3). This is the largest physics-risk found in the topology layer.

---

## 4) Topology Math & PBC Audit for $z=2,3,4,8$

### 4.1 PBC math: what “no coordinate leaks” means

For a torus topology (periodic boundary conditions), all indices must map to the valid range:

- 1D: $i \in \{0,1,\dots,L-1\}$
- 2D: $(i,j) \in \{0,\dots,L-1\}^2$
- 3D: $(i,j,k) \in \{0,\dots,L-1\}^3$

The standard PBC mapping is:

$$
\mathrm{wrap}(x) = (x \bmod L)
$$

In Python/NumPy, `% L` already maps negative values into $[0,L-1]$. The code’s frequent use of `(x + L) % L` is conservative and correct.

---

### 4.2 $z=2$ (Chain 1D)

Neighbor rule implemented in `_neighbors_1d`:

$$
\text{nn}(i) = \{(i-1)\bmod L,\ (i+1)\bmod L\}
$$

Audit:
- Wrap logic uses `(i - 1 + L) % L` and `(i + 1) % L` → ✅ correct.
- The neighbor table is symmetric (if $j$ is neighbor of $i$, $i$ is neighbor of $j$) → ✅ correct.

---

### 4.3 $z=3$ (Hexagonal 2D) — **Highest Risk Area**

Current `_neighbors_hex` returns 3 neighbors per site:

- right: $(i, j+1)$
- down: $(i+1, j)$
- down-left: $(i+1, j-1)$

(all wrapped mod $L$).

Mathematical concerns:

1. **Symmetry / undirectedness is not guaranteed**
   - Example: if site $A=(i,j)$ includes “right neighbor” $B=(i,j+1)$, the neighbor list of $B$ does **not** include $A$ (it includes $(i,j+2)$ as its “right”).
   - Therefore, the adjacency is *directed* (or “forward-only”), not an undirected hex lattice.

2. **Energy formula mismatch risk**
   - `_njit_total_energy` uses $-\tfrac{J}{2}\,\text{pair\_sum}$, which assumes every bond is counted twice.
   - With a forward-only neighbor list, each bond is counted once; dividing by 2 **undercounts**.

3. **ΔE mismatch risk**
   - The ΔE formula used in Metropolis assumes the Hamiltonian is built from the *same* undirected neighbor set.
   - If the neighbor list is directed/forward-only, the computed ΔE may not match the true energy difference.

**Audit verdict for z=3:**
- PBC wrap math itself (`jp=(j+1)%L`, `jm=(j-1+L)%L`, `ip=(i+1)%L`) is ✅ correct.
- **Graph/topology correctness is ⚠️ not validated**. The current simplified neighbor set likely does not represent a proper hex/honeycomb topology with consistent undirected bonds.

**Actionable recommendation (for C++ migration):**
- Decide the intended lattice: *triangular (z=6)* vs *honeycomb/hexagonal (z=3)*.
- Implement a neighbor function that guarantees **undirected bonds** (symmetric neighbor lists) *or* explicitly store **unique edges** and remove the 1/2 factor.

---

### 4.4 $z=4$ (Square lattice 2D)

Neighbor rule implemented in `_neighbors_square`:

$$
\text{nn}(i,j)=\{(i-1,j),(i+1,j),(i,j-1),(i,j+1)\}\quad\text{with PBC}
$$

Flattening convention:

$$
\text{idx}(i,j)=iL+j
$$

Audit:
- Wrap uses `(i±1+L)%L` and `(j±1+L)%L` → ✅ correct.
- Symmetry holds (up/down, left/right are reciprocal) → ✅ correct.

The notebook also includes a pedagogical coordinate version `get_neighbors_2d(i,j,L)` using `(i+di) % L`, `(j+dj) % L`, which is mathematically equivalent and correctly wraps negative steps.

---

### 4.5 $z=8$ (BCC lattice 3D)

Neighbor rule implemented in `_neighbors_bcc`:

$$
\text{nn}(i,j,k)=\{(i\pm1,j\pm1,k\pm1)\}\quad (8\ \text{combinations})
$$

Flattening convention:

$$
\text{idx}(i,j,k)=iL^2+jL+k
$$

Audit:
- Wrap uses `(coord + d + L) % L` with $d\in\{-1,+1\}$ → ✅ correct.
- Symmetry holds because the inverse offsets exist → ✅ correct.
- Coordination number = 8 everywhere → consistent with nearest neighbors on a BCC represented as diagonal connections on a simple cubic index set.

---

## 5) Numerical Stability of Metropolis Acceptance: `exp(-beta * dE)`

### 5.1 Underflow/overflow analysis

The acceptance probability for $\Delta E>0$ is:

$$
P_{acc}=\exp(-\beta\Delta E)
$$

The code uses a safe short-circuit:
- If $\Delta E \le 0$: accept without calling `exp`
- Else: evaluate `exp(-beta * dE)`

**Overflow risk:**
- Overflow occurs if `exp(x)` with $x \gtrsim 709$ in float64.
- Here we compute `exp(-beta*dE)` only when $\beta\Delta E > 0$, so the exponent is negative. This cannot overflow.

**Underflow risk:**
- Underflow occurs when $\beta\Delta E$ is large and positive: `exp(-beta*dE) → 0`.
- This is physically correct (almost sure rejection) and is numerically benign.

**Edge case:** $T \to 0$.
- In the optimized class backend, beta is set as `beta = 1.0 / T if T > 0 else 1e12`.
- This drives `exp(-beta*dE)` to 0 for any $\Delta E>0$, producing greedy descent dynamics, which is consistent with a $T=0$ limit.

### 5.2 Recommendations for robust C++

- Maintain the branch structure to avoid `exp` for $\Delta E\le 0$.
- Optionally guard:
  - float64: if $\beta\Delta E > 745$, set `p=0`
  - float32: if $\beta\Delta E > 88`, set `p=0`

This is not a “log-sum-exp” situation; we are computing a scalar acceptance probability, not a sum of exponentials.

---

## 6) LUT (Look-Up Table) Viability for C++

### 6.1 Why a LUT works for Ising single-spin flips

For a fixed temperature and external field, the only variable inside ΔE is the neighbor sum:

$$
\Delta E = 2s_i(J\,n_b + H), \quad n_b=\sum_{j\in nn(i)} s_j
$$

For a fixed coordination number $z$, the neighbor sum $n_b$ can only take values in $[-z,\dots,z]$ in integer steps (steps of 2 in the undiluted ±1 case; steps of 1 when vacancies $s=0$ are allowed).

Thus, for each topology, the number of possible $n_b$ values is small:

- z=2: 5 values (−2..2)
- z=3: 7 values (−3..3)
- z=4: 9 values (−4..4)
- z=8: 17 values (−8..8)

### 6.2 LUT definition

Define the local field:

$$
h(n_b) = J\,n_b + H
$$

Then if $s_i h > 0$ (spin aligned with local field), the flip is energetically unfavorable and:

$$
P_{acc} = \exp(-2\beta |h(n_b)|)
$$

If $s_i h \le 0$, accept with probability 1.

A LUT can store $w[n_b] = \exp(-2\beta|J n_b + H|)$ for the integer range $n_b\in[-z,z]$.

### 6.3 Practical constraints

- LUT is valid and beneficial **when $T$, $J$, and $H$ are fixed** during a run (which is the typical case in these simulations).
- If $H$ varies continuously within a run (e.g., time-dependent field), you would need to rebuild the LUT each time $H$ changes.

### 6.4 Scientific acceptability

A LUT does not approximate physics; it is mathematically identical to calling `exp` repeatedly, just faster.

---

## 7) Randomness Audit: Proposal Symmetry and Statistical Independence

### 7.1 Proposal distribution (random site selection)

The Numba kernels choose a random occupied site:
- Early kernel: `site = occupied_sites[np.random.randint(0, len(occupied_sites))]`
- Optimized kernel: `i = occ_idx[np.random.randint(n_occ)]`

This yields a **uniform proposal distribution on occupied sites** (with replacement). This is a standard random-sequential update.

### 7.2 Independence between site selection and acceptance uniform

The algorithm consumes two independent uniforms *in sequence*:
1. an integer RNG for the site index
2. a floating RNG for the acceptance test

They are successive outputs from the same pseudorandom generator. For a modern RNG (NumPy’s PCG family in Python; Numba’s internal RNG in `njit`), these are treated as statistically independent for Monte Carlo purposes.

**Important correctness condition:** independence is not strictly required; what is required is:
- symmetric proposal probabilities (true here)
- acceptance consistent with detailed balance using *some* random uniform in (0,1)

### 7.3 Parallel RNG caveat (Numba `prange`)

The notebook includes a `@njit(parallel=True)` simulation across independent H values. In parallel Monte Carlo, the main risk is **correlated random streams across threads** if RNG state is not handled per-thread/per-replica.

For C++ migration (especially OpenMP/TBB/CUDA), treat this as a design requirement:
- each replica/thread gets an independent RNG stream (e.g., counter-based RNG or distinct seeds via splitmix64)

---

## 8) C++ Mathematical Blueprint: `delta_E` / acceptance template (bit-packing ready)

This section is written as a physics-faithful mathematical template, with an eye toward bit-packed spin storage.

### 8.1 Data model assumptions

- Spins stored as bits (packed) or int8:
  - For bit-pack: `bit = 0/1` maps to $s\in\{-1,+1\}$ via `s = bit ? +1 : -1`.
  - Vacancies (dilution) are best handled by a separate occupancy bitset; if using the notebook’s trick $s=0$, you cannot represent that directly in 1-bit storage.

- Neighbor indices precomputed as `int32 neighbors[N][z]`.

### 8.2 Minimal-FP ΔE template (fixed J,H,T)

**Goal:** compute acceptance with minimal floating operations.

Let `nb_sum` be an integer in [-z..z].

1) Compute local field:

$$
h = J\,nb\_sum + H
$$

2) Energetic sign test (no exp):

- If $s_i h \le 0$: accept (ΔE ≤ 0)
- Else accept with $\exp(-2\beta|h|)$

### 8.3 LUT-accelerated pseudocode

```cpp
// Precompute per (beta, J, H, z)
// w[nb_sum + z] = exp(-2 * beta * abs(J * nb_sum + H))
std::array<double, 2*Z_MAX + 1> w;

inline int spin_pm1(Bitset spins, int i) {
    // Example mapping: bit=1 => +1, bit=0 => -1
    return spins.get(i) ? +1 : -1;
}

inline int neighbor_sum_pm1(const Bitset& spins, const int* nbrs_i, int z) {
    int sum = 0;
    for (int k = 0; k < z; ++k) {
        const int j = nbrs_i[k];
        sum += spins.get(j) ? +1 : -1;
    }
    return sum;
}

inline bool metropolis_accept(
    int s_i, int nb_sum, double beta, double J, double H,
    const std::array<double, 2*Z_MAX + 1>& w, RNG& rng, int z
) {
    // local field
    const double h = J * double(nb_sum) + H;

    // ΔE = 2*s_i*h
    // If ΔE <= 0 accept
    if (double(s_i) * h <= 0.0) return true;

    // Else accept with exp(-beta*ΔE) = exp(-2*beta*abs(h))
    const double p = w[nb_sum + z];
    return rng.uniform01() < p;
}
```

### 8.4 Bit-packing + dilution

If dilution is required (q<1), you have two choices:

1) **Separate occupancy bitset** (recommended for bit-packed spins)
   - Occupancy decides if a site participates
   - Spin bits are defined only for occupied sites

2) **Tri-state storage** (int8: −1,0,+1)
   - Simpler, matches notebook
   - Not 1-bit, but still cache-friendly

### 8.5 Topology note for C++

For z=3 (hex/honeycomb), do not port the simplified neighbor definition as-is without first enforcing:
- undirected symmetry **or** unique-edge energy counting (no 1/2 factor)
- a consistent ΔE computed from the true Hamiltonian

---

## 9) Validation Tests to Run Before/After C++ Port

These are mathematical invariants that catch subtle topology and ΔE mismatches:

1. **Energy/ΔE consistency test**
   - For random configuration, pick random occupied site i.
   - Compute ΔE via local neighbor sum.
   - Compute energy difference by explicit total energy recomputation:
     $$\Delta E_{check} = E(\text{flipped}) - E(\text{original})$$
   - Verify $\Delta E = \Delta E_{check}$ for many trials.

2. **Neighbor symmetry test (if using 1/2 factor)**
   - Verify: $j \in nn(i) \Rightarrow i \in nn(j)$ for all sites.

3. **PBC bounds test**
   - Verify all neighbor indices are within `[0, N-1]`.

4. **Detailed balance sanity**
   - For small L (e.g., L=4), compare histogram of energies/magnetizations against known exact enumeration (2D square) at a few T.

---

## Executive Verdict

- **Hamiltonian and ΔE physics:** ✅ consistent with the standard Ising convention for z=2,4,8 implementations.
- **PBC / torus mapping (`% L`):** ✅ mathematically correct and safe against coordinate leaks.
- **Numerical stability of `exp(-beta*dE)`:** ✅ safe from overflow due to short-circuit; underflow is benign.
- **LUT feasibility:** ✅ mathematically exact and highly viable for C++ given discrete neighbor sums.
- **Randomness independence:** ✅ acceptable for Metropolis; treat parallel RNG streams as a C++ design requirement.
- **Hexagonal z=3 topology:** ⚠️ current “simplified” neighbor construction is the dominant correctness risk; verify symmetry/energy-counting before porting.
