# Technical Audit v1.5: Scalability & Bottleneck Report

**Document Date:** February 15, 2026
**Audited Artifact:** `foundations/code/Proyecto_Ising_MonteCarlo.ipynb`
**Focus:** Memory Limits, Algorithmic Complexity, and C++ Performance Targets.

---

## 1. The "Breaking Point" Analysis

Maximum usable lattice size before application failure on a standard workstation (16 GB RAM).

### 1D/2D Lattices (Memory Bound)
The primary constraint is the **Neighbors Look-up Table (LUT)**, stored as `int32` array of shape $(N, z)$.

| Topology | Coordination ($z$) | Memory per Site | RAM Limit (12GB usable) | Max $L$ (Python) |
| :--- | :---: | :---: | :---: | :---: |
| **Chain 1D** | 2 | ~18 bytes | $N \approx 6.6 \times 10^8$ | **$L \approx 660,000,000$** |
| **Square 2D** | 4 | ~26 bytes | $N \approx 4.6 \times 10^8$ | **$L \approx 21,000$** |
| **Hexagonal 2D** | 3 | ~22 bytes | $N \approx 5.4 \times 10^8$ | **$L \approx 23,000$** |

### 3D Lattices (The Critical Bottleneck)
The BCC lattice ($z=8$) scales cubically with $L$, consuming memory rapidly.

| Topology | Coordination ($z$) | Memory per Site | RAM Limit (12GB usable) | Max $L$ (Python) |
| :--- | :---: | :---: | :---: | :---: |
| **BCC 3D** | 8 | ~42 bytes | $N \approx 2.8 \times 10^8$ | **$L \approx 650$** |

**Verdict:** The current Python implementation cannot simulate a $1000 \times 1000 \times 1000$ system ($N=10^9$) because the neighbor table would require **32 GB of RAM**, exceeding typical workstation limits.

---

## 2. Hidden Iterations & Complexity

### 2.1 The Random Number Generator (RNG) Bottleneck
Inside `_njit_metropolis_sweeps`, the code executes:
```python
i = occ_idx[np.random.randint(n_occ)]
```
- **Cost:** One call to the PRNG state updater + modulo arithmetic per spin flip.
- **Scaling:** $O(N \times \text{Sweeps})$.
- **Inefficiency:** Generating random numbers one-by-one inside a hot loop prevents vectorization and incurs high function-call overhead, even in Numba. C++ implementations often use LCGs inline or block-generation.

### 2.2 Serial Execution Model
The current Numba optimization (`@njit`) is applied to `_njit_metropolis_sweeps`, but it runs **single-threaded** for a given simulation.
- **Parallelism:** Only exists across *independent* simulations (different $H$ or $T$ values via `simulate_multiple_H_parallel`).
- **Impact:** A large lattice ($L=2000$) runs on 1 core, leaving 7 cores idle.
- **Target:** C++ Checkboard decomposition to use all 8 cores for a *single* lattice.

---

## 3. Memory Footprint: Current vs. Proposed

Comparison for a theoretically large target: **3D Lattice with $L=1000$ ($10^9$ sites)**.

### Current Python ($N=10^9$, $z=8$)
- **Spins (`int8`):** 1 GB
- **Neighbors (`int32`):** 32 GB ($10^9 \times 8 \times 4$ bytes)
- **Indices (`int64`):** 8 GB
- **Total:** **~41 GB** (Unusable)

### Proposed C++ Bit-Packed ($N=10^9$, $z=8$)
- **Spins (1-bit):** 125 MB ($10^9 / 8$)
- **Neighbors:** **0 MB** (Implicit calculation on-the-fly via bitwise shifts, no LUT stored).
- **Total:** **~0.125 GB**

**Verdict:** Implicit neighbor calculation in C++ is **mandatory** for large 3D systems. Storing neighbors explicitly is the scalability killer.

---

## 4. C++ Performance Targets (v2.0)

Estimated speed comparison based on standard Monte Carlo logic.

| Metric | Python (Numba) | C++ (Optimized) | Speedup Factor |
| :--- | :--- | :--- | :--- |
| **Core Usage** | Single-core | 8-core (OpenMP) | 8x |
| **Data Structure** | Array Iteration | Bit-wise Operations | 2x-4x |
| **RNG** | `np.random` (per call) | Xoshiro/PCG (inline) | 2x |
| **Throughput** | ~150 MegaFlips/sec | ~2-4 GigaFlips/sec | **~15x - 25x** |

**Key KPI:** The C++ engine must exceed **1 GigaFlip per second** (aggregated) to justify the migration effort.

---

## 5. Benchmarking Strategy (Baseline)

Run these 3 scenarios in the current Python notebook to establish the "Base Speed".

### Benchmark A: "The Sprinter" (High Throughput)
- **System:** Square 2D, $L=100$ ($10^4$ sites).
- **Params:** $T = 100.0$ (High T = random usage), $H=0$. 100,000 Sweeps.
- **Goal:** Measure raw Single-Core Flip Rate (flips/sec) without memory bottlenecks.

### Benchmark B: "The Endurance" (Cache Stress)
- **System:** Square 2D, $L=4000$ ($1.6 \times 10^7$ sites).
- **Params:** $T = 2.27$ (Critical), $H=0$. 100 Sweeps.
- **Goal:** Measure impact of Last-Level Cache (LLC) misses when lattice > Cache size.

### Benchmark C: "The Heavy Lifter" (Complexity)
- **System:** BCC 3D, $L=100$ ($10^6$ sites).
- **Params:** $T = 4.0$, $H=0$. 1,000 Sweeps.
- **Goal:** Baseline for 3D topology ($z=8$) which involves heavier indexing math.

---

**Next Step:** Execute these benchmarks in Python before changing any code.
