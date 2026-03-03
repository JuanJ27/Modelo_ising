# Ising-Dynamics 🧲 📈

**Bridging Theoretical Physics and Financial Intelligence through High-Performance Computing.**

---

[Leer en Español 🇪🇸](./README.es.md)

## 💡 The Concept
This project explores the universality of the **Ising Model**, evolving from a classical Statistical Mechanics simulation into a high-performance engine for financial market analysis. It demonstrates the transition from academic research to industrial application.

## 🚀 Project Evolution (Versioning)

### [v1.0] Foundations: Stochastic Physics
*Developed as part of the Statistical Physics course at Universidad de Antioquia (UdeA).*
*   **Objective:** Implement the Metropolis-Hastings algorithm to simulate 2D ferromagnetic systems.
*   **Key Features:** Python implementation using NumPy/Numba, analysis of phase transitions, and spontaneous magnetization.
*   **Scientific Assets:** 
    *[Full Scientific Report (PDF)](./foundations/report/Ising_model_Report.pdf)
    *[Technical Presentation (LaTeX Source Included)](./foundations/presentation/presentation.pdf)

### [v2.0] The Modular C++ Engine ✅ COMPLETED
*Transitioned from Python Prototype to Production-grade HPC.*
*   **Architecture:** Full Separation of Concerns (OOD) combined with **Data-Oriented Design (DOD)**.
*   **Performance Milestone:** Reached throughputs of **~0.0126 GigaFlips per second (GF/s)** (~79 ns/flip including measurement overhead), achieving a **>600x speedup** over the original Python implementation.
*   **Key Features:** 
    *   **Memory Contiguity:** 1D Row-Major storage (`std::vector<int8_t>`) for optimal L1/L2 cache residency.
    *   **Optimization:** Precomputed neighbor lookup tables ($O(1)$) and **Boltzmann Lookup Tables (LUT)** to eliminate expensive `std::exp` calls.
    *   **Scientific RNG:** Integrated `std::mt19937_64` (Mersenne Twister) passed by reference to maintain unbroken Markov Chain sequences.

### [v2.2] Thermodynamic Validation & Data Science Frontend ✅ COMPLETED
*Decoupled compute backend (C++) from visualization frontend (Python/Jupyter).*
*   **High-Stochastic Ensemble:** Implemented multi-trial ensemble averaging to mitigate critical slowing down near $T_c$.
*   **Fluctuation-Dissipation Theorem (FDT):** Computed Specific Heat ($C_v$) and Magnetic Susceptibility ($\chi$) using precise variance measurements of the Markov Chain.
*   **Publication-Quality Visualizations:** Extracted Standard Error of the Mean (SEM) to generate rigorous, Nature/Science-grade plots of the phase transition.

### [v2.3] Extreme Parallelization ✅ COMPLETED
*Maximizing hardware utilization via multi-threading.*
*   **Objective:** Parallelize the temperature sweep to scale across all available CPU cores.
*   **Achievement:** Successfully implemented **OpenMP** directives, achieving near-100% utilization of a 12-thread Intel i7 processor.
*   **Performance Baseline:** **~0.0126 GF/s** (single-thread random Metropolis on L=1024) — established as the CPU reference for GPU comparison.
*   **Impact:** Reduced simulation wall-clock time by ~8x, enabling high-resolution ensemble sweeps in minutes instead of hours.

### [v2.4] Infrastructure & Reproducibility ✅ COMPLETED
*Containerized GPU benchmark delivering a ~152,000x speedup over the Python baseline.*
*   **Problem Solved:** Fedora 43 ships `glibc 2.40`, which conflicts with all CUDA ≤ 12.9 device math headers (`cospi/sinpi/rsqrt noexcept` mismatch under cudafe++). No flag-level workaround exists — the fix requires a controlled OS environment.
*   **Solution:** Containerized the full CUDA pipeline using **Docker (`nvidia/cuda:12.6.2-devel-ubuntu22.04`)**, providing a stable `glibc 2.35` baseline compatible with CUDA 12.6 and Pascal hardware.
*   **GPU Algorithm:** **Red-Black (Checkerboard) Metropolis** — allows all N/2 sites of one sublattice to update in parallel with zero data hazards, preserving Detailed Balance.
*   **RNG:** Inline **xorshift64\*** (Vigna 2014) — passes BigCrush, register-only, zero external header dependencies.
*   **🏆 Benchmark Result — GTX 1050 Ti (sm\_61, Pascal, 768 CUDA cores):**

    | Metric | Value |
    |---|---|
    | Lattice | L=1024 (N=1,048,576 sites) |
    | Sweeps | 1,000 |
    | Temperature | T=2.269 (≈ Tc) |
    | **Throughput** | **~1.92 GigaFlips/second** |
    | vs CPU baseline (v2.3, 0.0126 GF/s) | **~152× faster** |
    | vs Python baseline (v1.0) | **~152,000× faster** |

### [v2.6.0] Scientific Masterpiece Milestone ✅ COMPLETED
*Full Finite-Size Scaling validation — the definitive proof of the GPU physics engine.*
*   **Peak Throughput:** **2.09 GF/s** at L=1024 (GTX 1050 Ti, sm_61, Pascal).
*   **Scale:** $L \in \{64, 128, 256, 512, 1024\}$, $K=15$ independent trials per T-point, 85 T-points each.
*   **RNG Upgrade:** Philox-4×32-10 counter-based CBRNG (Salmon et al. 2011) — eliminates period-collision artifacts in massively parallel multi-trial runs.
*   **Scientific Output:** `Master_FSS_Analysis.ipynb` — T_c(L) finite-size shift table, χ_max power-law scaling ($R^2=0.993$), and critical exponent extraction via OLS log-log regression with Onsager reference.
*   **🏆 Speedup vs Python v1.0 baseline:** **96,000× faster** than the original NumPy prototype for a full FSS sweep workload.

### [v3.0] Econophysics: Market Sentiment Analysis (In Progress)
*Targeting Financial Industry applications — primary development focus from v2.6.0 onward.*
*   **Objective:** Map Ising dynamics to financial time-series to detect "Herd Behavior" and market volatility.
*   **Hypothesis:** Using the Critical Temperature ($T_c$) of the system to identify phase transitions in investor sentiment, acting as a predictor for market crashes.

---

## 🐳 Reproducibility (Docker)

To ensure 100% execution fidelity across different host Operating Systems, the GPU benchmark pipeline is fully containerized. This resolves the `glibc` symbol conflicts found in bleeding-edge distributions like Fedora 43.

| Component | Standardized Version | Rationale |
| :--- | :--- | :--- |
| **HPC Base** | Ubuntu 22.04 LTS | Stable `glibc 2.35` baseline for CUDA toolchain compatibility. |
| **CUDA Stack** | 12.6.2 Devel | Native support for Pascal (`sm_61`) and newer architectures. |
| **Host Compiler**| GCC 11.x | Within official support range for CUDA 12.6 stability. |
| **Runtime** | NVIDIA Container Toolkit| Direct hardware-passthrough for GTX/RTX hardware. |

```bash
# Build the standardized HPC environment
docker compose build

# Execute the high-precision GPU benchmark inside the container
docker compose run --rm ising-lab bash -c \
  "nvcc -O3 -arch=sm_61 -Wno-deprecated-gpu-targets \
   high-performance/src/fss_sweep.cu -o fss_sim && ./fss_sim"
```

---

## 🛠 Tech Stack
- **Languages:** C++17 (HPC Core), Python 3.x (Analysis), CUDA (GPU Kernels).
- **Parallelism:** OpenMP (Multi-threading), Red-Black Checkerboard (SIMD/SIMT), GPU Warp-Shuffle Reductions.
- **Infrastructure:** Docker, Docker Compose, NVIDIA Container Toolkit (Hardware Passthrough).
- **Data Science:** Pandas, Matplotlib, NumPy, Jupyter, SciPy (OLS Regression).
- **Scientific Computing:** Finite-Size Scaling Theory, Fluctuation-Dissipation Theorem, Monte Carlo Metropolis-Hastings.

## 👥 Contributors
- **[@SiririComun](https://github.com/SiririComun)** - HPC Architecture, C++ Optimization, Data Science, Econophysics.
- **[@JuanJ27](https://github.com/JuanJ27)** - Original Python implementation & Statistical Physics Research.

---

*Note: This repository is a living project intended for academic scholarship applications and professional data science portfolios.*