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

### [v2.0] The Modular C++ Engine
*Transitioned from Python Prototype to Production-grade HPC.*
*   **Architecture:** Full Separation of Concerns (OOD) combined with **Data-Oriented Design (DOD)**.
*   **Performance Milestone:** Reached throughputs of **~0.0126 GigaFlips per second (GF/s)** (~79 ns/flip including measurement overhead), achieving a **>600x speedup** over the original Python implementation.
*   **Key Features:** 
    *   **Memory Contiguity:** 1D Row-Major storage (`std::vector<int8_t>`) for optimal L1/L2 cache residency.
    *   **Optimization:** Precomputed neighbor lookup tables ($O(1)$) and **Boltzmann Lookup Tables (LUT)** to eliminate expensive `std::exp` calls.
    *   **Scientific RNG:** Integrated `std::mt19937_64` (Mersenne Twister) passed by reference to maintain unbroken Markov Chain sequences.

### [v2.2] Thermodynamic Validation & Data Science Frontend
*Decoupled compute backend (C++) from visualization frontend (Python/Jupyter).*
*   **High-Stochastic Ensemble:** Implemented multi-trial ensemble averaging to mitigate critical slowing down near $T_c$.
*   **Fluctuation-Dissipation Theorem (FDT):** Computed Specific Heat ($C_v$) and Magnetic Susceptibility ($\chi$) using precise variance measurements of the Markov Chain.
*   **Publication-Quality Visualizations:** Extracted Standard Error of the Mean (SEM) to generate rigorous, Nature/Science-grade plots of the phase transition.

### [v2.3] Extreme Parallelization 
*Maximizing hardware utilization via multi-threading.*
*   **Objective:** Parallelize the temperature sweep to scale across all available CPU cores.
*   **Achievement:** Successfully implemented **OpenMP** directives, achieving near-100% utilization of a 12-thread Intel i7 processor.
*   **Impact:** Reduced simulation wall-clock time by ~8x, enabling high-resolution ensemble sweeps in minutes instead of hours.

### [v2.4] Infrastructure & Reproducibility (In Progress)
*Containerizing the HPC pipeline for universal deployment.*
*   **Objective:** Eliminate "Environment Drift" and toolchain conflicts by isolating the C++/CUDA stack.
*   **Solution:** Developing a **Dockerized HPC environment** based on Ubuntu 22.04 LTS. This ensures that the high-performance kernels run identically on local hardware, cloud instances, or research clusters.
*   **Impact:** Simplifies the "GTX 1050 Ti to Modern OS" bridge, providing a stable sandbox for GPU-accelerated research.

### [v3.0] Econophysics: Market Sentiment Analysis (Planned)
*Targeting Financial Industry applications (Bancolombia Talento B).*
*   **Objective:** Map Ising dynamics to financial time-series to detect "Herd Behavior" and market volatility.
*   **Hypothesis:** Using the Critical Temperature ($T_c$) of the system to identify phase transitions in investor sentiment, acting as a predictor for market crashes.

---

## 🐳 Reproducibility (Docker)

To ensure 100% execution fidelity across different host Operating Systems, the GPU benchmark pipeline is fully containerized.

| Component | Standardized Version | Rationale |
| :--- | :--- | :--- |
| **HPC Base** | Ubuntu 22.04 LTS | Long-term support with stable ABI/glibc headers. |
| **CUDA Stack** | 12.6.x Devel | Verified compatibility with Pascal (`sm_61`) and newer architectures. |
| **Compiler** | GCC 11.x | Industry-standard host compiler for numerical stability. |
| **Runtime** | Nvidia Container Toolkit | Direct hardware-passthrough for GTX/RTX hardware. |

---

## 🛠 Tech Stack
- **Languages:** C++17 (HPC Core), Python 3.x (Analysis), CUDA (GPU Kernels).
- **Parallelism:** OpenMP (Multi-threading), Red-Black Checkerboard (SIMT).
- **Infrastructure:** Docker, Docker Compose, NVIDIA Container Toolkit.
- **Data Science:** Pandas, Matplotlib, NumPy, Jupyter.

## 👥 Contributors
- **[@SiririComun](https://github.com/SiririComun)** - HPC Architecture, C++ Optimization, Data Science, Econophysics.
- **[@JuanJ27](https://github.com/JuanJ27)** - Original Python implementation & Statistical Physics Research.

---

*Note: This repository is a living project intended for academic scholarship applications and professional data science portfolios.*