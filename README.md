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

### [v2.3] Extreme Parallelization (In Progress)
*Maximizing hardware utilization.*
*   **Objective:** Implement **OpenMP** multi-threading to parallelize the temperature sweep, pushing CPU utilization to 100% across all available hardware cores.

### [v3.0] Econophysics: Market Sentiment Analysis (Planned)
*Targeting Financial Industry applications (Bancolombia Talento B).*
*   **Objective:** Map Ising dynamics to financial time-series to detect "Herd Behavior" and market volatility.
*   **Hypothesis:** Using the Critical Temperature ($T_c$) of the system to identify phase transitions in investor sentiment, acting as a predictor for market crashes.

---

## 🛠 Tech Stack
- **Compute Backend:** C++17, OpenMP, Data-Oriented Design.
- **Data Science Frontend:** Python 3.x, Jupyter, Pandas, Matplotlib, Seaborn.
- **Tools:** VS Code (Agentic Workflows), g++, Git, Overleaf.

## 👥 Contributors
- **[@SiririComun](https://github.com/SiririComun)** - HPC Architecture, C++ Optimization, Data Science, Econophysics.
- **[@JuanJ27](https://github.com/JuanJ27)** - Original Python implementation & Statistical Physics Research.

---

*Note: This repository is a living project intended for academic scholarship applications and professional data science portfolios.*