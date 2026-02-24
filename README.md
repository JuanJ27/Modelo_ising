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
*   **Key Features:** Optimized Python implementation using NumPy/Numba, analysis of phase transitions, and spontaneous magnetization.
*   **Scientific Assets:** 
    *   [Full Scientific Report (PDF)](./foundations/report/Ising_model_Report.pdf)
    *   [Technical Presentation (LaTeX Source Included)](./foundations/presentation/presentation.pdf)

### [v1.8 / v2.0] High-Performance Engine (In Progress)
*Transitioning from Prototype to Production-grade HPC.*
*   **Status:** Foundational Kernels completed. Currently architecting the modular simulation library.
*   **Performance Milestone:** Validated a **989x speedup** (~50 ns/flip) compared to the original Python implementation.
*   **Key Features:** 
    *   **Memory Architecture:** Contiguous 1D layout for optimal L1/L2 cache locality.
    *   **Bit-Packing:** 64:1 memory compression using bitwise primitives (SET/GET/XOR).
    *   **Scientific RNG:** Integrated `std::mt19937_64` (Mersenne Twister) for high-entropy, long-period simulations.
    *   **Efficiency:** Implemented **Boltzmann Lookup Tables (LUT)** to eliminate expensive transcendental calculations.
*   **Standard:** Written in **C++17** focusing on Data-Oriented Design (DOD).

### [v3.0] Econophysics: Market Sentiment Analysis (Planned)
*Targeting Financial Industry applications (Bancolombia Talento B).*
*   **Objective:** Map Ising dynamics to financial time-series to detect "Herd Behavior" and market volatility.
*   **Hypothesis:** Using the Critical Temperature ($T_c$) of the system to identify phase transitions in investor sentiment, acting as a predictor for market crashes.
*   **Data Source:** Historical returns from the Colombian Stock Exchange (BVC).

---

## 🛠 Tech Stack
- **Languages:** Python (Data Analysis), C++ (Simulation Engine), LaTeX (Documentation).
- **Libraries:** NumPy, Matplotlib, Pybind11 (Future).
- **Tools:** VS Code (Agentic Workflows), g++, Git, Overleaf.

## 👥 Contributors
- **[@SiririComun](https://github.com/SiririComun)** - Research, C++ Optimization, Econophysics.
- **[@JuanJ27](https://github.com/JuanJ27)** - Original Python implementation & Statistical Physics Research.

---

*Note: This repository is a living project intended for academic scholarship applications and professional data science portfolios.*
