# Ising-Dynamics 🧲 📈
[Leer en Español 🇪🇸](./README.es.md)
**Bridging Theoretical Physics and Financial Intelligence through High-Performance Computing.**

---

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

### [v2.0] High-Performance Engine (In Progress)
*Focusing on Software Engineering and Computational Efficiency.*
*   **Objective:** Migrate the simulation core to **C++20** to achieve a 100x+ speedup.
*   **Key Features:** Bit-packing (storing 64 spins per integer), high-quality randomness with Mersenne Twister (MT19937), and hybrid Python/C++ integration.
*   **Skill Highlight:** Low-level memory management and High-Performance Computing (HPC) patterns.

### [v3.0] Econophysics: Market Sentiment Analysis (Planned)
*Targeting Financial Industry applications (Bancolombia Talento B).*
*   **Objective:** Map Ising dynamics to financial time-series to detect "Herd Behavior" and market volatility.
*   **Hypothesis:** Using the Critical Temperature ($T_c$) of the system to identify phase transitions in investor sentiment, acting as a predictor for market crashes.
*   **Data Source:** Historical returns from the Colombian Stock Exchange (BVC).

---

## 🛠 Tech Stack
- **Languages:** Python (Data Analysis), C++ (Simulation Engine), LaTeX (Documentation).
- **Libraries:** NumPy, Matplotlib, Pybind11 (Future).
- **Tools:** VS Code, Git/GitHub, Overleaf.

## 👥 Contributors
- **[@SiririComun](https://github.com/SiririComun)** - Research, C++ Optimization, Econophysics.
- **[@JuanJ27](https://github.com/JuanJ27)** - Original Python implementation & Statistical Physics Research.

---

*Note: This repository is a living project intended for academic scholarship applications and professional data science portfolios.*
