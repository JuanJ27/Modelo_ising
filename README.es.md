# Ising-Dynamics 🧲 📈

**Uniendo la Física Teórica y la Inteligencia Financiera mediante Computación de Alto Rendimiento.**

---

[Read in English 🇺🇸](./README.md)

## 💡 El Concepto
Este proyecto explora la universalidad del **Modelo de Ising**, evolucionando desde una simulación clásica de Mecánica Estadística hacia un motor de alto rendimiento para el análisis de mercados financieros. Demuestra la transición de la investigación académica a la aplicación industrial.

## 🚀 Evolución del Proyecto (Versiones)

### [v1.0] Fundamentos: Física Estocástica
*Desarrollado como parte del curso de Física Estadística en la Universidad de Antioquia (UdeA).*
*   **Objetivo:** Implementar el algoritmo de Metropolis-Hastings para simular sistemas ferromagnéticos en 2D.
*   **Características:** Implementación optimizada en Python usando NumPy/Numba, análisis de transiciones de fase y magnetización espontánea.
*   **Activos Científicos:** 
    *   [Reporte Científico Completo (PDF)](./foundations/report/Ising_model_Report.pdf)
    *   [Presentación Técnica (Fuente LaTeX incluida)](./foundations/presentation/presentation.pdf)

### [v2.0] Motor de Alto Rendimiento (En progreso)
*Enfoque en Ingeniería de Software y Eficiencia Computacional.*
*   **Objetivo:** Migrar el núcleo de la simulación a **C++20** para lograr una aceleración de más de 100x.
*   **Características:** Empaquetamiento de bits (almacenando 64 espines por entero), aleatoriedad de alta calidad con Mersenne Twister (MT19937) e integración híbrida Python/C++.
*   **Habilidad Destacada:** Gestión de memoria de bajo nivel y patrones de Computación de Alto Rendimiento (HPC).

### [v3.0] Econofísica: Análisis de Sentimiento de Mercado (Planeado)
*Dirigido a aplicaciones en la industria financiera (Bancolombia Talento B).*
*   **Objetivo:** Mapear la dinámica de Ising a series de tiempo financieras para detectar "Comportamiento de Rebaño" (Herd Behavior) y volatilidad.
*   **Hipótesis:** Utilizar la Temperatura Crítica ($T_c$) del sistema para identificar transiciones de fase en el sentimiento de los inversores, actuando como un predictor de crisis de mercado.
*   **Fuente de Datos:** Retornos históricos de la Bolsa de Valores de Colombia (BVC).

---

## 🛠 Stack Tecnológico
- **Lenguajes:** Python (Análisis de Datos), C++ (Motor de Simulación), LaTeX (Documentación).
- **Librerías:** NumPy, Matplotlib, Pybind11 (Futuro).
- **Herramientas:** VS Code, Git/GitHub, Overleaf.

## 👥 Colaboradores
- **[@SiririComun](https://github.com/SiririComun)** - Investigación, Optimización en C++, Econofísica.
- **[@JuanJ27](https://github.com/JuanJ27)** - Implementación original en Python e investigación de Física Estadística.

---

*Nota: Este repositorio es un proyecto vivo destinado a aplicaciones de becas académicas y portafolios profesionales de ciencia de datos.*