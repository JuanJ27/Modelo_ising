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
*   **Características:** Implementación en Python con NumPy/Numba, análisis de transiciones de fase y magnetización espontánea.
*   **Activos Científicos:**
    *   [Reporte Científico Completo (PDF)](./foundations/report/Ising_model_Report.pdf)
    *   [Presentación Técnica (Fuente LaTeX incluida)](./foundations/presentation/presentation.pdf)

### [v2.0] Motor Modular en C++ ✅ COMPLETADO
*Transición del prototipo en Python a un motor HPC de nivel de producción.*
*   **Arquitectura:** Separación completa de responsabilidades (OOD) combinada con **Diseño Orientado a Datos (DOD)**.
*   **Hito de Rendimiento:** **~0.0126 GigaFlips por segundo (GF/s)** (~79 ns/flip), logrando una **aceleración >600x** sobre la implementación original en Python.
*   **Características Clave:**
    *   **Contigüidad de Memoria:** Almacenamiento 1D en orden mayor de fila (`std::vector<int8_t>`) para residencia óptima en caché L1/L2.
    *   **Optimización:** Tablas de vecinos precomputadas ($O(1)$) y **Tablas de Búsqueda (LUT) de Boltzmann** para eliminar llamadas costosas a `std::exp`.
    *   **RNG Científico:** `std::mt19937_64` (Mersenne Twister) pasado por referencia para mantener cadenas de Markov continuas.

### [v2.2] Validación Termodinámica y Frontend de Ciencia de Datos ✅ COMPLETADO
*Desacoplamiento del backend de cómputo (C++) del frontend de visualización (Python/Jupyter).*
*   **Ensamble de Alta Estocasticidad:** Promediado por ensamble con múltiples ensayos para mitigar el desaceleramiento crítico cerca de $T_c$.
*   **Teorema de Fluctuación-Disipación (TFD):** Calor Específico ($C_v$) y Susceptibilidad Magnética ($\chi$) calculados via medición precisa de varianza de la Cadena de Markov.
*   **Visualizaciones de Calidad Publicable:** Error Estándar de la Media (SEM) para gráficos rigurosos de transición de fase.

### [v2.3] Paralelización Extrema ✅ COMPLETADO
*Maximizando la utilización del hardware con multi-threading.*
*   **Logro:** Implementación de directivas **OpenMP**, alcanzando ~100% de utilización en un procesador Intel i7 de 12 hilos.
*   **Referencia de Rendimiento CPU:** **~0.0126 GF/s** (L=1024) — línea base establecida para comparación con GPU.
*   **Impacto:** Reducción del tiempo de simulación ~8x, habilitando barridos de ensamble de alta resolución en minutos.

### [v2.4] Infraestructura y Reproducibilidad ✅ COMPLETADO
*Benchmark GPU en contenedor con una aceleración de ~152,000x sobre la línea base en Python.*
*   **Problema Resuelto:** Fedora 43 incluye `glibc 2.40`, que conflictúa con los headers de matemáticas de dispositivo de todos los toolkits CUDA ≤ 12.9 (`noexcept` en `cospi/sinpi/rsqrt`). La solución requiere un entorno de OS controlado.
*   **Solución:** Pipeline CUDA completamente containerizado con **Docker (`nvidia/cuda:12.6.2-devel-ubuntu22.04`)**, proveyendo una base `glibc 2.35` compatible con CUDA 12.6 y hardware Pascal.
*   **Algoritmo GPU:** **Metropolis Red-Black (Tablero de Ajedrez)** — permite actualizar N/2 sitios de un sublattice en paralelo sin conflictos de datos, preservando el Balance Detallado.
*   **RNG:** **xorshift64\*** inline (Vigna 2014) — pasa BigCrush, solo registros, sin dependencias externas de headers.
*   **🏆 Resultado del Benchmark — GTX 1050 Ti (sm\_61, Pascal, 768 núcleos CUDA):**

    | Métrica | Valor |
    |---|---|
    | Red | L=1024 (N=1,048,576 sitios) |
    | Barridos | 1,000 |
    | Temperatura | T=2.269 (≈ Tc) |
    | **Rendimiento** | **~1.92 GigaFlips/segundo** |
    | vs CPU (v2.3, 0.0126 GF/s) | **~152× más rápido** |
    | vs Python (v1.0) | **~152,000× más rápido** |

### [v3.0] Econofísica: Análisis de Sentimiento de Mercado (Planeado)
*Dirigido a aplicaciones en la industria financiera.*
*   **Objetivo:** Mapear la dinámica de Ising a series de tiempo financieras para detectar "Comportamiento de Rebaño" (Herd Behavior) y volatilidad de mercado.
*   **Hipótesis:** Utilizar la Temperatura Crítica ($T_c$) del sistema para identificar transiciones de fase en el sentimiento de los inversores, como predictor de crisis.
*   **Fuente de Datos:** Retornos históricos de la Bolsa de Valores de Colombia (BVC).

---

## 🐳 Reproducibilidad (Docker)

El pipeline de benchmark GPU corre dentro de un entorno containerizado para garantizar fidelidad de ejecución en cualquier OS anfitrión.

| Componente | Versión Estandarizada | Justificación |
| :--- | :--- | :--- |
| **OS Base** | Ubuntu 22.04 LTS | `glibc 2.35` — completamente compatible con los headers de CUDA 12.6 |
| **CUDA Stack** | 12.6.2 Devel | Compatibilidad verificada con Pascal (`sm_61`) y arquitecturas posteriores |
| **Compilador** | GCC 11 (Ubuntu defecto) | Dentro del rango de soporte oficial de CUDA 12.6 (≤ GCC 12) |
| **Runtime GPU** | NVIDIA Container Toolkit | Passthrough directo del hardware GTX/RTX al contenedor |

```bash
# Construir la imagen
docker compose build

# Entrar al laboratorio con acceso a GPU
docker compose run ising-lab

# Compilar y ejecutar el benchmark de una sola vez
docker compose run ising-lab bash -c \
  "nvcc -O3 -arch=sm_61 -Wno-deprecated-gpu-targets \
   high-performance/src/cuda_kernel.cu -o ising_cuda && ./ising_cuda"
```

---

## 🛠 Stack Tecnológico
- **Lenguajes:** C++17 (Motor HPC), Python 3.x (Análisis), CUDA (Kernels GPU).
- **Paralelismo:** OpenMP (Multi-threading), Red-Black Checkerboard (SIMT).
- **Infraestructura:** Docker, Docker Compose, NVIDIA Container Toolkit.
- **Ciencia de Datos:** Pandas, Matplotlib, NumPy, Jupyter.
- **Herramientas:** VS Code (Agentic Workflows), g++, nvcc, Git, Overleaf.

## 👥 Colaboradores
- **[@SiririComun](https://github.com/SiririComun)** - Arquitectura HPC, Optimización C++, Ciencia de Datos, Econofísica.
- **[@JuanJ27](https://github.com/JuanJ27)** - Implementación original en Python e investigación de Física Estadística.

---

*Nota: Este repositorio es un proyecto vivo destinado a aplicaciones de becas académicas y portafolios profesionales de ciencia de datos.*