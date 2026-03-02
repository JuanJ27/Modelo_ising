# =============================================================================
# Ising-Dynamics HPC Environment
# Phase 2.4 — Dockerized GPU Benchmark Lab
#
# WHY THIS BASE IMAGE:
#   Fedora 43 ships glibc 2.40, which introduces noexcept(true) on cospi/sinpi/
#   rsqrt. This conflicts with CUDA ≤ 12.9's crt/math_functions.h declarations
#   under cudafe++'s C++17 enforcement — a hard toolchain incompatibility.
#   Ubuntu 22.04 LTS ships glibc 2.35, which is fully compatible with CUDA 12.6.
#   This container provides a stable, reproducible build environment decoupled
#   from the host OS.
#
# TARGET HARDWARE:  NVIDIA GTX 1050 Ti (sm_61 / Pascal)
# CUDA TOOLKIT:     12.6.2
# HOST COMPILER:    GCC 11 (Ubuntu 22.04 default, within CUDA 12.6's support)
# =============================================================================

FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# System dependencies
#   build-essential  — GCC 11, g++, make
#   cmake            — build system for future multi-target compilation
#   git              — source control inside container
#   libomp-dev       — OpenMP runtime (Phase 2.3 CPU benchmark parity)
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------
WORKDIR /app/Ising-Dynamics

# Default command: opens a bash shell for interactive benchmark sessions
CMD ["/bin/bash"]
