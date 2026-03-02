/**
 * @file cuda_kernel.cu
 * @brief 2D Ising Model – GPU Checkerboard Metropolis Benchmark
 *
 * Phase 2.4 | Target: GTX 1050 Ti (sm_61 / Pascal) | CUDA 12.6
 *
 * Physics:
 *   H = -J Σ s_i s_j    (square lattice, PBC)
 *   ΔE = 2J · s_i · Σ_nn s_j
 *   P(accept) = min(1, exp(-β·ΔE))    [Metropolis criterion]
 *
 * Parallelism: Red-Black (Checkerboard) decomposition.
 *   Sublattice-even  {(i+j)%2 == 0} and sublattice-odd {(i+j)%2 == 1}
 *   are updated in alternating half-sweeps. Within each half-sweep every
 *   site's four neighbours belong to the OTHER sublattice → zero data
 *   hazards, no atomics, Detailed Balance preserved.
 *
 * RNG: inline xorshift64* (Vigna 2014) – passes BigCrush, register-only,
 *   zero external dependencies.
 *
 * Headers: <stdio.h> and <cuda_runtime.h> ONLY (no C++ stdlib).
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* ── Simulation parameters ─────────────────────────────────────────────── */
#define L        1024
#define N        (L * L)          /* 1 048 576 sites */
#define N_SWEEPS 1000
#define T_SIM    2.269             /* ≈ Tc(2D Ising) */
#define J_COUP   1.0

/* ── Boltzmann LUT in GPU constant memory ───────────────────────────────
 * lut[k] = min(1, exp(-β·ΔE))   where   ΔE = 2J·(k-2)·2
 * k=0 → ΔE=-8J (always accept) … k=4 → ΔE=+8J (hardest reject)
 * Mirrors MetropolisEngine::rebuild_lut() exactly.               */
__constant__ float c_lut[5];

/* ── CUDA error-check macro ─────────────────────────────────────────────── */
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t _e = (call);                                          \
        if (_e != cudaSuccess) {                                          \
            printf("[CUDA ERROR] %s:%d  %s\n",                           \
                   __FILE__, __LINE__, cudaGetErrorString(_e));           \
            return 1;                                                     \
        }                                                                 \
    } while (0)

/* ── xorshift64* RNG (Vigna 2014) ──────────────────────────────────────
 * State is a per-thread unsigned long long (uint64_t equivalent).
 * Returns a uniform float in (0, 1].                              */
__device__ __forceinline__
float rng_next(unsigned long long* state)
{
    unsigned long long x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    /* Multiply-step then map to (0,1] via 2^-32 scale on upper 32 bits */
    unsigned long long mixed = x * 0x2545F4914F6CDD1DULL;
    return (float)((unsigned int)(mixed >> 32) + 1U) * 2.3283064370807974e-10f;
}

/* ── Kernel 1: Seed per-site RNG states ────────────────────────────────── */
__global__
void kernel_seed_rng(unsigned long long* __restrict__ rng_states,
                     unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    /* Splitmix64 scramble for unique, non-zero per-site seeds */
    unsigned long long z = seed + (unsigned long long)idx * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    rng_states[idx] = z ? z : 1ULL;   /* never leave state == 0 */
}

/* ── Kernel 2: One checkerboard half-sweep ──────────────────────────────
 * parity=0 → update sites where (row+col) is even  (red sublattice)
 * parity=1 → update sites where (row+col) is odd   (black sublattice)  */
__global__
void kernel_metropolis(signed char*         __restrict__ lattice,
                       unsigned long long*  __restrict__ rng_states,
                       int parity)
{
    int col = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int row = (int)(blockIdx.y * blockDim.y + threadIdx.y);

    if (row >= L || col >= L) return;
    if (((row + col) & 1) != parity) return;   /* skip wrong sublattice */

    int site = row * L + col;

    /* Periodic-boundary neighbours */
    int up    = ((row - 1 + L) % L) * L + col;
    int down  = ((row + 1)     % L) * L + col;
    int left  =  row * L + (col - 1 + L) % L;
    int right =  row * L + (col + 1)     % L;

    int spin = (int)lattice[site];
    int nb   = (int)lattice[up] + (int)lattice[down]
             + (int)lattice[left] + (int)lattice[right];

    /* LUT index: identical to MetropolisEngine::lut_index_from_spin_sum
     * dE_unit = spin * nb  ∈ {-4,-2,0,+2,+4}
     * lut_idx = dE_unit/2 + 2  ∈ {0,1,2,3,4}                       */
    int lut_idx = spin * nb / 2 + 2;

    float r = rng_next(&rng_states[site]);
    if (r < c_lut[lut_idx])
        lattice[site] = (signed char)(-spin);
}

/* ── Host: build and upload Boltzmann LUT ──────────────────────────────── */
static int upload_lut(double beta)
{
    float h_lut[5];
    int k;
    for (k = 0; k < 5; ++k) {
        int    dE_unit = (k - 2) * 2;
        double delta_E = 2.0 * J_COUP * (double)dE_unit;
        h_lut[k] = (delta_E <= 0.0) ? 1.0f : (float)exp(-beta * delta_E);
    }
    cudaError_t e = cudaMemcpyToSymbol(c_lut, h_lut, 5 * sizeof(float));
    if (e != cudaSuccess) { printf("LUT upload failed\n"); return 1; }

    printf("Boltzmann LUT  beta=%.6f  (T=%.4f)\n", beta, 1.0/beta);
    for (k = 0; k < 5; ++k)
        printf("  lut[%d]  dE/2J=%+d  p=%.6f\n", k, (k-2)*2, h_lut[k]);
    printf("\n");
    return 0;
}

/* ── main ─────────────────────────────────────────────────────────────── */
int main(void)
{
    printf("==========================================================\n");
    printf("  2D Ising Model - GPU Checkerboard Metropolis Benchmark\n");
    printf("  L=%-4d  N=%-8d  Sweeps=%-5d  T=%.4f\n",
           L, N, N_SWEEPS, T_SIM);
    printf("==========================================================\n\n");

    /* Device info */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    {
        int cores = prop.multiProcessorCount *
                    ((prop.major == 6 && prop.minor == 1) ? 128 : 64);
        printf("Device  : %s\n", prop.name);
        printf("Arch    : sm_%d%d  |  SMs: %d  |  CUDA cores: ~%d\n",
               prop.major, prop.minor, prop.multiProcessorCount, cores);
        printf("Clock   : %.0f MHz  |  VRAM: %u MiB\n\n",
               prop.clockRate / 1e3,
               (unsigned)(prop.totalGlobalMem >> 20));
    }

    /* Upload physics */
    if (upload_lut(1.0 / T_SIM) != 0) return 1;

    /* Allocate device memory */
    signed char*        d_lattice    = 0;
    unsigned long long* d_rng_states = 0;
    CUDA_CHECK(cudaMalloc((void**)&d_lattice,    (size_t)N * sizeof(signed char)));
    CUDA_CHECK(cudaMalloc((void**)&d_rng_states, (size_t)N * sizeof(unsigned long long)));

    printf("Memory  : lattice=%u KiB  |  RNG states=%u MiB\n",
           (unsigned)((size_t)N * sizeof(signed char) / 1024),
           (unsigned)((size_t)N * sizeof(unsigned long long) / (1024*1024)));

    /* Initialise lattice to +1 (ordered state) on host, copy to device */
    {
        signed char* h = (signed char*)malloc((size_t)N * sizeof(signed char));
        if (!h) { printf("malloc failed\n"); return 1; }
        int i;
        for (i = 0; i < N; ++i) h[i] = 1;
        CUDA_CHECK(cudaMemcpy(d_lattice, h, (size_t)N, cudaMemcpyHostToDevice));
        free(h);
    }

    /* Seed RNG states */
    {
        int threads = 256;
        int blocks  = (N + threads - 1) / threads;
        kernel_seed_rng<<<blocks, threads>>>(d_rng_states, 20260302ULL);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("RNG     : %d threads seeded (xorshift64*)\n\n", N);
    }

    /* Launch config: 32x8 = 256 threads/block, coalesced column access */
    dim3 block(32, 8);
    dim3 grid((L + block.x - 1) / block.x,
              (L + block.y - 1) / block.y);
    printf("Kernel  : grid=(%u x %u)  block=(%u x %u)\n\n",
           grid.x, grid.y, block.x, block.y);

    /* Timing */
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    printf("Running %d sweeps ...\n", N_SWEEPS);
    CUDA_CHECK(cudaEventRecord(t0));

    /* Main loop: one sweep = one red half + one black half */
    int s;
    for (s = 0; s < N_SWEEPS; ++s) {
        kernel_metropolis<<<grid, block>>>(d_lattice, d_rng_states, 0);
        kernel_metropolis<<<grid, block>>>(d_lattice, d_rng_states, 1);
    }

    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    double elapsed_s      = (double)ms * 1e-3;
    double total_attempts = (double)N * (double)N_SWEEPS;
    double gf_s           = total_attempts / elapsed_s / 1.0e9;

    printf("Done.\n\n");
    printf("==========================================================\n");
    printf("  RESULTS\n");
    printf("  Elapsed        : %.4f ms  (%.6f s)\n", ms, elapsed_s);
    printf("  Spin attempts  : %.4e\n", total_attempts);
    printf("  ----------------------------------------------------------\n");
    printf("  Throughput     : %.4f  GigaFlips/second\n", gf_s);
    printf("==========================================================\n");

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_lattice);
    cudaFree(d_rng_states);
    return 0;
}
