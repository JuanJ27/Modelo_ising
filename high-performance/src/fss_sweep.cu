/**
 * @file fss_sweep.cu
 * @brief Finite-Size Scaling (FSS) sweep for 2D Ising Model
 *
 * Phase 2.5 | Target: GTX 1050 Ti (sm_61) | Built on cuda_kernel.cu engine
 *
 * Physics:
 *   For each (L, T), measures:
 *     m_abs  = < |M/N| >          (mean absolute magnetization per site)
 *     chi    = beta * N * ( <m^2> - <|m|>^2 )  (susceptibility via FDT)
 *
 *   The susceptibility chi peaks at Tc(L) which shifts with L as:
 *     chi_max(L) ~ L^(gamma/nu)   [FSS hypothesis]
 *   Fitting this gives the critical exponent ratio gamma/nu.
 *   For the 2D Ising universality class (exact): gamma/nu = 7/4 = 1.75.
 *
 * Critical slowing down note:
 *   The autocorrelation time near Tc scales as tau ~ L^z (z~2.17 for 2D
 *   Ising). For L=1024 near Tc, true thermalization requires O(10^5) sweeps.
 *   N_THERM=1000 / N_MEAS=500 is a good-faith demonstration sweep; the FSS
 *   signal is visible and the exponent can be estimated, but for publication
 *   quality one would run O(10^4) – O(10^6) sweeps per (L,T) point.
 *
 * Performance design:
 *   - Default CUDA stream: all kernels for a given (L,T) are queued without
 *     intermediate host synchronization. ONE cudaDeviceSynchronize() and ONE
 *     cudaMemcpy per (L,T) point — minimal PCIe/sync overhead.
 *   - Magnetization: shared-memory block reduction (1 atomicAdd per block)
 *     stored into d_m_values[s]  → no D2H inside the measurement loop.
 *
 * Output: fss_data.csv  (columns: L, T, m_abs, chi)
 *
 * Compile (inside Docker / CUDA 12.6 + Ubuntu 22.04):
 *   nvcc -O3 -arch=sm_61 -Wno-deprecated-gpu-targets \
 *        fss_sweep.cu -o fss_sim
 *
 * Headers: <stdio.h>, <stdlib.h>, <cuda_runtime.h> ONLY.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* ── FSS parameters ──────────────────────────────────────────────────────── */
static const int FSS_SIZES[]  = {64, 128, 256, 512, 1024};
static const int N_FSS_SIZES  = 5;

#define T_MIN     2.200
#define T_MAX     2.400
#define T_STEP    0.005     /* 41 temperature points */

#define N_THERM   1000      /* thermalization sweeps per (L,T)  */
#define N_MEAS    500       /* measurement sweeps per (L,T)     */

#define J_COUP    1.0

/* ── Boltzmann LUT in constant memory ───────────────────────────────────────
 * c_lut[k] = min(1, exp(-β·ΔE))
 * k = spin*neighbor_sum/2 + 2  ∈ {0,1,2,3,4}
 * Identical encoding to MetropolisEngine::rebuild_lut()                    */
__constant__ float c_lut[5];

/* ── CUDA error-check macro ─────────────────────────────────────────────── */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            printf("[CUDA ERROR] %s:%d  %s\n",                              \
                   __FILE__, __LINE__, cudaGetErrorString(_e));              \
            return 1;                                                        \
        }                                                                    \
    } while (0)

/* ── xorshift64* RNG (Vigna 2014) ──────────────────────────────────────────
 * Register-only, zero header dependencies, passes BigCrush.               */
__device__ __forceinline__
float rng_next(unsigned long long* state)
{
    unsigned long long x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    unsigned long long mixed = x * 0x2545F4914F6CDD1DULL;
    return (float)((unsigned int)(mixed >> 32) + 1U) * 2.3283064370807974e-10f;
}

/* ── Kernel 1: Seed RNG states (runtime N) ──────────────────────────────── */
__global__
void kernel_seed_rng(unsigned long long* __restrict__ rng_states,
                     unsigned long long seed,
                     int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    /* Splitmix64 scramble — guarantees unique non-zero states */
    unsigned long long z = seed + (unsigned long long)idx * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    rng_states[idx] = z ? z : 1ULL;
}

/* ── Kernel 2: Red-Black Metropolis (runtime Lsize) ────────────────────────
 * parity=0 → red   sublattice {(row+col)%2 == 0}
 * parity=1 → black sublattice {(row+col)%2 == 1}
 * Neighbours are on the opposite sublattice for the current parity →
 * zero write-conflicts, Detailed Balance preserved.                        */
__global__
void kernel_metropolis(signed char*        __restrict__ lattice,
                       unsigned long long* __restrict__ rng_states,
                       int Lsize,
                       int parity)
{
    int col = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int row = (int)(blockIdx.y * blockDim.y + threadIdx.y);

    if (row >= Lsize || col >= Lsize) return;
    if (((row + col) & 1) != parity) return;

    int site  = row * Lsize + col;
    int up    = ((row - 1 + Lsize) % Lsize) * Lsize + col;
    int down  = ((row + 1)         % Lsize) * Lsize + col;
    int left  =  row * Lsize + (col - 1 + Lsize) % Lsize;
    int right =  row * Lsize + (col + 1)          % Lsize;

    int spin    = (int)lattice[site];
    int nb      = (int)lattice[up] + (int)lattice[down]
                + (int)lattice[left] + (int)lattice[right];
    int lut_idx = spin * nb / 2 + 2;

    float r = rng_next(&rng_states[site]);
    if (r < c_lut[lut_idx])
        lattice[site] = (signed char)(-spin);
}

/* ── Kernel 3: Fill lattice with constant spin value ───────────────────── */
__global__
void kernel_fill(signed char* __restrict__ lattice, int N, signed char val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) lattice[idx] = val;
}

/* ── Kernel 4: Block-reduction magnetization → d_m_values[meas_idx] ──────
 * Uses shared memory tree reduction (256 threads/block) then one
 * atomicAdd per block to d_m_values[meas_idx].
 * This gives ~N/256 atomics per call vs N atomics with naive approach.    */
__global__
void kernel_mag_reduce(const signed char* __restrict__ lattice,
                       int                              N,
                       int*               __restrict__ d_m_values,
                       int                              meas_idx)
{
    __shared__ int sdata[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * 256 + tid;

    sdata[tid] = (gid < N) ? (int)lattice[gid] : 0;
    __syncthreads();

    /* Warp-unrolled tree reduction */
    if (256 >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (256 >= 128) { if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads(); }
    /* Last warp — no __syncthreads() needed */
    if (tid < 32) {
        volatile int* v = sdata;
        v[tid] += v[tid + 32];
        v[tid] += v[tid + 16];
        v[tid] += v[tid +  8];
        v[tid] += v[tid +  4];
        v[tid] += v[tid +  2];
        v[tid] += v[tid +  1];
    }

    if (tid == 0)
        atomicAdd(&d_m_values[meas_idx], sdata[0]);
}

/* ── Upload Boltzmann LUT for given beta ────────────────────────────────── */
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
    return (e == cudaSuccess) ? 0 : 1;
}

/* ── main ─────────────────────────────────────────────────────────────── */
int main(void)
{
    int n_T = (int)((T_MAX - T_MIN) / T_STEP + 1.5);  /* 41 points */

    printf("================================================================\n");
    printf("  2D Ising Finite-Size Scaling Sweep\n");
    printf("  L  in {64, 128, 256, 512, 1024}\n");
    printf("  T  in [%.3f, %.3f]  dT=%.3f  (%d points)\n",
           T_MIN, T_MAX, T_STEP, n_T);
    printf("  N_THERM=%d  N_MEAS=%d\n", N_THERM, N_MEAS);
    printf("================================================================\n\n");

    /* Device info */
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("Device : %s  (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);
    }

    /* Open CSV output */
    FILE* fout = fopen("fss_data.csv", "w");
    if (!fout) { printf("Cannot open fss_data.csv for writing\n"); return 1; }
    fprintf(fout, "L,T,m_abs,chi\n");

    /* Host sample buffer — reused for all (L,T).  N_MEAS ints. */
    int* h_m_values = (int*)malloc((size_t)N_MEAS * sizeof(int));
    if (!h_m_values) { printf("malloc failed\n"); fclose(fout); return 1; }

    /* Device m_values array: N_MEAS ints, allocated once. */
    int* d_m_values = 0;
    CUDA_CHECK(cudaMalloc((void**)&d_m_values, (size_t)N_MEAS * sizeof(int)));

    int size_idx;
    for (size_idx = 0; size_idx < N_FSS_SIZES; ++size_idx) {

        int    L = FSS_SIZES[size_idx];
        int    N = L * L;
        double t_start_L;
        (void)t_start_L;

        printf("--- L = %4d  (N = %7d) ---\n", L, N);

        /* ── Allocate per-L device memory ──────────────────────────── */
        signed char*        d_lattice    = 0;
        unsigned long long* d_rng_states = 0;
        CUDA_CHECK(cudaMalloc((void**)&d_lattice,
                              (size_t)N * sizeof(signed char)));
        CUDA_CHECK(cudaMalloc((void**)&d_rng_states,
                              (size_t)N * sizeof(unsigned long long)));

        /* ── Seed RNG once per L ──────────────────────────────────── */
        {
            int threads = 256;
            int blocks  = (N + threads - 1) / threads;
            kernel_seed_rng<<<blocks, threads>>>(
                d_rng_states,
                20260302ULL + (unsigned long long)L * 7919ULL,
                N);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        /* ── Kernel launch configs for this L ───────────────────────
         * Metropolis: 32×8 block, 2D grid covering L×L lattice.
         * Magnetization: 256-wide 1D grid capped at 4096 blocks.    */
        dim3 upd_block(32, 8);
        dim3 upd_grid((L + upd_block.x - 1) / upd_block.x,
                      (L + upd_block.y - 1) / upd_block.y);

        int mag_blocks = (N + 255) / 256;
        if (mag_blocks > 4096) mag_blocks = 4096;

        int fill_threads = 256;
        int fill_blocks  = (N + fill_threads - 1) / fill_threads;

        /* ── Temperature loop ────────────────────────────────────── */
        int t_idx;
        for (t_idx = 0; t_idx < n_T; ++t_idx) {

            double T    = T_MIN + t_idx * T_STEP;
            double beta = 1.0 / T;

            /* Upload LUT for this temperature */
            if (upload_lut(beta) != 0) {
                printf("  LUT upload failed T=%.4f\n", T); continue;
            }

            /* ── Reset lattice to ordered state (+1) ─────────────── */
            kernel_fill<<<fill_blocks, fill_threads>>>(d_lattice, N, (signed char)1);

            /* ── Thermalization ────────────────────────────────────
             * All kernel launches are queued on the default stream;
             * no intermediate sync — CUDA guarantees in-order exec. */
            int s;
            for (s = 0; s < N_THERM; ++s) {
                kernel_metropolis<<<upd_grid, upd_block>>>(d_lattice, d_rng_states, L, 0);
                kernel_metropolis<<<upd_grid, upd_block>>>(d_lattice, d_rng_states, L, 1);
            }

            /* ── Clear measurement accumulator ───────────────────── */
            CUDA_CHECK(cudaMemsetAsync(d_m_values, 0,
                                       (size_t)N_MEAS * sizeof(int), 0));

            /* ── Measurement sweeps ────────────────────────────────
             * Each meas sweep: one full Metropolis sweep (red+black)
             * then one magnetization reduction into d_m_values[s].
             * No sync inside the loop — all work is pipelined on the
             * default stream.                                        */
            for (s = 0; s < N_MEAS; ++s) {
                kernel_metropolis<<<upd_grid, upd_block>>>(d_lattice, d_rng_states, L, 0);
                kernel_metropolis<<<upd_grid, upd_block>>>(d_lattice, d_rng_states, L, 1);
                kernel_mag_reduce<<<mag_blocks, 256>>>(d_lattice, N, d_m_values, s);
            }

            /* ONE synchronization point per (L,T) ──────────────── */
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());

            /* ONE device-to-host copy per (L,T) ─────────────────── */
            CUDA_CHECK(cudaMemcpy(h_m_values, d_m_values,
                                  (size_t)N_MEAS * sizeof(int),
                                  cudaMemcpyDeviceToHost));

            /* ── Compute observables on host ─────────────────────── */
            double m_sum  = 0.0;
            double m2_sum = 0.0;
            for (s = 0; s < N_MEAS; ++s) {
                double m      = (double)h_m_values[s] / (double)N;
                double m_abs  = (m < 0.0) ? -m : m;
                m_sum  += m_abs;
                m2_sum += m * m;
            }
            double m_abs_mean = m_sum  / (double)N_MEAS;
            double m2_mean    = m2_sum / (double)N_MEAS;

            /* chi = beta * N * ( <m^2> - <|m|>^2 )  [FDT] */
            double chi = beta * (double)N * (m2_mean - m_abs_mean * m_abs_mean);

            fprintf(fout, "%d,%.4f,%.8f,%.8f\n", L, T, m_abs_mean, chi);
        }

        printf("  L=%4d complete  (%d T-points written)\n", L, n_T);

        cudaFree(d_lattice);
        cudaFree(d_rng_states);
    }

    cudaFree(d_m_values);
    free(h_m_values);
    fclose(fout);

    printf("\n================================================================\n");
    printf("  FSS sweep complete.\n");
    printf("  Output : fss_data.csv  (%d rows)\n", N_FSS_SIZES * n_T);
    printf("  Next   : fit chi_max(L) ~ L^(gamma/nu)  [expect 1.75 for 2D Ising]\n");
    printf("================================================================\n");
    return 0;
}
