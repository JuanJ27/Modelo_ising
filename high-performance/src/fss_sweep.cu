// =============================================================================
// fss_sweep.cu
// Phase 2.5: GPU-Accelerated Finite-Size Scaling (FSS)
//
// TARGET HARDWARE : GTX 1050 Ti (sm_61 / Pascal) | CUDA 12.6
//
// PHYSICS CONTRACT  (must reproduce thermo_sweep.cpp exactly at L=64):
//   Hamiltonian    : H = -J * Σ_{<i,j>} s_i·s_j    (J=1, H_ext=0)
//   Energy/site    : e = -J/N * Σ_i (s_i·s_right + s_i·s_down)
//                    [counts each bond exactly once; identical to the
//                     factor-1/2 formula used in thermo_sweep.cpp]
//   Magnetization  : m = (Σ_i s_i) / N   and   |m| = |Σ_i s_i| / N
//   FDT (per trial, time-average moments):
//     Cv  = N/T² * (⟨e²⟩_t − ⟨e⟩_t²)
//     χ   = N/T  * (⟨m²⟩_t − ⟨|m|⟩_t²)      ← uses |m|, NOT signed m
//   SEM (across K trials):
//     SEM(X) = σ_K(X)/√K,  σ²_K = ⟨X²⟩_K − ⟨X⟩_K²
//     Guard:  var = max(0, ...)  against floating-point underflow
//
// ALGORITHM:
//   Red-Black Checkerboard Metropolis.
//   Sites with (i+j)%2==0 (red) and (i+j)%2==1 (black) are updated in
//   alternating half-sweeps.  Within each half-sweep neighbours are frozen
//   (they belong to the opposite sublattice) → no data hazards, no atomics,
//   Detailed Balance preserved.  A full sweep = red half-sweep + black
//   half-sweep.
//
// RNG:
//   Philox-4x32-10 implemented as a pure inline __device__ function.
//   Counter-based: (trial, T_index, color, sweep, thread_idx) → unique
//   4×32 output without any stored state.  Passes BigCrush.
//   Constants: Salmon et al. 2011 (Random123 paper).
//
// REDUCTION STRATEGY:
//   Two-phase parallel reduction per production sweep:
//   Phase 1 (observe_kernel): each block reduces its N/BLOCK sites into
//     partial sums (ΣM, Σbonds) stored in a shared-memory tree, then
//     writes one pair of int32 values to a small device buffer.
//   Phase 2 (finalize_kernel): a single block sums all partial pairs,
//     derives the 5 per-sweep scalars (m, |m|, m², e, e²) and
//     atomically adds them to per-trial double accumulators d_acc[5].
//   After all PROD_SWEEPS, d_acc[5] is copied to host for FDT and SEM.
//
// T-SCHEDULE (identical to thermo_sweep.cpp):
//   Coarse ΔT=0.05 for T∈[1.0,2.05] ∪ [2.45,4.0]
//   Fine   ΔT=0.01 for T∈[2.10,2.40]   (critical window)
//
// OUTPUT CSV (matches Phase2_Validation.ipynb schema):
//   L,T,m_avg,m_err,chi,chi_err,e_avg,e_err,cv,cv_err
//
// COMPILE:
//   nvcc -O3 -arch=sm_61 -Wno-deprecated-gpu-targets \
//        high-performance/src/fss_sweep.cu -o fss_sim
// =============================================================================

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// CUDA error-check macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            printf("[CUDA ERROR] %s:%d  %s\n",                                \
                   __FILE__, __LINE__, cudaGetErrorString(_err));              \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// SIMULATION PARAMETERS
// ---------------------------------------------------------------------------
#define WARMUP_SWEEPS   2000
#define PROD_SWEEPS    10000
#define K_TRIALS          15    // ensemble size per (L, T)
#define BLOCK_SIZE       256    // threads per block (multiple of 32)
#define MEAS_STRIDE       10    // observe every N-th sweep; reduces reduction overhead ~90%
                                // Physical validity: autocorr time τ << MEAS_STRIDE at all T
                                // N_MEAS = PROD_SWEEPS / MEAS_STRIDE = 1000 independent samples

// Onsager exact T_c for 2D square lattice, J=1
#define T_CRIT 2.26918531421288f

// ---------------------------------------------------------------------------
// Boltzmann LUT in GPU constant memory (per-temperature, updated on host).
//
// lut[k] = min(1, exp(-ΔE/T))  where the index k encodes ΔE:
//   ΔE = 2J·s_i·(Σ_nn s_j),   s_i = ±1,  nn_sum ∈ {-4,-2,0,+2,+4}
//   snn_product = s_i * nn_sum ∈ {-4,-2,0,+2,+4}
//   k = snn_product/2 + 2  ∈ {0,1,2,3,4}   (ΔE = 4J·(k-2))
// k=0,1,2 → ΔE≤0 → lut=1.0f (always accept)
// k=3     → ΔE=+4J → exp(-4/T)
// k=4     → ΔE=+8J → exp(-8/T)
// ---------------------------------------------------------------------------
__constant__ float c_boltz[5];

// =============================================================================
// PHILOX-4×32-10 COUNTER-BASED RNG (Salmon et al., SC 2011)
//
// Fully stateless: call philox4x32(counter, key) → 4 pseudorandom uint32.
// Thread-unique streams are addressed by (sweep, thread_idx) as counter
// and (trial, t_idx, color) as key.  No RNG state stored between calls.
// =============================================================================
static const uint32_t PHILOX_M0 = 0xD2511F53u;
static const uint32_t PHILOX_M1 = 0xCD9E8D57u;
static const uint32_t PHILOX_W0 = 0x9E3779B9u;   // Golden-ratio weyl constant
static const uint32_t PHILOX_W1 = 0xBB67AE85u;   // sqrt(3)-1 weyl constant

struct Philox4x32State {
    uint32_t ctr[4];
    uint32_t key[2];
};

__device__ __forceinline__
void philox_round(uint32_t ctr[4], uint32_t key[2])
{
    // Multiply-high / multiply-low via CUDA PTX intrinsic
    uint32_t hi0 = __umulhi(PHILOX_M0, ctr[0]);
    uint32_t lo0 =           PHILOX_M0 * ctr[0];
    uint32_t hi1 = __umulhi(PHILOX_M1, ctr[2]);
    uint32_t lo1 =           PHILOX_M1 * ctr[2];

    ctr[0] = hi1 ^ ctr[1] ^ key[0];
    ctr[1] = lo1;
    ctr[2] = hi0 ^ ctr[3] ^ key[1];
    ctr[3] = lo0;
}

__device__ __forceinline__
void philox_bumpkey(uint32_t key[2])
{
    key[0] += PHILOX_W0;
    key[1] += PHILOX_W1;
}

// Returns 4 pseudorandom uint32 values for the given (counter, key) address.
__device__ __forceinline__
void philox4x32_10(uint32_t ctr[4], uint32_t key[2])
{
    // Unrolled 10 Philox rounds
    philox_round(ctr, key); philox_bumpkey(key);
    philox_round(ctr, key); philox_bumpkey(key);
    philox_round(ctr, key); philox_bumpkey(key);
    philox_round(ctr, key); philox_bumpkey(key);
    philox_round(ctr, key); philox_bumpkey(key);
    philox_round(ctr, key); philox_bumpkey(key);
    philox_round(ctr, key); philox_bumpkey(key);
    philox_round(ctr, key); philox_bumpkey(key);
    philox_round(ctr, key); philox_bumpkey(key);
    philox_round(ctr, key);   // final round — key NOT bumped
}

// Convenience: generate a single uniform float in [0, 1) for one thread.
// Parameters uniquely address one stream per (trial, t_idx, color, sweep, thread).
__device__ __forceinline__
float philox_uniform(uint32_t trial, uint32_t t_idx,
                     uint32_t color,  uint32_t sweep,
                     uint32_t thread_idx)
{
    uint32_t ctr[4] = { color, sweep, thread_idx, 0u };
    uint32_t key[2] = { trial, t_idx };
    philox4x32_10(ctr, key);
    // Map ctr[0] to float in [0, 1) via 24-bit mantissa (avoids rounding to 1.0)
    return (float)(ctr[0] >> 8) * (1.0f / (float)(1 << 24));
}

// =============================================================================
// KERNEL 1: RED-BLACK CHECKERBOARD METROPOLIS UPDATE
//
// One thread per site in the active sublattice (N/2 threads total).
// Periodic boundary conditions via integer modulo.
//
// Thread mapping (for half-lattice index `tid` and sublattice `color`):
//   row       = tid / (L/2)
//   col_half  = tid % (L/2)
//   col       = 2*col_half + ((row + color) & 1)
//
// Performance notes:
//   • Kernel is fully loop-free — the compiler emits maximum ILP automatically.
//   • Philox4x32_10 rounds are explicitly unrolled (10 inline calls = zero
//     branch mispredictions, all operands live in registers).
//   • The 4 neighbour reads hit L1/L2 cache on Pascal since the spin array
//     is accessed with unit or L-stride patterns.
// =============================================================================
__global__
void redblack_kernel(int8_t* __restrict__ spins,
                     int L,
                     int color,      // 0 = red, 1 = black
                     uint32_t trial,
                     uint32_t t_idx,
                     uint32_t sweep)
{
    const int half_N  = (L * L) >> 1;
    const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= half_N) return;

    // Map linear tid → (row, col) in the full L×L lattice
    const int row      = tid / (L >> 1);
    const int col_half = tid % (L >> 1);
    const int col      = 2 * col_half + ((row + color) & 1);

    // Periodic-boundary neighbours (pre-compute modulo for L)
    const int row_up   = (row == 0)     ? L - 1 : row - 1;
    const int row_dn   = (row == L - 1) ? 0     : row + 1;
    const int col_lt   = (col == 0)     ? L - 1 : col - 1;
    const int col_rt   = (col == L - 1) ? 0     : col + 1;

    const int si = row    * L + col;
    const int8_t s = spins[si];

    // Neighbour sum ∈ {-4,-2,0,+2,+4}
    const int nn_sum = (int)spins[row_up * L + col]
                     + (int)spins[row_dn * L + col]
                     + (int)spins[row    * L + col_lt]
                     + (int)spins[row    * L + col_rt];

    // LUT index: k = s * nn_sum / 2 + 2 ∈ {0,1,2,3,4}
    const int k = ((int)s * nn_sum) / 2 + 2;

    // Metropolis acceptance via Boltzmann LUT
    const float r = philox_uniform(trial, t_idx, (uint32_t)color,
                                   sweep, (uint32_t)tid);
    if (r < c_boltz[k]) {
        spins[si] = -s;   // flip
    }
}

// =============================================================================
// KERNEL 2 (Phase 1): PER-BLOCK REDUCTION OF MAGNETIZATION AND BOND ENERGY
//
// Each thread reads one site; computes:
//   m_i    = spin value (+1 or -1) — contributes to ΣM
//   bond_i = s_i·s_right + s_i·s_down  (counts each bond exactly once)
//
// REDUCTION STRATEGY — Two-level warp-shuffle tree (replaces shared-mem tree):
//   Level 1: __shfl_down_sync across 32 lanes (5 rounds, 0 syncthreads)
//   Level 2: 8 warp leaders → 64-byte __shared__ staging → warp 0 reduces
//            with 3 more shuffle rounds (1 syncthreads total vs. 8 in old code)
//   Result: one int32 pair per block written to d_pM/d_pbonds.
//
//   Shared memory footprint: 8+8 int32 = 64 bytes/block vs. 2048 bytes/block
//   → higher SM occupancy, better register file pressure on Pascal.
// =============================================================================
__global__
void observe_kernel(const int8_t* __restrict__ spins,
                    int L,
                    int32_t* __restrict__ d_pM,
                    int32_t* __restrict__ d_pbonds)
{
    const int N    = L * L;
    const int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int ltid = threadIdx.x;
    const int lane   = ltid & 31;           // lane within warp [0,31]
    const int warpId = ltid >> 5;           // warp index within block [0,7]

    int32_t local_M = 0, local_bonds = 0;

    if (tid < N) {
        const int row = tid / L;
        const int col = tid % L;
        const int8_t s = spins[tid];
        const int col_rt = (col == L - 1) ? 0 : col + 1;
        const int row_dn = (row == L - 1) ? 0 : row + 1;
        local_M     = (int32_t)s;
        local_bonds = (int32_t)s * (int32_t)spins[row    * L + col_rt]
                    + (int32_t)s * (int32_t)spins[row_dn * L + col];
    }

    // ── Level 1: intra-warp reduction — no shared memory, no __syncthreads ──
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_M     += __shfl_down_sync(0xFFFFFFFFu, local_M,     offset);
        local_bonds += __shfl_down_sync(0xFFFFFFFFu, local_bonds, offset);
    }

    // ── Level 2: cross-warp reduction through 64-byte staging buffer ──────
    // BLOCK_SIZE=256 → 8 warps → 8 slots each
    __shared__ int32_t ws_M[8], ws_B[8];
    if (lane == 0) {
        ws_M[warpId] = local_M;
        ws_B[warpId] = local_bonds;
    }
    __syncthreads();   // one barrier (vs. 8 in the old binary tree)

    if (warpId == 0) {
        // Warp 0 loads 8 warp-leader values and reduces with 3 shuffle rounds
        int32_t wM = (lane < 8) ? ws_M[lane] : 0;
        int32_t wB = (lane < 8) ? ws_B[lane] : 0;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            wM += __shfl_down_sync(0xFFFFFFFFu, wM, offset);
            wB += __shfl_down_sync(0xFFFFFFFFu, wB, offset);
        }
        if (lane == 0) {
            d_pM[blockIdx.x]     = wM;
            d_pbonds[blockIdx.x] = wB;
        }
    }
}

// =============================================================================
// KERNEL 3 (Phase 2): FINALIZE REDUCTION & ACCUMULATE INTO TRIAL SUMS
//
// Single-block kernel.  Sums n_blocks partial values from d_pM/d_pbonds,
// derives per-measurement scalars (m, |m|, m², e, e²) and adds to d_acc[5].
//
// Uses same two-level warp-shuffle strategy as observe_kernel:
//   Each thread strips-loops over d_pM/d_pbonds (grid-stride), then
//   warp shuffles reduce to 8 warp leaders, then warp 0 final-reduces.
//   Only 1 __syncthreads total (vs. log2(BLOCK_SIZE)=8 in old code).
// =============================================================================
__global__
void finalize_kernel(const int32_t* __restrict__ d_pM,
                     const int32_t* __restrict__ d_pbonds,
                     int n_blocks,
                     int N,
                     double* __restrict__ d_acc)  // [5] per-trial accumulators
{
    const int ltid   = threadIdx.x;
    const int lane   = ltid & 31;
    const int warpId = ltid >> 5;

    // Strip-loop: each thread accumulates a slice of the partial-sum array
    int32_t sum_M = 0, sum_bonds = 0;
    for (int i = ltid; i < n_blocks; i += BLOCK_SIZE) {
        sum_M     += d_pM[i];
        sum_bonds += d_pbonds[i];
    }

    // ── Level 1: intra-warp reduction ──────────────────────────────────────
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_M     += __shfl_down_sync(0xFFFFFFFFu, sum_M,     offset);
        sum_bonds += __shfl_down_sync(0xFFFFFFFFu, sum_bonds, offset);
    }

    // ── Level 2: cross-warp reduction ──────────────────────────────────────
    __shared__ int32_t ws_M[8], ws_B[8];
    if (lane == 0) {
        ws_M[warpId] = sum_M;
        ws_B[warpId] = sum_bonds;
    }
    __syncthreads();

    if (warpId == 0) {
        int32_t wM = (lane < 8) ? ws_M[lane] : 0;
        int32_t wB = (lane < 8) ? ws_B[lane] : 0;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            wM += __shfl_down_sync(0xFFFFFFFFu, wM, offset);
            wB += __shfl_down_sync(0xFFFFFFFFu, wB, offset);
        }
        if (lane == 0) {
            const double inv_N = 1.0 / (double)N;
            const double M_tot = (double)wM;
            const double B_tot = (double)wB;
            const double m     =  M_tot * inv_N;
            const double abs_m = (M_tot >= 0.0 ? M_tot : -M_tot) * inv_N;
            const double m2    = m * m;
            // Bond formula: e = -J * Σ(right+down bonds) / N   (J=1)
            const double e     = -B_tot * inv_N;
            const double e2    = e * e;
            // sm_61 supports atomicAdd(double*,...) natively
            atomicAdd(&d_acc[0], m);
            atomicAdd(&d_acc[1], abs_m);
            atomicAdd(&d_acc[2], m2);
            atomicAdd(&d_acc[3], e);
            atomicAdd(&d_acc[4], e2);
        }
    }
}

// =============================================================================
// HOST HELPERS
// =============================================================================

// Rebuild host-side Boltzmann table and upload to __constant__ memory.
// Matches MetropolisEngine::rebuild_lut() from the CPU reference.
static void upload_boltz_lut(float T)
{
    float h_lut[5];
    for (int k = 0; k < 5; ++k) {
        const float dE = 4.0f * (float)(k - 2);  // ΔE/J = 4*(k-2)
        h_lut[k] = (dE <= 0.0f) ? 1.0f : expf(-dE / T);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_boltz, h_lut, 5 * sizeof(float)));
}

// Replicate thermo_sweep.cpp T-schedule exactly:
//   coarse ΔT=0.05 outside [2.10,2.40], fine ΔT=0.01 inside.
static void make_temperature_schedule(float* T_arr, int* n_temps)
{
    int n = 0;
    // Coarse pre-critical: 1.00 → 2.05
    for (int k = 0; ; ++k) {
        float T = 1.0f + k * 0.05f;
        if (T > 2.09f) break;
        T_arr[n++] = T;
    }
    // Fine critical window: 2.10 → 2.40
    for (int k = 0; ; ++k) {
        float T = 2.10f + k * 0.01f;
        if (T > 2.405f) break;
        T_arr[n++] = T;
    }
    // Coarse post-critical: 2.45 → 4.00
    for (int k = 0; ; ++k) {
        float T = 2.45f + k * 0.05f;
        if (T > 4.005f) break;
        T_arr[n++] = T;
    }
    *n_temps = n;
}

// =============================================================================
// RUN ONE (L, T) TEMPERATURE POINT — K_TRIALS independent Markov chains.
// Writes one CSV row (without L column prefix — caller prepends L).
// =============================================================================
static void run_temp_point(int L, int N, float T,
                            uint32_t t_idx,
                            int8_t*   d_spins,
                            int32_t*  d_pM,
                            int32_t*  d_pbonds,
                            double*   d_acc,
                            int       n_obs_blocks,
                            double*   out)   // [9]: m,m_err,chi,chi_err,e,e_err,cv,cv_err (8 values + unused)
{
    // Ensemble accumulators (across K_TRIALS)
    double ens_m    = 0.0, ens_m2    = 0.0;
    double ens_absm = 0.0, ens_absm2 = 0.0;
    double ens_chi  = 0.0, ens_chi2  = 0.0;
    double ens_e    = 0.0, ens_e2    = 0.0;
    double ens_cv   = 0.0, ens_cv2   = 0.0;

    upload_boltz_lut(T);

    const int half_N          = N >> 1;
    const int rb_blocks       = (half_N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // N_MEAS: number of actual measurement points per trial.
    // Striding reduces kernel-launch overhead by (MEAS_STRIDE-1)/MEAS_STRIDE = 90%
    // while keeping 1000 effectively-independent samples for the FDT.
    const int    N_MEAS = PROD_SWEEPS / MEAS_STRIDE;
    const double inv_P  = 1.0 / (double)N_MEAS;

    double sum_gflips = 0.0;   // accumulate GFlips/s across trials for reporting

    for (int trial = 0; trial < K_TRIALS; ++trial) {

        // ── Initialize all-spin-up lattice (ordered initial condition) ──────
        // Matches the CPU reference: IsingLattice(L, Square2D, +1)
        CUDA_CHECK(cudaMemset(d_spins, 1, N * sizeof(int8_t)));

        // ── Reset per-trial accumulators ───────────────────────────────────
        CUDA_CHECK(cudaMemset(d_acc, 0, 5 * sizeof(double)));

        // ── THERMALIZATION: discard WARMUP_SWEEPS sweeps ───────────────────
        for (int s = 0; s < WARMUP_SWEEPS; ++s) {
            redblack_kernel<<<rb_blocks, BLOCK_SIZE>>>(
                d_spins, L, 0, (uint32_t)trial, t_idx, (uint32_t)s);
            redblack_kernel<<<rb_blocks, BLOCK_SIZE>>>(
                d_spins, L, 1, (uint32_t)trial, t_idx, (uint32_t)s);
        }

        // ── PRODUCTION: update every sweep, measure every MEAS_STRIDE sweeps.
        //    CUDA event pair brackets the entire production phase for timing.
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
        cudaEventRecord(ev0);

        for (int s = 0; s < PROD_SWEEPS; ++s) {
            const uint32_t sw = (uint32_t)(WARMUP_SWEEPS + s);

            redblack_kernel<<<rb_blocks, BLOCK_SIZE>>>(
                d_spins, L, 0, (uint32_t)trial, t_idx, sw);
            redblack_kernel<<<rb_blocks, BLOCK_SIZE>>>(
                d_spins, L, 1, (uint32_t)trial, t_idx, sw);

            // Measure only every MEAS_STRIDE sweeps — 90% fewer reducer launches.
            // Physical: MEAS_STRIDE=10 >> τ_int everywhere except deep in the
            // critical window at large L, where it provides further decorrelation.
            if (s % MEAS_STRIDE == 0) {
                // Phase-1: N-wide warp-shuffle block reductions → d_pM/d_pbonds
                observe_kernel<<<n_obs_blocks, BLOCK_SIZE>>>(
                    d_spins, L, d_pM, d_pbonds);

                // Phase-2: single-block warp-shuffle finalization → d_acc[5]
                finalize_kernel<<<1, BLOCK_SIZE>>>(
                    d_pM, d_pbonds, n_obs_blocks, N, d_acc);
            }
        }

        cudaEventRecord(ev1);
        CUDA_CHECK(cudaDeviceSynchronize());
        float ms_trial = 0.0f;
        cudaEventElapsedTime(&ms_trial, ev0, ev1);
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);

        // GigaFlips/s = spin-flip attempts / time
        //   attempts = N sites × PROD_SWEEPS full sweeps (2 half-sweeps each)
        //   time     = ms_trial / 1000  seconds
        const double gflips = (double)N * (double)PROD_SWEEPS
                              / ((double)ms_trial * 1.0e6);
        sum_gflips += gflips;

        // ── Copy per-trial accumulators to host ────────────────────────────
        double h_acc[5];
        CUDA_CHECK(cudaMemcpy(h_acc, d_acc, 5 * sizeof(double),
                              cudaMemcpyDeviceToHost));

        // ── Single-trial FDT observables (time-averages over PROD_SWEEPS) ──
        //
        //   avg_* = accumulated sum / PROD_SWEEPS
        //   Cv  = N/T² * (⟨e²⟩_t − ⟨e⟩_t²)
        //   χ   = N/T  * (⟨m²⟩_t − ⟨|m|⟩_t²)   ← defined with |m|
        //
        const double avg_m    = h_acc[0] * inv_P;
        const double avg_absm = h_acc[1] * inv_P;
        const double avg_m2   = h_acc[2] * inv_P;
        const double avg_e    = h_acc[3] * inv_P;
        const double avg_e2   = h_acc[4] * inv_P;

        const double T_d = (double)T;
        const double trial_cv  = (double)N / (T_d * T_d)
                                 * (avg_e2  - avg_e    * avg_e);
        const double trial_chi = (double)N / T_d
                                 * (avg_m2  - avg_absm * avg_absm);

        // ── Ensemble first- and second-moment accumulators ─────────────────
        ens_m    += avg_m;       ens_m2    += avg_m    * avg_m;
        ens_absm += avg_absm;    ens_absm2 += avg_absm * avg_absm;
        ens_chi  += trial_chi;   ens_chi2  += trial_chi * trial_chi;
        ens_e    += avg_e;       ens_e2    += avg_e    * avg_e;
        ens_cv   += trial_cv;    ens_cv2   += trial_cv * trial_cv;
    }

    // ── Ensemble mean ────────────────────────────────────────────────────────
    const double inv_K     = 1.0 / (double)K_TRIALS;
    const double mean_absm = ens_absm * inv_K;
    const double mean_chi  = ens_chi  * inv_K;
    const double mean_e    = ens_e    * inv_K;
    const double mean_cv   = ens_cv   * inv_K;

    // ── SEM: σ_K(X)/√K  with max(0,·) variance guard ─────────────────────
    //   Matches thermo_sweep.cpp sem() lambda exactly.
    #define SEM(sum_x, sum_x2) \
        (sqrt(fmax(0.0, (sum_x2)*inv_K - ((sum_x)*inv_K)*((sum_x)*inv_K)) * inv_K))

    const double sem_absm = SEM(ens_absm, ens_absm2);
    const double sem_chi  = SEM(ens_chi,  ens_chi2);
    const double sem_e    = SEM(ens_e,    ens_e2);
    const double sem_cv   = SEM(ens_cv,   ens_cv2);
    #undef SEM

    // ── Pack output: [m_avg, m_err, chi, chi_err, e_avg, e_err, cv, cv_err, gflips]
    out[0] = mean_absm;
    out[1] = sem_absm;
    out[2] = mean_chi;
    out[3] = sem_chi;
    out[4] = mean_e;
    out[5] = sem_e;
    out[6] = mean_cv;
    out[7] = sem_cv;
    out[8] = sum_gflips / (double)K_TRIALS;   // avg GFlips/s over all trials
}

// =============================================================================
// MAIN
// =============================================================================
int main(void)
{
    // =========================================================================
    // PARITY CHECK FLAG
    //
    //   true  → run only L=64, K=15 for cross-validation against
    //            the CPU thermo_sweep.cpp reference at the same parameters.
    //            Output identical CSV schema; compare each T-point to within
    //            3·SEM to confirm statistical parity before scaling up.
    //
    //   false → full FSS run: L ∈ {64, 128, 256, 512, 1024}, K=15 each.
    // =========================================================================
    const int PARITY_CHECK_ONLY = 0;   // ← SET TO 0 FOR FULL FSS RUN

    // -------------------------------------------------------------------------
    // Lattice sizes for the full FSS sweep
    // -------------------------------------------------------------------------
    const int all_L[]    = { 64, 128, 256, 512, 1024 };
    const int n_all_L    = 5;
    const int parity_L[] = { 64 };
    const int n_parity_L = 1;

    const int*  L_arr  = PARITY_CHECK_ONLY ? parity_L : all_L;
    const int   n_L    = PARITY_CHECK_ONLY ? n_parity_L : n_all_L;

    // -------------------------------------------------------------------------
    // Temperature schedule (shared across all L — required for FSS data collapse)
    // -------------------------------------------------------------------------
    float T_arr[200];
    int   n_temps = 0;
    make_temperature_schedule(T_arr, &n_temps);

    // -------------------------------------------------------------------------
    // Device query
    // -------------------------------------------------------------------------
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=================================================================\n");
    printf("  Ising-Dynamics | Phase 2.5 GPU-Accelerated FSS\n");
    printf("  Device   : %s  (sm_%d%d)\n",
           prop.name, prop.major, prop.minor);
    printf("  Mode     : %s\n",
           PARITY_CHECK_ONLY ? "PARITY CHECK (L=64 only)" : "FULL FSS RUN");
    printf("  L values : ");
    for (int i = 0; i < n_L; ++i) printf("%d%s", L_arr[i], i < n_L-1 ? ", " : "\n");
    printf("  Trials/T : %d\n",  K_TRIALS);
    printf("  Warm-up  : %d sweeps (discarded)\n", WARMUP_SWEEPS);
    printf("  Prod     : %d sweeps (%d measured, stride=%d)\n",
           PROD_SWEEPS, PROD_SWEEPS / MEAS_STRIDE, MEAS_STRIDE);
    printf("  T-points : %d\n",  n_temps);
    printf("  T_c exact: %.7f\n", (double)T_CRIT);
    printf("=================================================================\n\n");

    // -------------------------------------------------------------------------
    // Open CSV
    // -------------------------------------------------------------------------
    const char* CSV_PATH = "/app/Ising-Dynamics/high-performance/src/fss_data.csv";
    FILE* csv = fopen(CSV_PATH, "w");
    if (!csv) {
        printf("[ERROR] Cannot open %s\n", CSV_PATH);
        return 1;
    }
    // Schema — strict superset of Phase2_Validation.ipynb (adds L column)
    fprintf(csv, "L,T,m_avg,m_err,chi,chi_err,e_avg,e_err,cv,cv_err\n");
    fflush(csv);

    // -------------------------------------------------------------------------
    // Device memory allocation (sized for the largest L in this run)
    // -------------------------------------------------------------------------
    const int L_max = L_arr[n_L - 1];
    const int N_max = L_max * L_max;
    const int n_obs_blocks_max = (N_max + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int8_t*  d_spins  = NULL;
    int32_t* d_pM     = NULL;
    int32_t* d_pbonds = NULL;
    double*  d_acc    = NULL;

    CUDA_CHECK(cudaMalloc(&d_spins,  N_max             * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_pM,     n_obs_blocks_max  * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_pbonds, n_obs_blocks_max  * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_acc,    5                 * sizeof(double)));

    // -------------------------------------------------------------------------
    // Main nested loop: L → T
    // -------------------------------------------------------------------------
    for (int li = 0; li < n_L; ++li) {
        const int L = L_arr[li];
        const int N = L * L;
        const int n_obs_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

        printf("-- L = %d  (N = %d)  ------------------------------------------\n",
               L, N);

        for (int ti = 0; ti < n_temps; ++ti) {
            const float T = T_arr[ti];

            double obs[9];   // [m_avg, m_err, chi, chi_err, e_avg, e_err, cv, cv_err, gflips]
            run_temp_point(L, N, T, (uint32_t)ti,
                           d_spins, d_pM, d_pbonds, d_acc,
                           n_obs_blocks, obs);

            // Write CSV row — schema: L,T,m_avg,m_err,chi,chi_err,e_avg,e_err,cv,cv_err
            fprintf(csv, "%d,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n",
                    L, (double)T,
                    obs[0], obs[1],   // m_avg,  m_err
                    obs[2], obs[3],   // chi,    chi_err
                    obs[4], obs[5],   // e_avg,  e_err
                    obs[6], obs[7]);  // cv,     cv_err
            fflush(csv);

            printf("  [%3d/%d]  T=%.3f  <|m|>=%.4f ±%.4f  chi=%7.2f ±%.2f"
                   "  e=%.4f  cv=%.3f  %.2f GF/s\n",
                   ti + 1, n_temps, (double)T,
                   obs[0], obs[1], obs[2], obs[3], obs[4], obs[6], obs[8]);
            fflush(stdout);
        }
        printf("\n");
    }

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    fclose(csv);
    cudaFree(d_spins);
    cudaFree(d_pM);
    cudaFree(d_pbonds);
    cudaFree(d_acc);

    printf("=================================================================\n");
    printf("  Done.  CSV written to: %s\n", CSV_PATH);
    printf("=================================================================\n");

    return 0;
}
