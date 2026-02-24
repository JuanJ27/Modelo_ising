// =============================================================================
// 03_random_bench.cpp
// Phase 1.8: The C++ Awakening — RNG Benchmark for Ising Simulations
//
// PURPOSE:
//   Benchmark and compare three random number generation strategies relevant
//   to Monte Carlo physics simulations. Establish the correct RNG choice
//   for the Ising v2.0 C++ engine before writing the Metropolis kernel.
//
// COMPILATION:
//   /usr/bin/g++ -std=c++17 -O2 -Wall -Wextra -o 03_random_bench 03_random_bench.cpp
//
// Author : Physics Researcher Agent — Ising v2.0 Migration
// =============================================================================

#include <cstdint>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cstdlib>    // rand(), srand()
#include <ctime>
#include <vector>
#include <numeric>    // std::accumulate
#include <cmath>      // std::sqrt
#include <string>

// =============================================================================
// [DESIGN NOTE 1]: Why RNG Period Matters for Monte Carlo Physics
//
//   A Monte Carlo Ising simulation on an L=1000 lattice (N=10^6 sites)
//   running 10^5 sweeps consumes:
//
//     N * sweeps = 10^6 * 10^5 = 10^11 random numbers
//
//   The generator's PERIOD must vastly exceed this to guarantee that the
//   sequence does not repeat (a "cycle") during the simulation. Repeating
//   cycles introduce artificial correlations into the Markov chain,
//   biasing measured observables (magnetization, susceptibility) and
//   invalidating the Detailed Balance proof.
//
//   Generator periods:
//     Legacy rand()           : 2^31 - 1 ≈  2.1 * 10^9   (exhausted in ~21 sec)
//     std::minstd_rand (LCG)  : 2^31 - 1 ≈  2.1 * 10^9
//     std::mt19937 (32-bit)   : 2^19937 - 1               (effectively infinite)
//     std::mt19937_64 (64-bit): 2^19937 - 1               (effectively infinite)
//
//   For any lattice size L <= 10^4 and any simulation length physically
//   meaningful, mt19937 never cycles. This is why the Newman & Barkema Monte
//   Carlo textbook recommends it as the default for statistical physics.
// =============================================================================

// =============================================================================
// [DESIGN NOTE 2]: Why `rand() % 2` is Biased
//
//   The classic C idiom for a coin flip:
//     int flip = rand() % 2;
//
//   Two interrelated problems:
//
//   (A) MODULO BIAS:
//     rand() returns integers in [0, RAND_MAX]. If RAND_MAX+1 is not
//     divisible by 2, some outcomes appear more often. For RAND_MAX=32767
//     (common on older systems), values 0 and 1 are equally likely — no bias.
//     But for arbitrary ranges (e.g. rand() % 6 for a die), RAND_MAX+1 = 32768
//     is not divisible by 6, so 0-1-2-3 appear slightly more often than 4-5.
//     In a Metropolis acceptance test using rand() % N for large N, this
//     modulo bias shifts the site-selection probability, breaking ergodicity.
//
//   (B) LOW-BIT QUALITY:
//     Many Linear Congruential Generator (LCG) implementations (which underlie
//     most `rand()` implementations) have low entropy in the least-significant
//     bits. The low-bit pattern cycles with a much shorter period than RAND_MAX.
//     Using `& 1` or `% 2` directly reads the lowest bit — precisely the worst
//     bit in the generator. This can produce measurable spin correlations in
//     Ising 2D simulations near criticality.
//
//   std::uniform_int_distribution<int>(0, 1) uses rejection sampling internally
//   to guarantee perfectly uniform probabilities with no modulo bias.
// =============================================================================

// =============================================================================
// [DESIGN NOTE 3]: RNG State Management — Never Re-Seed in the Hot Loop
//
//   The Metropolis kernel structure is:
//
//     for each sweep:
//       for each site:
//         i = sample_site(rng)          // consumes 1 RNG state
//         if dE > 0:
//           accept = (uniform(rng) < exp(-beta*dE))  // consumes 1 RNG state
//
//   CRITICAL RULE: rng is constructed ONCE before the outer loop and passed
//   by reference into the kernel. It is NEVER re-created or re-seeded inside.
//
//   Why this matters:
//
//   1. RE-SEEDING COST: std::mt19937 initialization requires "warming up" the
//      state over 624 words (the state array). Re-seeding every sweep adds
//      O(N * sweeps * 624) initializations — catastrophic for performance.
//
//   2. STATISTICAL VALIDITY: If you re-seed every sweep with a time-based
//      seed (e.g., time(nullptr)), successive sweeps can receive the same
//      seed at resolution < 1 second. The simulation then replays the same
//      random sequence for multiple sweeps — completely destroying ergodicity
//      and Detailed Balance.
//
//   3. THREAD SAFETY: In parallel runs (OpenMP/TBB), each thread must own its
//      own rng instance (not shared) but each is seeded once at
//      thread-initialization time using independent seeds (e.g., splitmix64
//      or seed_seq with per-thread offsets). This is handled in Phase 2.
//
//   CORRECT PATTERN (C++ idiomatic):
//     std::random_device rd;
//     std::mt19937 rng(rd());   // seed ONCE — before any loop
//     std::uniform_real_distribution<double> uniform(0.0, 1.0);
//     for (int sweep = 0; sweep < N_sweeps; ++sweep)
//       metropolis_sweep(rng, uniform, lattice, ...);  // pass rng by reference
// =============================================================================


// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------
using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;
using Ns    = std::chrono::duration<double, std::nano>;

struct BenchResult {
    std::string label;
    double      time_ms;
    double      mrps;          // Millions of random numbers per second
    uint64_t    checksum;      // XOR accumulator to prevent dead-code elimination
};

void print_result(const BenchResult& r) {
    std::cout << "  " << std::left  << std::setw(38) << r.label
              << std::right << std::setw(10) << std::fixed << std::setprecision(2)
              << r.time_ms << " ms    "
              << std::setw(8) << std::setprecision(1) << r.mrps << " MRPS\n";
}

// ---------------------------------------------------------------------------
// Benchmark A: Legacy C rand()
// ---------------------------------------------------------------------------
BenchResult bench_c_rand(long N, unsigned seed) {
    std::srand(seed);

    uint64_t chk  = 0;
    auto     t0   = Clock::now();

    for (long i = 0; i < N; ++i) {
        chk ^= static_cast<uint64_t>(std::rand());
    }

    double ms    = Ms(Clock::now() - t0).count();
    double mrps  = (N / 1e6) / (ms / 1e3);
    return {"Legacy C rand()", ms, mrps, chk};
}

// ---------------------------------------------------------------------------
// Benchmark B: std::mt19937 — raw 32-bit draw
// ---------------------------------------------------------------------------
BenchResult bench_mt19937_raw(long N, std::mt19937& rng) {
    uint64_t chk = 0;
    auto     t0  = Clock::now();

    for (long i = 0; i < N; ++i) {
        chk ^= static_cast<uint64_t>(rng());
    }

    double ms    = Ms(Clock::now() - t0).count();
    double mrps  = (N / 1e6) / (ms / 1e3);
    return {"std::mt19937 (raw uint32)", ms, mrps, chk};
}

// ---------------------------------------------------------------------------
// Benchmark C: std::mt19937 + uniform_int_distribution<int>(0,1)
//   Simulates the binary site-selection decision in Metropolis.
// ---------------------------------------------------------------------------
BenchResult bench_mt19937_uid(long N, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, 1);
    uint64_t chk = 0;
    auto     t0  = Clock::now();

    for (long i = 0; i < N; ++i) {
        chk ^= static_cast<uint64_t>(dist(rng));
    }

    double ms    = Ms(Clock::now() - t0).count();
    double mrps  = (N / 1e6) / (ms / 1e3);
    return {"mt19937 + uniform_int_dist(0,1)", ms, mrps, chk};
}

// ---------------------------------------------------------------------------
// Benchmark D: std::mt19937 + uniform_real_distribution<double>(0,1)
//   The actual Metropolis acceptance test: `u < exp(-beta*dE)`.
// ---------------------------------------------------------------------------
BenchResult bench_mt19937_real(long N, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    uint64_t chk = 0;
    auto     t0  = Clock::now();

    for (long i = 0; i < N; ++i) {
        // Reinterpret the double bits into uint64 for checksum without UB
        double val = dist(rng);
        uint64_t bits;
        __builtin_memcpy(&bits, &val, sizeof(bits));
        chk ^= bits;
    }

    double ms    = Ms(Clock::now() - t0).count();
    double mrps  = (N / 1e6) / (ms / 1e3);
    return {"mt19937 + uniform_real_dist(0,1)", ms, mrps, chk};
}

// ---------------------------------------------------------------------------
// Benchmark E: std::mt19937_64 — 64-bit Mersenne Twister
//   Preferred for large-lattice simulations where site indices need >32 bits.
// ---------------------------------------------------------------------------
BenchResult bench_mt19937_64(long N, std::mt19937_64& rng) {
    uint64_t chk = 0;
    auto     t0  = Clock::now();

    for (long i = 0; i < N; ++i) {
        chk ^= rng();
    }

    double ms    = Ms(Clock::now() - t0).count();
    double mrps  = (N / 1e6) / (ms / 1e3);
    return {"std::mt19937_64 (raw uint64)", ms, mrps, chk};
}


// ---------------------------------------------------------------------------
// Statistical quality check: chi-squared test on a 0/1 coin flip sequence.
//   A fair coin over N flips should give N/2 ± sqrt(N)/2 heads.
//   We flag any generator whose deviation exceeds 3 sigma.
// ---------------------------------------------------------------------------
template<typename RNG, typename Dist>
void chi2_coin(const char* label, long N, RNG& rng, Dist& dist) {
    long heads = 0;
    for (long i = 0; i < N; ++i)
        if (dist(rng) & 1) ++heads;          // bit 0 of the raw output

    double expected = N / 2.0;
    double sigma    = std::sqrt(N / 4.0);    // sqrt(N*p*(1-p)), p=0.5
    double z        = (heads - expected) / sigma;

    std::cout << "  " << std::left << std::setw(38) << label
              << "heads=" << heads
              << "  z-score=" << std::fixed << std::setprecision(3) << z
              << (std::abs(z) < 3.0 ? "  [OK]" : "  [SUSPICIOUS]")
              << "\n";
}


int main() {
    constexpr long N_BENCH  = 100'000'000L;   // 100 million draws
    constexpr long N_CHISQ  =  10'000'000L;   // 10 million for quality test

    // -------------------------------------------------------------------------
    // Seeding: std::random_device reads from /dev/urandom (non-deterministic).
    // One call per generator — NEVER call rd() inside any loop.
    // -------------------------------------------------------------------------
    std::random_device rd;
    const unsigned seed32 = rd();
    const uint64_t seed64 = (static_cast<uint64_t>(rd()) << 32) | rd();

    std::mt19937    mt32(seed32);
    std::mt19937_64 mt64(seed64);

    std::cout << "============================================================\n";
    std::cout << "  Phase 1.8 | 03_random_bench.cpp\n";
    std::cout << "  RNG Benchmark for Monte Carlo Ising Simulations\n";
    std::cout << "============================================================\n\n";
    std::cout << "  N = " << N_BENCH / 1'000'000 << " million draws per benchmark\n";
    std::cout << "  Seed (mt32): 0x" << std::hex << seed32 << "\n";
    std::cout << "  Seed (mt64): 0x" << seed64    << std::dec << "\n\n";

    // =========================================================================
    // SECTION 1: Speed Benchmark
    // =========================================================================
    std::cout << "------------------------------------------------------------\n";
    std::cout << "  SECTION 1: Speed Comparison\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << "  " << std::left  << std::setw(38) << "Generator / Distribution"
              << std::right << std::setw(12) << "Time"
              << std::setw(12) << "MRPS" << "\n";
    std::cout << "  " << std::string(60, '-') << "\n";

    auto r_c    = bench_c_rand    (N_BENCH, seed32);
    auto r_mt32 = bench_mt19937_raw (N_BENCH, mt32);
    // Re-create mt32 to get a fresh state for the distribution benchmarks
    mt32.seed(seed32);
    auto r_uid  = bench_mt19937_uid (N_BENCH, mt32);
    mt32.seed(seed32);
    auto r_real = bench_mt19937_real(N_BENCH, mt32);
    auto r_mt64 = bench_mt19937_64  (N_BENCH, mt64);

    print_result(r_c);
    print_result(r_mt32);
    print_result(r_uid);
    print_result(r_real);
    print_result(r_mt64);

    std::cout << "\n  (Checksums — prevent dead-code elimination by optimizer)\n";
    std::cout << "  rand()   chk: " << std::hex << r_c.checksum    << "\n";
    std::cout << "  mt32 raw chk: " << r_mt32.checksum << "\n";
    std::cout << "  uid      chk: " << r_uid.checksum  << "\n";
    std::cout << "  real     chk: " << r_real.checksum << "\n";
    std::cout << "  mt64 raw chk: " << r_mt64.checksum << std::dec << "\n";

    // =========================================================================
    // SECTION 2: Statistical Quality — Chi-squared coin-flip test
    //   Tests whether the lowest bit of each generator is unbiased.
    //   This is the exact bit used by `rand() % 2` in naive code.
    // =========================================================================
    std::cout << "\n------------------------------------------------------------\n";
    std::cout << "  SECTION 2: Statistical Quality — Chi-squared Coin-Flip\n";
    std::cout << "  (Tests lowest bit quality over " << N_CHISQ / 1'000'000
              << " million draws)\n";
    std::cout << "------------------------------------------------------------\n";

    {
        // rand()
        std::srand(seed32);
        long heads = 0;
        for (long i = 0; i < N_CHISQ; ++i) heads += (std::rand() & 1);
        double z = (heads - N_CHISQ/2.0) / std::sqrt(N_CHISQ / 4.0);
        std::cout << "  " << std::left << std::setw(38) << "Legacy C rand() — bit 0"
                  << "heads=" << heads
                  << "  z-score=" << std::fixed << std::setprecision(3) << z
                  << (std::abs(z) < 3.0 ? "  [OK]" : "  [SUSPICIOUS — LCG low-bit weakness]")
                  << "\n";
    }
    {
        mt32.seed(seed32);
        std::uniform_int_distribution<uint32_t> raw32;
        chi2_coin("std::mt19937 — bit 0", N_CHISQ, mt32, raw32);
    }
    {
        mt64.seed(seed64);
        std::uniform_int_distribution<uint64_t> raw64;
        chi2_coin("std::mt19937_64 — bit 0", N_CHISQ, mt64, raw64);
    }

    // =========================================================================
    // SECTION 3: Implications for Ising v2.0 C++ Design
    // =========================================================================
    std::cout << "\n------------------------------------------------------------\n";
    std::cout << "  SECTION 3: Ising v2.0 Design Implications\n";
    std::cout << "------------------------------------------------------------\n";

    constexpr int    L        = 1000;
    constexpr long   N_sites  = static_cast<long>(L) * L;
    constexpr long   sweeps   = 100'000L;
    constexpr long   rng_need = N_sites * sweeps;

    // Period of mt19937 in scientific notation (approximation for display)
    // 2^19937 - 1 ≈ 10^(19937 * log10(2)) ≈ 10^6001
    double log10_period = 19937.0 * std::log10(2.0);

    std::cout << "\n  Simulation parameters:\n";
    std::cout << "    L                 = " << L        << "\n";
    std::cout << "    N sites           = " << N_sites  << "\n";
    std::cout << "    Sweeps            = " << sweeps   << "\n";
    std::cout << "    RNG draws needed  = " << rng_need << " = 10^"
              << std::fixed << std::setprecision(1)
              << std::log10(static_cast<double>(rng_need)) << "\n\n";

    std::cout << "  Generator period comparison:\n";
    std::cout << "    rand() period     ≈ 2^31   = 2.1 * 10^9  [EXHAUSTED in ~0.02s]\n";
    std::cout << "    mt19937 period    ≈ 2^19937 = 10^"
              << std::setprecision(0) << log10_period
              << "  [NEVER EXHAUSTED]\n\n";

    std::cout << "  Recommended RNG for Ising v2.0:\n";
    std::cout << "    Generator : std::mt19937_64\n";
    std::cout << "    Seed      : std::random_device (once, at startup)\n";
    std::cout << "    Site sel  : rng() % n_occ  (or uniform_int_distribution)\n";
    std::cout << "    Acceptance: uniform_real_distribution<double>(0.0, 1.0)\n";
    std::cout << "    Threading : One mt19937_64 instance per thread\n";
    std::cout << "                Seed each with splitmix64(global_seed + thread_id)\n\n";

    std::cout << "  WARNING — never do this:\n";
    std::cout << "    for (int sweep : sweeps)\n";
    std::cout << "      std::mt19937 rng(std::random_device{}());  // RE-SEEDING!\n";
    std::cout << "    Reason: mt19937 warm-up costs 624 state-array fills.\n";
    std::cout << "    Re-seeding 10^5 sweeps = 6.24*10^7 extra ops/run.\n";
    std::cout << "    Worse: same second → same seed → repeated sequence.\n";

    std::cout << "\n============================================================\n";
    return 0;
}
