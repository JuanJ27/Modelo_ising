// =============================================================================
// thermo_sweep.cpp
// Phase 2.2: High-Stochastic Ensemble — Mean Observables + Error Bars
//
// UPGRADE FROM PHASE 2.1:
//   Phase 2.1 produced one Markov-chain time-average per T-point.
//   Phase 2.2 wraps that single-trial measurement inside an ensemble loop
//   of NUM_TRIALS independent realizations (each with a distinct RNG seed).
//   This yields the Standard Error of the Mean (SEM) for every observable,
//   enabling rigorous error bars on all published data points.
//
// ERROR TAXONOMY — two conceptually distinct sources of uncertainty:
//
//   1. STATISTICAL UNCERTAINTY  (reducible, stochastic)
//      Origin  : finite Markov-chain length and finite number of trials.
//      Scaling : SEM ∝ 1/√(NUM_TRIALS · PROD_SWEEPS).
//      Remedy  : increase NUM_TRIALS or PROD_SWEEPS.
//      Reported: as ±error columns in the CSV (m_err, chi_err, e_err, cv_err).
//
//   2. SYSTEMATIC ERROR  (irreducible at fixed L, deterministic)
//      Origin  : finite-size effects — the correlation-length ξ is bounded
//                by the box size L, so observables shift from their L→∞ limits.
//      Manifestations:
//        • T_c(L) - T_c(∞) ~ L^{-1/ν}    (critical-point shift)
//        • χ_max(L)        ~ L^{γ/ν}      (peak divergence cut off at L)
//        • peak width      ~ L^{-1/ν}     (broader than the thermodynamic limit)
//      Remedy  : Finite-Size Scaling (FSS) — repeat at multiple L values
//                and extrapolate using the 2D Ising universality exponents
//                ν = 1, γ = 7/4, η = 1/4 (Onsager/Kaufman exact).
//      NOT reported here; FSS is deferred to Phase 2.3.
//
// PHYSICS:
//   Hamiltonian : H = -J * Σ_<i,j> s_i·s_j    (J=1, H_ext=0)
//   Observables : ⟨|m|⟩, χ, ⟨e⟩, Cv
//   FDT (per trial, from time-average moments):
//     Cv   = N/T² * (⟨e²⟩ − ⟨e⟩²)
//     χ    = N/T  * (⟨m²⟩ − ⟨|m|⟩²)
//   SEM (across NUM_TRIALS trials):
//     SEM(X) = σ(X) / √NUM_TRIALS
//            where σ²(X) = (1/K) Σ_k X_k² − [(1/K) Σ_k X_k]²
//
// T_SCHEDULE:
//   ΔT = 0.05  for T ∈ [1.0, 2.1)  and T ∈ (2.4, 4.0]   (coarse)
//   ΔT = 0.01  for T ∈ [2.1, 2.4]                         (fine — captures singularities)
//
// COMPILATION:
//   g++ -std=c++17 -O3 -I high-performance/src
//       high-performance/src/IsingLattice.cpp
//       high-performance/src/MetropolisEngine.cpp
//       high-performance/src/thermo_sweep.cpp
//       -o thermo_sweep
// =============================================================================

#include "IsingLattice.hpp"
#include "MetropolisEngine.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Energy per site: E/N
//
//   For the square lattice (z=4), each bond s_i·s_j is counted twice
//   (once from each endpoint) when iterating over all sites with sum_nn.
//   The factor 1/2 restores exact bond counting.
//
//   E        = -J * Σ_{bonds} s_i·s_j
//            = -J/2 * Σ_site s_i · sum_nn(i)
//   e = E/N  = -J/(2N) * Σ_site s_i · sum_nn(i)
// ---------------------------------------------------------------------------
double energy_per_site(const ising::hp::IsingLattice& lat, double J = 1.0) noexcept {
    double raw = 0.0;
    for (int site = 0; site < lat.site_count(); ++site) {
        raw += static_cast<int>(lat.get_spin_by_site(site))
             * lat.get_neighbor_sum_by_site(site);
    }
    return -J * raw / (2.0 * static_cast<double>(lat.site_count()));
}

// ---------------------------------------------------------------------------
// Magnetization per site: M/N  ∈ [-1, +1]
// ---------------------------------------------------------------------------
double magnetization_per_site(const ising::hp::IsingLattice& lat) noexcept {
    long sum = 0;
    for (int site = 0; site < lat.site_count(); ++site)
        sum += static_cast<int>(lat.get_spin_by_site(site));
    return static_cast<double>(sum) / static_cast<double>(lat.site_count());
}

// ---------------------------------------------------------------------------
// Temperature schedule construction.
//
//   Merges two grids:
//     Coarse (ΔT = 0.05): [1.00, 2.05] ∪ [2.45, 4.00]
//     Fine   (ΔT = 0.01): [2.10, 2.40]
//
//   The merged result is strictly ascending with no duplicates.
//   Exact endpoint arithmetic avoids floating-point drift by using
//   integer step counts (T = T_start + k * ΔT).
// ---------------------------------------------------------------------------
std::vector<double> make_temperature_schedule() {
    std::vector<double> t;
    t.reserve(120);

    // Coarse pre-critical: 1.00 → 2.05
    for (int k = 0; ; ++k) {
        double T = 1.0 + k * 0.05;
        if (T > 2.09) break;
        t.push_back(T);
    }

    // Fine critical window: 2.10 → 2.40
    for (int k = 0; ; ++k) {
        double T = 2.10 + k * 0.01;
        if (T > 2.405) break;
        t.push_back(T);
    }

    // Coarse post-critical: 2.45 → 4.00
    for (int k = 0; ; ++k) {
        double T = 2.45 + k * 0.05;
        if (T > 4.005) break;
        t.push_back(T);
    }

    return t;
}

} // namespace

int main() {
    // =========================================================================
    // SIMULATION PARAMETERS
    //
    //   Validation run  : L=50,  NUM_TRIALS=5   (~2-3 min,  verify CSV format)
    //   Production run  : L=100, NUM_TRIALS=10  (~2 h,      publication data)
    // =========================================================================
    constexpr int    L             = 100;    // ← SET TO 100 FOR PRODUCTION
    constexpr int    N             = L * L;
    constexpr int    NUM_TRIALS    = 10;     // ← SET TO 10  FOR PRODUCTION
    constexpr int    WARMUP_SWEEPS = 2000;
    constexpr int    PROD_SWEEPS   = 10000;

    constexpr double T_CRIT  = 2.2691853;  // Onsager exact: 2/ln(1+√2)
    constexpr double INV_P   = 1.0 / static_cast<double>(PROD_SWEEPS);
    constexpr double INV_K   = 1.0 / static_cast<double>(NUM_TRIALS);

    const char* CSV_PATH = "high-performance/src/production_data.csv";

    // -------------------------------------------------------------------------
    // Header
    // -------------------------------------------------------------------------
    std::cout << "=========================================================\n";
    std::cout << "  Ising-Dynamics | Phase 2.2 High-Stochastic Ensemble\n";
    std::cout << "  2D Square Lattice — FDT Observables + SEM Error Bars\n";
    std::cout << "=========================================================\n";
    std::cout << "  L             = " << L << "  (N = " << N << " sites)\n";
    std::cout << "  Trials/T-pt   = " << NUM_TRIALS    << "  (independent RNG seeds)\n";
    std::cout << "  Warm-up       = " << WARMUP_SWEEPS << " sweeps  (discarded per trial)\n";
    std::cout << "  Production    = " << PROD_SWEEPS   << " sweeps  (measured per trial)\n";
    std::cout << "  T_crit(exact) = " << T_CRIT        << "\n";
    std::cout << "  Output        -> " << CSV_PATH << "\n";

    const auto temps = make_temperature_schedule();
    std::cout << "  T-points      = " << temps.size() << "\n";
    std::cout << "  Total sweeps  = "
              << static_cast<long long>(temps.size()) * NUM_TRIALS
                 * (WARMUP_SWEEPS + PROD_SWEEPS)
              << "\n\n";

    // -------------------------------------------------------------------------
    // Open CSV — Phase 2.2 schema includes SEM columns
    // -------------------------------------------------------------------------
    std::ofstream csv(CSV_PATH);
    if (!csv.is_open()) {
        std::cerr << "ERROR: cannot open " << CSV_PATH << "\n";
        return 1;
    }
    // Schema: mean ± SEM for every FDT observable.
    // m_avg  / m_err  : signed magnetisation per site
    // chi    / chi_err: magnetic susceptibility per site (FDT)
    // e_avg  / e_err  : energy per site
    // cv     / cv_err : specific heat per site (FDT)
    csv << "T,m_avg,m_err,chi,chi_err,e_avg,e_err,cv,cv_err\n";

    const auto wall_start = std::chrono::high_resolution_clock::now();

    // =========================================================================
    // MAIN TEMPERATURE LOOP
    // =========================================================================
    for (std::size_t ti = 0; ti < temps.size(); ++ti) {
        const double T = temps[ti];

        // -----------------------------------------------------------------
        // ENSEMBLE ACCUMULATORS  (first- and second-moment across trials)
        //
        //   For observable X ∈ {m_abs, chi, e, cv}:
        //     ens_X  accumulates Σ_k X_k
        //     ens_X2 accumulates Σ_k X_k²
        //   allowing exact computation of σ²(X) = ⟨X²⟩ − ⟨X⟩²
        //   and SEM = σ(X)/√K  with K = NUM_TRIALS.
        // -----------------------------------------------------------------
        double ens_m    = 0.0, ens_m2    = 0.0;  // signed ⟨m⟩
        double ens_absm = 0.0, ens_absm2 = 0.0;  // ⟨|m|⟩
        double ens_chi  = 0.0, ens_chi2  = 0.0;
        double ens_e    = 0.0, ens_e2    = 0.0;
        double ens_cv   = 0.0, ens_cv2   = 0.0;

        // =================================================================
        // TRIAL LOOP — NUM_TRIALS independent Markov chains at this T
        // =================================================================
        for (int trial = 0; trial < NUM_TRIALS; ++trial) {

            // -----------------------------------------------------------
            // Fresh ordered lattice for every trial.
            //
            //   STATISTICAL UNCERTAINTY perspective:
            //   Each trial draws an independent path through phase space.
            //   Starting from an ordered state provides a deterministic and
            //   reproducible reference point; the warm-up phase then drives
            //   the chain to the equilibrium distribution regardless of T.
            //
            //   SYSTEMATIC ERROR perspective:
            //   The ordered initial condition introduces a positive-magnetisation
            //   bias that persists for ~ξ² sweeps (ξ = correlation length).
            //   WARMUP_SWEEPS = 2000 is sufficient for L=100 everywhere except
            //   very close to T_c where ξ → L (critical slowing-down).
            //   This residual bias is a finite-size systematic, not reduced
            //   by increasing NUM_TRIALS; only a longer warm-up or a
            //   cluster algorithm (Wolff/Swendsen-Wang) cures it.
            // -----------------------------------------------------------
            ising::hp::IsingLattice    lattice(L, ising::hp::IsingLattice::Topology::Square2D, +1);
            ising::hp::MetropolisEngine engine(lattice, T, 1.0);

            // Unique 64-bit seed per trial via hardware entropy source.
            // Combining two 32-bit words avoids period truncation on
            // platforms where std::random_device returns 32-bit values.
            std::random_device rd;
            std::mt19937_64 rng(
                (static_cast<std::uint64_t>(rd()) << 32)
                | static_cast<std::uint64_t>(rd())
            );

            // -------------------------------------------------------
            // THERMALIZATION — equilibrate the Markov chain
            // -------------------------------------------------------
            for (int s = 0; s < WARMUP_SWEEPS; ++s)
                engine.sweep(rng);

            // -------------------------------------------------------
            // PRODUCTION — accumulate single-trial time-average moments
            // -------------------------------------------------------
            double sum_e    = 0.0, sum_e2   = 0.0;
            double sum_m    = 0.0, sum_absm = 0.0, sum_m2 = 0.0;

            for (int s = 0; s < PROD_SWEEPS; ++s) {
                engine.sweep(rng);

                const double e    = energy_per_site(lattice);
                const double m    = magnetization_per_site(lattice);
                const double absm = std::abs(m);

                sum_e    += e;
                sum_e2   += e * e;
                sum_m    += m;
                sum_absm += absm;
                sum_m2   += m * m;
            }

            // -------------------------------------------------------
            // Single-trial FDT observables
            //
            //   These are time-averages over PROD_SWEEPS snapshots.
            //   Fluctuation-Dissipation Theorem (canonical ensemble):
            //     Cv = N/T² · (⟨e²⟩_t − ⟨e⟩_t²)
            //     χ  = N/T  · (⟨m²⟩_t − ⟨|m|⟩_t²)
            // -------------------------------------------------------
            const double avg_e    = sum_e    * INV_P;
            const double avg_e2   = sum_e2   * INV_P;
            const double avg_m    = sum_m    * INV_P;
            const double avg_absm = sum_absm * INV_P;
            const double avg_m2   = sum_m2   * INV_P;

            const double trial_cv  = static_cast<double>(N) / (T * T)
                                     * (avg_e2  - avg_e    * avg_e);
            const double trial_chi = static_cast<double>(N) / T
                                     * (avg_m2  - avg_absm * avg_absm);

            // -------------------------------------------------------
            // Accumulate into ensemble first- and second-moments
            // -------------------------------------------------------
            ens_m    += avg_m;      ens_m2    += avg_m    * avg_m;
            ens_absm += avg_absm;   ens_absm2 += avg_absm * avg_absm;
            ens_chi  += trial_chi;  ens_chi2  += trial_chi * trial_chi;
            ens_e    += avg_e;      ens_e2    += avg_e    * avg_e;
            ens_cv   += trial_cv;   ens_cv2   += trial_cv * trial_cv;

        } // end trial loop

        // -----------------------------------------------------------------
        // ENSEMBLE MEAN
        // -----------------------------------------------------------------
        const double mean_m    = ens_m    * INV_K;
        const double mean_absm = ens_absm * INV_K;
        const double mean_chi  = ens_chi  * INV_K;
        const double mean_e    = ens_e    * INV_K;
        const double mean_cv   = ens_cv   * INV_K;

        // -----------------------------------------------------------------
        // STANDARD ERROR OF THE MEAN (SEM)
        //
        //   Variance of the sample:  σ²(X) = ⟨X²⟩_K − ⟨X⟩_K²
        //   SEM = σ(X) / √K
        //
        //   Physical meaning:
        //   SEM tells you how precisely the mean is known given K trials.
        //   SEM → 0 as K → ∞  (statistical uncertainty is reducible).
        //   SEM does NOT capture the finite-size bias — that is systematic.
        //
        //   Guard against rounding-induced negative variance with max(0,·).
        // -----------------------------------------------------------------
        auto sem = [&](double sum_x, double sum_x2) -> double {
            const double mean_x  = sum_x  * INV_K;
            const double mean_x2 = sum_x2 * INV_K;
            const double var     = std::max(0.0, mean_x2 - mean_x * mean_x);
            return std::sqrt(var * INV_K);   // σ / √K
        };

        const double sem_m    = sem(ens_m,   ens_m2);
        const double sem_chi  = sem(ens_chi, ens_chi2);
        const double sem_e    = sem(ens_e,   ens_e2);
        const double sem_cv   = sem(ens_cv,  ens_cv2);

        // Write CSV row — one row per T-point
        csv << std::scientific << std::setprecision(8)
            << T          << ","
            << mean_m     << "," << sem_m   << ","
            << mean_chi   << "," << sem_chi << ","
            << mean_e     << "," << sem_e   << ","
            << mean_cv    << "," << sem_cv  << "\n";
        csv.flush();  // guarantee data on disk before next T-point

        // Progress line
        const double elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - wall_start
        ).count();

        std::cout << "  [" << std::setw(3) << (ti + 1) << "/" << temps.size() << "]"
                  << "  T=" << std::fixed      << std::setprecision(3) << T
                  << "  <|m|>="  << std::setprecision(4) << mean_absm
                  << "  ±"       << std::setprecision(4) << sem(ens_absm, ens_absm2)
                  << "  chi="    << std::setprecision(2) << mean_chi
                  << "  ±"       << std::setprecision(2) << sem_chi
                  << "  t="      << std::setprecision(0) << elapsed << "s\n";
        std::cout.flush();
    }

    const double total_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - wall_start
    ).count();

    std::cout << "\n  Done. Total wall time : "
              << std::fixed << std::setprecision(1) << total_s << " s\n";
    std::cout << "  CSV written to       : " << CSV_PATH << "\n";
    std::cout << "=========================================================\n";

    return 0;
}
