// =============================================================================
// thermo_sweep.cpp
// Phase 2.1: Thermodynamic Validation — Data Generation Pipeline
//
// PURPOSE:
//   Drive the modular C++ engine through a temperature sweep and measure
//   thermodynamic observables using the Fluctuation-Dissipation Theorem (FDT).
//   Outputs a production-ready CSV for downstream phase-transition analysis
//   and comparison with the Onsager exact solution.
//
// PHYSICS:
//   Hamiltonian : H = -J * Σ_<i,j> s_i·s_j    (J=1, H_ext=0)
//   Observables : ⟨|m|⟩, χ, ⟨e⟩, Cv
//   FDT:
//     Cv   = N/T² * (⟨e²⟩ − ⟨e⟩²)      [intensive: divides by N internally]
//     χ    = N/T  * (⟨m²⟩ − ⟨|m|⟩²)
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
    constexpr int   L             = 100;
    constexpr int   N             = L * L;       // 10,000 sites
    constexpr int   WARMUP_SWEEPS = 2000;
    constexpr int   PROD_SWEEPS   = 10000;

    // T_CRIT (Onsager exact): 2/ln(1+√2)
    constexpr double T_CRIT = 2.2691853;

    const char* CSV_PATH = "high-performance/src/production_data.csv";

    // -------------------------------------------------------------------------
    // Header
    // -------------------------------------------------------------------------
    std::cout << "=======================================================\n";
    std::cout << "  Ising-Dynamics | Phase 2.1 Thermodynamic Sweep\n";
    std::cout << "  2D Square Lattice — Fluctuation-Dissipation Pipeline\n";
    std::cout << "=======================================================\n";
    std::cout << "  L             = " << L << "  (N = " << N << " sites)\n";
    std::cout << "  Warm-up       = " << WARMUP_SWEEPS << " sweeps  (discarded)\n";
    std::cout << "  Production    = " << PROD_SWEEPS   << " sweeps  (measured)\n";
    std::cout << "  T_crit(exact) = " << T_CRIT        << "\n";
    std::cout << "  Output        -> " << CSV_PATH << "\n";

    const auto temps = make_temperature_schedule();
    std::cout << "  T-points      = " << temps.size() << "\n\n";

    // -------------------------------------------------------------------------
    // Open CSV
    // -------------------------------------------------------------------------
    std::ofstream csv(CSV_PATH);
    if (!csv.is_open()) {
        std::cerr << "ERROR: cannot open " << CSV_PATH << "\n";
        return 1;
    }
    csv << "T,m_avg,m_abs_avg,chi,e_avg,cv\n";

    const auto wall_start = std::chrono::high_resolution_clock::now();

    // -------------------------------------------------------------------------
    // Main temperature loop
    // -------------------------------------------------------------------------
    for (std::size_t ti = 0; ti < temps.size(); ++ti) {
        const double T  = temps[ti];

        // Fresh ordered lattice at each T.
        // Starting from the all-+1 ferromagnetic state is intentional:
        //   T < T_c : system stays ordered → warm-up confines it to one domain.
        //   T > T_c : warm-up disorders the lattice into the paramagnetic phase.
        // This gives faster equilibration than a random start at low T.
        ising::hp::IsingLattice   lattice(L, ising::hp::IsingLattice::Topology::Square2D, +1);
        ising::hp::MetropolisEngine engine(lattice, T, 1.0);

        // Seed independently per T to remove inter-temperature RNG correlations.
        std::random_device rd;
        std::mt19937_64 rng(
            (static_cast<std::uint64_t>(rd()) << 32) | static_cast<std::uint64_t>(rd())
        );

        // -----------------------------------------------------------------
        // THERMALIZATION (warm-up) — data NOT recorded
        //
        //   Scientific necessity: the Markov chain must reach stationarity
        //   before observables can represent the equilibrium ensemble.
        //   Sweeps during warm-up are systematically biased by the initial
        //   condition (all spins +1) and cannot enter any ensemble average
        //   without violating the ergodic hypothesis.
        // -----------------------------------------------------------------
        for (int s = 0; s < WARMUP_SWEEPS; ++s)
            engine.sweep(rng);

        // -----------------------------------------------------------------
        // PRODUCTION MEASUREMENT — accumulate moments
        // -----------------------------------------------------------------
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

        // -----------------------------------------------------------------
        // Ensemble averages
        // -----------------------------------------------------------------
        const double inv_P  = 1.0 / static_cast<double>(PROD_SWEEPS);
        const double avg_e    = sum_e    * inv_P;
        const double avg_e2   = sum_e2   * inv_P;
        const double avg_m    = sum_m    * inv_P;
        const double avg_absm = sum_absm * inv_P;
        const double avg_m2   = sum_m2   * inv_P;

        // -----------------------------------------------------------------
        // Fluctuation-Dissipation Theorem
        //
        //   Specific heat (per site):
        //     Cv = N/T² * (⟨e²⟩ − ⟨e⟩²)
        //   Magnetic susceptibility (per site):
        //     χ  = N/T  * (⟨m²⟩ − ⟨|m|⟩²)
        //
        //   Note: using ⟨|m|⟩ rather than ⟨m⟩ for χ removes the
        //   cancellation from spin-flip domain symmetry at T near T_c.
        // -----------------------------------------------------------------
        const double cv  = static_cast<double>(N) / (T * T) * (avg_e2  - avg_e    * avg_e);
        const double chi = static_cast<double>(N) / T       * (avg_m2  - avg_absm * avg_absm);

        // Write row
        csv << std::scientific << std::setprecision(8)
            << T       << ","
            << avg_m   << ","
            << avg_absm << ","
            << chi     << ","
            << avg_e   << ","
            << cv      << "\n";
        csv.flush();   // guarantee data on disk after each T-point

        // Progress to stdout
        const double elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - wall_start
        ).count();

        std::cout << "  [" << std::setw(3) << (ti + 1) << "/" << temps.size() << "]"
                  << "  T=" << std::fixed << std::setprecision(3) << T
                  << "  <|m|>="  << std::setprecision(4) << avg_absm
                  << "  chi="    << std::setprecision(2) << chi
                  << "  cv="     << std::setprecision(2) << cv
                  << "  t="      << std::setprecision(0) << elapsed << "s\n";
        std::cout.flush();
    }

    const double total_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - wall_start
    ).count();

    std::cout << "\n  Done. Total wall time : "
              << std::fixed << std::setprecision(1) << total_s << " s\n";
    std::cout << "  CSV written to       : " << CSV_PATH << "\n";
    std::cout << "=======================================================\n";

    return 0;
}
