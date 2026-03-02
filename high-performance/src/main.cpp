#include "IsingLattice.hpp"
#include "MetropolisEngine.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>

namespace {

double magnetization_per_site(const ising::hp::IsingLattice& lattice) noexcept {
    long sum = 0;
    for (int site = 0; site < lattice.site_count(); ++site) {
        sum += static_cast<int>(lattice.get_spin_by_site(site));
    }
    return static_cast<double>(sum) / static_cast<double>(lattice.site_count());
}

} // namespace

int main() {
    using Clock = std::chrono::high_resolution_clock;

    constexpr int L = 200;
    constexpr int WARMUP_SWEEPS = 1'000;
    constexpr int PRODUCTION_SWEEPS = 10'000;
    constexpr double T_CRIT = 2.269185;

    ising::hp::IsingLattice lattice(L, ising::hp::IsingLattice::Topology::Square2D, +1);
    ising::hp::MetropolisEngine engine(lattice, T_CRIT, 1.0);

    std::random_device rd;
    std::mt19937_64 rng((static_cast<std::uint64_t>(rd()) << 32) | rd());

    std::cout << "============================================================\n";
    std::cout << "  Ising-Dynamics | Phase 2.0 Production Engine\n";
    std::cout << "============================================================\n";
    std::cout << "  Lattice size          : " << L << " x " << L << " = " << lattice.site_count() << " sites\n";
    std::cout << "  Critical temperature  : T_c = " << T_CRIT << "\n";
    std::cout << "  Warm-up sweeps        : " << WARMUP_SWEEPS << "\n";
    std::cout << "  Production sweeps     : " << PRODUCTION_SWEEPS << "\n\n";

    // Scientific protocol:
    // We separate thermalization (warm-up) from measurement to avoid bias from
    // the ordered initial condition (all spins +1). Observables measured before
    // equilibrium do not represent the stationary Boltzmann ensemble.
    for (int sweep = 0; sweep < WARMUP_SWEEPS; ++sweep) {
        engine.sweep(rng);
    }

    double mag_sum = 0.0;
    double mag_abs_sum = 0.0;

    auto t0 = Clock::now();
    for (int sweep = 0; sweep < PRODUCTION_SWEEPS; ++sweep) {
        engine.sweep(rng);
        const double m = magnetization_per_site(lattice);
        mag_sum += m;
        mag_abs_sum += std::abs(m);
    }
    const auto t1 = Clock::now();

    const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double total_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();

    const std::uint64_t total_flip_attempts =
        static_cast<std::uint64_t>(lattice.site_count()) * static_cast<std::uint64_t>(PRODUCTION_SWEEPS);
    const double ns_per_flip = total_ns / static_cast<double>(total_flip_attempts);
    const double gf_per_s =
        static_cast<double>(total_flip_attempts) /
        std::chrono::duration<double>(t1 - t0).count() / 1.0e9;

    const double avg_m = mag_sum / static_cast<double>(PRODUCTION_SWEEPS);
    const double avg_abs_m = mag_abs_sum / static_cast<double>(PRODUCTION_SWEEPS);

    std::cout << "-------------------- Performance Metrics -------------------\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Total production time : " << total_ms << " ms\n";
    std::cout << std::setprecision(2);
    std::cout << "  Nanoseconds / flip    : " << ns_per_flip << " ns/flip\n";
    std::cout << std::setprecision(4);
    std::cout << "  Throughput            : " << gf_per_s << " GF/s\n";

    std::cout << "------------------- Magnetization Check --------------------\n";
    std::cout << std::setprecision(6);
    std::cout << "  <m> (production avg)  : " << avg_m << "\n";
    std::cout << "  <|m|> (prod avg)      : " << avg_abs_m << "\n";
    std::cout << "============================================================\n";

    return 0;
}
