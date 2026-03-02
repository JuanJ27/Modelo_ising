#pragma once

#include <array>
#include <cstdint>
#include <random>

#include "IsingLattice.hpp"

namespace ising::hp {

/**
 * @file MetropolisEngine.hpp
 * @brief Production Metropolis driver operating on an encapsulated IsingLattice.
 *
 * @details
 * This class is the Phase 2.0 engine layer for the production C++ architecture:
 * declaration/implementation split, strict encapsulation, and DOD-friendly hot paths.
 *
 * Core rules:
 * - Owns thermodynamic parameters (`beta`, `J`) and a Boltzmann LUT for square 2D.
 * - Consumes `IsingLattice` by reference (non-owning), preserving data locality.
 * - Requires RNG (`std::mt19937_64`) passed by reference into `step()` / `sweep()`
 *   to maintain one continuous Markov chain stream and avoid expensive state copies.
 */
class MetropolisEngine {
public:
    explicit MetropolisEngine(IsingLattice& lattice,
                              double temperature,
                              double coupling_J = 1.0);

    /**
     * @brief Perform one random single-site Metropolis attempt.
     * @return true if spin flip accepted.
     */
    bool step(std::mt19937_64& rng) noexcept;

    /**
     * @brief Perform one sweep (N attempts, where N = lattice site count).
     * @return Number of accepted flips in this sweep.
     */
    long sweep(std::mt19937_64& rng) noexcept;

    /**
     * @brief Update temperature and rebuild Boltzmann LUT.
     */
    void set_temperature(double temperature);

    [[nodiscard]] double temperature() const noexcept { return temperature_; }
    [[nodiscard]] double beta() const noexcept { return beta_; }
    [[nodiscard]] double coupling_J() const noexcept { return coupling_J_; }

private:
    IsingLattice& lattice_;

    double temperature_;
    double beta_;
    double coupling_J_;

    std::array<double, 5> boltzmann_lut_{};

    std::uniform_int_distribution<int> site_dist_;
    std::uniform_real_distribution<double> uniform01_;

    void rebuild_lut() noexcept;
    [[nodiscard]] int lut_index_from_spin_sum(int spin, int neighbor_sum) const noexcept;
};

} // namespace ising::hp
