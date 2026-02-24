#include "MetropolisEngine.hpp"

#include <cmath>
#include <stdexcept>

namespace ising::hp {

MetropolisEngine::MetropolisEngine(IsingLattice& lattice,
                                   double temperature,
                                   double coupling_J)
    : lattice_(lattice),
      temperature_(temperature),
      beta_(0.0),
      coupling_J_(coupling_J),
      site_dist_(0, lattice.site_count() - 1),
      uniform01_(0.0, 1.0) {
    if (lattice_.site_count() <= 0) {
        throw std::invalid_argument("MetropolisEngine: lattice site count must be positive");
    }
    if (temperature_ <= 0.0) {
        throw std::invalid_argument("MetropolisEngine: temperature must be > 0");
    }
    if (lattice_.coordination_number() != 4) {
        throw std::logic_error("MetropolisEngine: current LUT implementation expects z=4 (Square2D)");
    }

    beta_ = 1.0 / temperature_;
    rebuild_lut();
}

bool MetropolisEngine::step(std::mt19937_64& rng) noexcept {
    const int site = site_dist_(rng);
    const int i = lattice_.row_from_site(site);
    const int j = lattice_.col_from_site(site);

    const int spin = static_cast<int>(lattice_.get_spin(i, j));
    const int neighbor_sum = lattice_.get_neighbor_sum(i, j);
    const int lut_idx = lut_index_from_spin_sum(spin, neighbor_sum);

    if (boltzmann_lut_[lut_idx] >= uniform01_(rng)) {
        lattice_.flip_spin(i, j);
        return true;
    }
    return false;
}

long MetropolisEngine::sweep(std::mt19937_64& rng) noexcept {
    long accepted = 0;
    const int attempts = lattice_.site_count();
    for (int n = 0; n < attempts; ++n) {
        if (step(rng)) {
            ++accepted;
        }
    }
    return accepted;
}

void MetropolisEngine::set_temperature(double temperature) {
    if (temperature <= 0.0) {
        throw std::invalid_argument("MetropolisEngine: temperature must be > 0");
    }
    temperature_ = temperature;
    beta_ = 1.0 / temperature_;
    rebuild_lut();
}

void MetropolisEngine::rebuild_lut() noexcept {
    for (int k = 0; k < 5; ++k) {
        // dE_unit ∈ {-4, -2, 0, +2, +4} for square lattice
        const int dE_unit = (k - 2) * 2;
        const double delta_E = 2.0 * coupling_J_ * static_cast<double>(dE_unit);
        boltzmann_lut_[k] = (delta_E <= 0.0) ? 1.0 : std::exp(-beta_ * delta_E);
    }
}

int MetropolisEngine::lut_index_from_spin_sum(int spin, int neighbor_sum) const noexcept {
    // spin ∈ {-1,+1}, neighbor_sum ∈ {-4,-2,0,+2,+4}
    // dE_unit = spin * neighbor_sum ∈ {-4,-2,0,+2,+4}
    // index   = dE_unit/2 + 2 -> [0..4]
    return (spin * neighbor_sum) / 2 + 2;
}

} // namespace ising::hp
