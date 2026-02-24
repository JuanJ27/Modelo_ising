#include "IsingLattice.hpp"

namespace ising::hp {

IsingLattice::IsingLattice(int linear_size, Topology topology, int8_t initial_spin)
    : linear_size_(linear_size),
      site_count_(linear_size > 0 ? linear_size * linear_size : 0),
      topology_(topology),
      coordination_number_(coordination_for(topology)),
      spins_(site_count_, initial_spin >= 0 ? int8_t(+1) : int8_t(-1)),
      neighbors_(site_count_ * coordination_number_, 0) {
    if (linear_size_ <= 0) {
        throw std::invalid_argument("IsingLattice: linear_size must be positive");
    }
    build_neighbor_table();
}

int8_t IsingLattice::get_spin(int i, int j) const noexcept {
    return spins_[index_2d(i, j)];
}

void IsingLattice::flip_spin(int i, int j) noexcept {
    const int site = index_2d(i, j);
    spins_[site] = static_cast<int8_t>(-spins_[site]);
}

void IsingLattice::set_spin(int i, int j, int8_t value) noexcept {
    spins_[index_2d(i, j)] = (value >= 0) ? int8_t(+1) : int8_t(-1);
}

int IsingLattice::get_neighbor_sum(int i, int j) const noexcept {
    return get_neighbor_sum_by_site(index_2d(i, j));
}

int IsingLattice::get_neighbor_sum_by_site(int site) const noexcept {
    int sum = 0;
    const int base = site * coordination_number_;
    for (int k = 0; k < coordination_number_; ++k) {
        sum += static_cast<int>(spins_[neighbors_[base + k]]);
    }
    return sum;
}

void IsingLattice::build_neighbor_table() {
    switch (topology_) {
        case Topology::Square2D:
            build_square_2d_neighbors();
            return;
        case Topology::Chain1D:
            throw std::logic_error("IsingLattice: Chain1D neighbor builder not implemented yet");
        case Topology::BCC3D:
            throw std::logic_error("IsingLattice: BCC3D neighbor builder not implemented yet");
        default:
            throw std::logic_error("IsingLattice: unknown topology");
    }
}

void IsingLattice::build_square_2d_neighbors() {
    assert(coordination_number_ == 4);

    for (int i = 0; i < linear_size_; ++i) {
        for (int j = 0; j < linear_size_; ++j) {
            const int site = index_2d(i, j);
            const int base = site * coordination_number_;

            neighbors_[base + 0] = index_2d(i - 1, j);
            neighbors_[base + 1] = index_2d(i + 1, j);
            neighbors_[base + 2] = index_2d(i, j - 1);
            neighbors_[base + 3] = index_2d(i, j + 1);
        }
    }
}

} // namespace ising::hp
