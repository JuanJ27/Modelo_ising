#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

/**
 * @file IsingLattice.hpp
 * @brief Production-grade Ising lattice container for the high-performance C++ engine.
 *
 * @details
 * This header is the modular, encapsulated evolution of the UdeA prototype
 * (`foundations/code/Proyecto_Ising_MonteCarlo.ipynb`) into a production-ready,
 * Data-Oriented Design (DOD) core.
 *
 * Design goals:
 * - Strong encapsulation: spin state is private (`std::vector<int8_t>`), preventing
 *   aliasing bugs observed in the v1.5 architecture audit.
 * - DOD-friendly memory layout: contiguous spin array + contiguous neighbor table.
 * - Hot-path API for Metropolis kernels:
 *   - `get_spin(i,j)`
 *   - `flip_spin(i,j)`
 *   - `get_neighbor_sum(i,j)`
 * - Modern C++ safety/performance:
 *   - const-correct accessors
 *   - `noexcept` on hot methods
 * - Extensibility:
 *   - Topology-agnostic core via precomputed neighbor table
 *   - New topologies (1D chain, 3D BCC, etc.) can be added by introducing new
 *     neighbor builders without rewriting the simulation kernel interface.
 */
namespace ising::hp {

class IsingLattice {
public:
    /**
     * @brief Supported topology tags for neighbor-table generation.
     */
    enum class Topology : std::uint8_t {
        Square2D = 0,
        Chain1D,
        BCC3D
    };

    /**
     * @brief Construct lattice with periodic boundaries and uniform initial spin.
     * @param linear_size Linear size L (for Square2D: N = L*L).
     * @param topology Topology selector (currently Square2D is implemented).
     * @param initial_spin Initial spin value; non-negative -> +1, negative -> -1.
     */
    explicit IsingLattice(int linear_size,
                          Topology topology = Topology::Square2D,
                          int8_t initial_spin = +1)
        : linear_size_(linear_size),
          site_count_(linear_size > 0 ? linear_size * linear_size : 0),
          topology_(topology),
          coordination_number_(coordination_for(topology)),
          spins_(site_count_, initial_spin >= 0 ? int8_t(+1) : int8_t(-1)),
          neighbors_(site_count_ * coordination_number_, 0) {
        assert(linear_size_ > 0);
        build_neighbor_table();
    }

    /** @brief Returns linear size L. */
    [[nodiscard]] int linear_size() const noexcept { return linear_size_; }

    /** @brief Returns total number of sites N. */
    [[nodiscard]] int site_count() const noexcept { return site_count_; }

    /** @brief Returns topology coordination number z. */
    [[nodiscard]] int coordination_number() const noexcept { return coordination_number_; }

    /** @brief Returns current topology tag. */
    [[nodiscard]] Topology topology() const noexcept { return topology_; }

    /**
     * @brief Read spin at lattice coordinate (i,j), applying periodic boundaries.
     */
    [[nodiscard]] int8_t get_spin(int i, int j) const noexcept {
        return spins_[index_2d(i, j)];
    }

    /**
     * @brief Flip spin at lattice coordinate (i,j): +1 <-> -1.
     */
    void flip_spin(int i, int j) noexcept {
        const int site = index_2d(i, j);
        spins_[site] = static_cast<int8_t>(-spins_[site]);
    }

    /**
     * @brief Optional setter for initialization/protocol use (+1 or -1 normalization).
     */
    void set_spin(int i, int j, int8_t value) noexcept {
        spins_[index_2d(i, j)] = (value >= 0) ? int8_t(+1) : int8_t(-1);
    }

    /**
     * @brief Sum nearest-neighbor spins around (i,j), using precomputed neighbor table.
     *
     * For Square2D this returns s_up + s_down + s_left + s_right.
     */
    [[nodiscard]] int get_neighbor_sum(int i, int j) const noexcept {
        const int site = index_2d(i, j);
        int sum = 0;
        const int base = site * coordination_number_;
        for (int k = 0; k < coordination_number_; ++k) {
            sum += static_cast<int>(spins_[neighbors_[base + k]]);
        }
        return sum;
    }

private:
    int linear_size_;
    int site_count_;
    Topology topology_;
    int coordination_number_;

    std::vector<int8_t> spins_;
    std::vector<int> neighbors_;

    [[nodiscard]] static int coordination_for(Topology topology) noexcept {
        switch (topology) {
            case Topology::Square2D: return 4;
            case Topology::Chain1D:  return 2;
            case Topology::BCC3D:    return 8;
            default:                 return 4;
        }
    }

    [[nodiscard]] int wrap(int x) const noexcept {
        const int m = x % linear_size_;
        return (m < 0) ? (m + linear_size_) : m;
    }

    [[nodiscard]] int index_2d(int i, int j) const noexcept {
        return wrap(i) * linear_size_ + wrap(j);
    }

    void build_neighbor_table() noexcept {
        switch (topology_) {
            case Topology::Square2D:
                build_square_2d_neighbors();
                return;
            case Topology::Chain1D:
            case Topology::BCC3D:
                // Placeholder until dedicated builders are added in Phase 2.x.
                // Keeps interface stable while signaling clear extension points.
                build_square_2d_neighbors();
                return;
            default:
                build_square_2d_neighbors();
                return;
        }
    }

    void build_square_2d_neighbors() noexcept {
        assert(coordination_number_ == 4);

        for (int i = 0; i < linear_size_; ++i) {
            for (int j = 0; j < linear_size_; ++j) {
                const int site = index_2d(i, j);
                const int base = site * coordination_number_;

                neighbors_[base + 0] = index_2d(i - 1, j); // up
                neighbors_[base + 1] = index_2d(i + 1, j); // down
                neighbors_[base + 2] = index_2d(i, j - 1); // left
                neighbors_[base + 3] = index_2d(i, j + 1); // right
            }
        }
    }
};

} // namespace ising::hp
