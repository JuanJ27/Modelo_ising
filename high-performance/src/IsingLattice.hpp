#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
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
                                                    int8_t initial_spin = +1);

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
    [[nodiscard]] int8_t get_spin(int i, int j) const noexcept;

    /**
     * @brief Flip spin at lattice coordinate (i,j): +1 <-> -1.
     */
    void flip_spin(int i, int j) noexcept;

    /**
     * @brief Optional setter for initialization/protocol use (+1 or -1 normalization).
     */
    void set_spin(int i, int j, int8_t value) noexcept;

    /**
     * @brief Sum nearest-neighbor spins around (i,j), using precomputed neighbor table.
     *
     * For Square2D this returns s_up + s_down + s_left + s_right.
     */
    [[nodiscard]] int get_neighbor_sum(int i, int j) const noexcept;

    /**
     * @brief Read spin by linear site index [0, N).
     */
    [[nodiscard]] int8_t get_spin_by_site(int site) const noexcept { return spins_[site]; }

    /**
     * @brief Flip spin by linear site index [0, N).
     */
    void flip_spin_by_site(int site) noexcept { spins_[site] = static_cast<int8_t>(-spins_[site]); }

    /**
     * @brief Sum nearest-neighbor spins by linear site index [0, N).
     */
    [[nodiscard]] int get_neighbor_sum_by_site(int site) const noexcept;

    /**
     * @brief Convert linear site index to row coordinate.
     */
    [[nodiscard]] int row_from_site(int site) const noexcept { return site / linear_size_; }

    /**
     * @brief Convert linear site index to column coordinate.
     */
    [[nodiscard]] int col_from_site(int site) const noexcept { return site % linear_size_; }

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

    void build_neighbor_table();
    void build_square_2d_neighbors();
};

} // namespace ising::hp
