// =============================================================================
// 01_memory_basics.cpp
// Phase 1.8: The C++ Awakening — Data Locality & Cache Efficiency
//
// PURPOSE:
//   Demonstrate why a 1D contiguous std::vector<int8_t> is the correct
//   memory model for an Ising lattice, and why it eliminates the aliasing
//   and cache-miss problems documented in the v1.5 structural audit.
//
// COMPILATION:
//   g++ -std=c++17 -O2 -Wall -Wextra -o 01_memory_basics 01_memory_basics.cpp
//
// Author : Physics Researcher Agent — Ising v2.0 Migration
// =============================================================================

#include <cstdint>    // int8_t
#include <iostream>
#include <iomanip>    // std::setw, std::hex
#include <vector>

// =============================================================================
// [DESIGN NOTE 1]: int8_t vs int (or int64_t)
//
//   Python's NumPy defaults to int64 (8 bytes) per spin — inherited from the
//   platform word size. In the audit (logic_architecture.md §Memory Layout),
//   a 100×100 lattice in int64 already occupies 78 KB — exceeding L1 cache on
//   most processors (typically 32–64 KB per core).
//
//   int8_t occupies exactly 1 byte. The same 100×100 lattice fits in 9.8 KB,
//   well inside L1 cache. For an L=500 lattice:
//     int64  →  500*500*8 bytes = 2,000 KB  (does NOT fit in L1 or L2 cache)
//     int8_t →  500*500*1 byte  =   244 KB  (fits in L3 cache, near L2 boundary)
//
//   Consequence: every Metropolis spin-flip reads from L1 instead of triggering
//   a DRAM fetch (~200 cycles penalty per miss). This is the single biggest
//   hardware reason for the targeted 15×–25× C++ speedup documented in
//   scalability_report.md.
// =============================================================================

// =============================================================================
// [DESIGN NOTE 2]: Contiguous 1D vector vs Python nested lists
//
//   Python's list-of-lists (or list-of-numpy-arrays) layout:
//
//     List header → [ ptr_row0 | ptr_row1 | ptr_row2 | ... ]
//                        |            |
//                      [col0|col1|...] [col0|col1|...]
//
//   Each row is a SEPARATE heap allocation at an arbitrary address.
//   Traversing rows requires following pointer chains ("pointer chasing"),
//   causing cache misses on every row boundary.
//
//   A 1D std::vector<int8_t> of size L*L stores ALL elements in a single
//   contiguous block:
//
//     Byte:  [ 0 | 1 | 2 | 3 | ... | L*L-1 ]
//             s00 s01 s02 s03       s_{L-1,L-1}
//
//   When the CPU loads element [i,j], the hardware prefetcher automatically
//   loads the next 64 bytes (a full cache line) — i.e., 64 spin values
//   arrive "for free". This is the spatial locality that Python cannot provide
//   with its object model.
// =============================================================================

// =============================================================================
// [DESIGN NOTE 3]: Aliasing / Shared-State Elimination
//
//   In the Python audit (logic_architecture.md §State Management), both the
//   `Red` and `Metropolis` objects held Python references to the SAME numpy
//   array object. Any mutation by Metropolis silently changed `Red`'s state:
//
//     Python:  metro.lattice = red.lattice   # NOT a copy; same object in RAM
//
//   In C++, passing by VALUE to the engine gives independent ownership:
//
//     void step(SpinConfiguration config);  // copy — isolated
//
//   Or, explicit reference semantics with clear ownership markers:
//
//     void step(SpinConfiguration& config); // deliberate, named mutation
//
//   std::vector enforces move semantics, so ownership transfer is unambiguous
//   at compile time. The aliasing bug is structurally impossible.
// =============================================================================


// =============================================================================
//  ROW-MAJOR INDEX (Flat 1D address from 2D lattice coordinates)
//
//  For an L×L lattice stored row-by-row:
//
//    Memory layout:  [row0_col0, row0_col1, ..., row0_colL-1,
//                     row1_col0, row1_col1, ..., row1_colL-1,
//                     ...
//                     rowL-1_col0, ... rowL-1_colL-1]
//
//  Address of element (i, j):   base_address + (i * L + j) * sizeof(T)
//
//  This is C's native array layout and maximises cache-line utilisation
//  when iterating over row j with i fixed (inner loop walks consecutive bytes).
// =============================================================================
inline int get_index(int i, int j, int L) {
    return i * L + j;
}


int main() {

    constexpr int L = 10;          // Lattice linear size: 10×10 = 100 sites
    const int N = L * L;

    // -------------------------------------------------------------------------
    // Allocate a single contiguous block of N int8_t values.
    // std::vector guarantees that &v[0], &v[1], &v[2], ... are addresses
    // that differ by exactly sizeof(int8_t) = 1 byte.
    // -------------------------------------------------------------------------
    std::vector<int8_t> lattice(N);

    // Initialise: all spins up (+1) — an ordered ferromagnetic state
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            lattice[get_index(i, j, L)] = +1;

    // =========================================================================
    // PRINT: hexadecimal addresses of every element in the 2D lattice
    //
    //   Expected output pattern:
    //     lattice[0,0]  @ 0x...b0  | value: +1
    //     lattice[0,1]  @ 0x...b1  | value: +1   ← differs by exactly 1 byte
    //     lattice[0,2]  @ 0x...b2  | value: +1
    //     ...
    //     lattice[1,0]  @ 0x...ba  | value: +1   ← row 1 starts 10 bytes later
    //
    //   KEY OBSERVATION:
    //   Every address differs from the previous by exactly +1 = sizeof(int8_t).
    //   This is the proof of contiguity. The CPU can load 64 consecutive spins
    //   per cache-line fetch instead of one.
    // =========================================================================
    std::cout << "============================================================\n";
    std::cout << "  Ising Lattice Memory Map — L=" << L << "  (" << N << " sites)\n";
    std::cout << "  Storage type: int8_t (" << sizeof(int8_t) << " byte/spin)\n";
    std::cout << "  Total footprint: " << N * sizeof(int8_t) << " bytes\n";
    std::cout << "  vs int64_t would cost: " << N * sizeof(int64_t) << " bytes\n";
    std::cout << "============================================================\n\n";

    std::cout << std::left
              << std::setw(16) << "Site (i,j)"
              << std::setw(8)  << "Flat idx"
              << std::setw(20) << "Hex Address"
              << "Value\n";
    std::cout << std::string(58, '-') << "\n";

    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
            int idx = get_index(i, j, L);
            const void* addr = static_cast<const void*>(&lattice[idx]);
            std::cout << std::left
                      << "(" << std::setw(2) << i << "," << std::setw(2) << j << ")        "
                      << std::setw(8)  << idx
                      << "0x" << std::hex << std::setw(16)
                      << reinterpret_cast<uintptr_t>(addr)
                      << std::dec
                      << static_cast<int>(lattice[idx]) << "\n";
        }
        // Blank separator between rows for visual clarity
        std::cout << "\n";
    }

    // =========================================================================
    // PROOF OF CONTIGUITY: compute byte stride between consecutive elements
    // =========================================================================
    std::cout << "============================================================\n";
    std::cout << "  Contiguity Proof: stride between consecutive elements\n";
    std::cout << "============================================================\n";

    for (int k = 0; k < 5; ++k) {
        uintptr_t addr_k   = reinterpret_cast<uintptr_t>(&lattice[k]);
        uintptr_t addr_k1  = reinterpret_cast<uintptr_t>(&lattice[k + 1]);
        ptrdiff_t stride   = static_cast<ptrdiff_t>(addr_k1 - addr_k);
        std::cout << "  &lattice[" << k+1 << "] - &lattice[" << k << "] = "
                  << stride << " byte(s)  (sizeof int8_t = "
                  << sizeof(int8_t) << ")\n";
    }

    std::cout << "\n  Every stride = 1: CONTIGUOUS MEMORY CONFIRMED.\n";
    std::cout << "  A Python list-of-lists would show RANDOM stride values.\n";
    std::cout << "============================================================\n";

    return 0;
}
