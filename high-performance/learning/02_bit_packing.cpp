// =============================================================================
// 02_bit_packing.cpp
// Phase 1.8: The C++ Awakening — Bit-Packing for the Ising Lattice
//
// PURPOSE:
//   Demonstrate the theoretical maximum of Data Locality for spin storage.
//   In 01_memory_basics.cpp we used int8_t (1 byte per spin). Here we go 8x
//   further: we pack 8 spins into a single uint8_t byte using bitwise
//   operations, achieving a 64:1 compression ratio versus Python's int64
//   default.
//
// COMPILATION:
//   g++ -std=c++17 -O2 -Wall -Wextra -o 02_bit_packing 02_bit_packing.cpp
//
// Author : Physics Researcher Agent — Ising v2.0 Migration
// =============================================================================

#include <cstdint>
#include <iostream>
#include <iomanip>
#include <string>
#include <array>
#include <cassert>

// =============================================================================
// [DESIGN NOTE 1]: The Memory Compression Ratio
//
//   Python default (int64):    8 bytes  per spin
//   int8_t (01_memory_basics): 1 byte   per spin  →  8x improvement
//   uint8_t bit-packed:        1 byte   per 8 SPINS → 64x improvement
//
//   For a 2D lattice of size L=10000 (N=10^8 sites):
//
//     int64  storage : 10^8 * 8  =  800 MB   (does NOT fit in L3 cache)
//     int8_t storage : 10^8 * 1  =  100 MB   (fits in large L3, not L2)
//     bit-pack       : 10^8 / 8  =  12.5 MB  (fits in most L3 caches!)
//
//   At L=2000 (N=4*10^6):
//     int64  : 32 MB   (L3 miss territory)
//     int8_t :  4 MB   (likely L3)
//     bit-pack: 500 KB (fits in L2 on most modern CPUs)
//
//   This difference determines whether the Metropolis hot loop reads from
//   cache (≈4 cycles) or from DRAM (≈200 cycles). It is THE key factor
//   behind the targeted 15–25x speedup documented in scalability_report.md.
// =============================================================================

// =============================================================================
// [DESIGN NOTE 2]: Cache-Line Spin Density
//
//   A CPU cache line is universally 64 bytes wide.
//   The number of spins that arrive "for free" per cache-line fetch:
//
//     int64  : 64 bytes /  8 bytes  =    8 spins  per cache line
//     int8_t : 64 bytes /  1 byte   =   64 spins  per cache line
//     bit-pack: 64 bytes * 8 bits  = 512 spins  per cache line
//
//   During a Metropolis sweep, when we read the spin at site i to compute
//   ΔE, we also need the z=4 nearest-neighbor spins. With bit-packing and
//   row-major flat storage, all 4 neighbors of interior sites will already
//   be in the same or adjacent cache lines — making the neighbor fetch
//   effectively free.
//
//   This is called "Implicit Prefetching by the Hardware": the prefetcher
//   sees the sequential access pattern and loads the next cache line before
//   we ask for it. Bit-packing maximizes the density of useful physics data
//   per prefetch event.
// =============================================================================

// =============================================================================
// [DESIGN NOTE 3]: The Memory Wall
//
//   The "Memory Wall" (Wulf & McKee, 1995) refers to the growing gap between
//   CPU computation speed (~3 GHz, ~3*10^9 ops/sec) and memory bandwidth
//   (~50 GB/s for DDR5). For spin-flip workloads:
//
//     Compute bound : time ∝ (# float ops per flip)  — improved by SIMD/branchless
//     Memory bound  : time ∝ (# cache misses)        — improved by data compression
//
//   A single-spin Metropolis flip requires:
//     1. Read s_i
//     2. Read z neighbors
//     3. Compute ΔE = 2*s_i*(J*Σs_j + H)   (integer + 1 FP multiply)
//     4. Acceptance test
//     5. Conditional flip
//
//   Step 1–2 dominate cost when N > LLC size. Bit-packing directly attacks
//   the memory-bound regime: 8x more spins fit in cache per byte loaded,
//   so cache misses drop by ~8x. This is why scalability_report.md lists
//   "Implicit neighbor calculation + bit-packing" as mandatory for L > 1000.
// =============================================================================


// =============================================================================
//  BITWISE PRIMITIVE OPERATIONS
//  Documented as standalone functions for pedagogical clarity.
//  In production C++, these are inlined by the compiler at -O2.
// =============================================================================

/// SET bit k in byte storage to 1 (represents spin UP, mapped to +1)
/// Mechanism: OR with a mask that has only bit k set.
///   mask = 1 << k   e.g. k=3 → 00001000
///   OR sets bit k regardless of its current value.
inline void bit_set(uint8_t& storage, int k) {
    storage |= (uint8_t(1) << k);
}

/// CLEAR bit k in byte storage to 0 (represents spin DOWN, mapped to -1)
/// Mechanism: AND with the bitwise complement of the mask.
///   mask   = 1 << k   e.g. k=3 → 00001000
///   ~mask             →         11110111
///   AND clears bit k regardless of its current value.
inline void bit_clear(uint8_t& storage, int k) {
    storage &= ~(uint8_t(1) << k);
}

/// GET spin at bit k: returns +1 (up) if bit=1, -1 (down) if bit=0
/// Mechanism: shift right to bring bit k to position 0, then mask with 1.
///   (storage >> k) & 1  → 0 or 1
///   Map 0→-1, 1→+1 via the standard "2*bit - 1" trick.
inline int bit_get(uint8_t storage, int k) {
    int bit = (storage >> k) & 1;
    return 2 * bit - 1;          // {0,1} → {-1, +1}
}

/// FLIP spin at bit k: toggles 0→1 or 1→0 (the Metropolis update step)
/// Mechanism: XOR with a mask that has only bit k set.
///   XOR({0,1}, 1) = {1,0}  — exactly a bitwise NOT at position k.
///   This is the cheapest possible spin flip — a single XOR instruction.
inline void bit_flip(uint8_t& storage, int k) {
    storage ^= (uint8_t(1) << k);
}

/// Helper: pretty-print a byte as 8 binary digits (MSB left)
std::string byte_to_binary(uint8_t val) {
    std::string s(8, '0');
    for (int i = 7; i >= 0; --i)
        s[7 - i] = ((val >> i) & 1) ? '1' : '0';
    return s;
}

/// Helper: print spin value with sign
std::string spin_str(int s) {
    return s == +1 ? "+1 (UP)" : "-1 (DOWN)";
}


int main() {

    std::cout << "============================================================\n";
    std::cout << "  Phase 1.8 | 02_bit_packing.cpp\n";
    std::cout << "  Bit-Packing Primitives for Ising Spin Storage\n";
    std::cout << "============================================================\n\n";

    // =========================================================================
    // SECTION 1: Basic Bitwise Operations on a Single Byte
    // =========================================================================
    std::cout << "------------------------------------------------------------\n";
    std::cout << "  SECTION 1: Single-byte Bitwise Operations\n";
    std::cout << "------------------------------------------------------------\n";

    uint8_t storage = 0;
    std::cout << "  Initial byte    : " << byte_to_binary(storage)
              << "  (all spins DOWN = -1)\n\n";

    // SET: pack spin UP at bit positions 0, 2, 5, 7
    int set_positions[] = {0, 2, 5, 7};
    std::cout << "  [SET] Setting spin UP at bits: 0, 2, 5, 7\n";
    for (int k : set_positions) {
        bit_set(storage, k);
        std::cout << "    After set(bit=" << k << ")  : "
                  << byte_to_binary(storage) << "\n";
    }

    std::cout << "\n  Byte after all SETs: " << byte_to_binary(storage) << "\n";
    std::cout << "  (Bits 0,2,5,7 = 1; bits 1,3,4,6 = 0)\n\n";

    // GET: read each of the 8 spin values
    std::cout << "  [GET] Reading all 8 spins from the packed byte:\n";
    for (int k = 0; k < 8; ++k) {
        int spin = bit_get(storage, k);
        std::cout << "    bit[" << k << "] = " << ((storage >> k) & 1)
                  << "  →  spin = " << spin
                  << "  (" << (spin == 1 ? "UP  " : "DOWN") << ")\n";
    }

    // FLIP: toggle bits 2 and 5 (simulating a Metropolis acceptance)
    std::cout << "\n  [FLIP] Metropolis acceptance: flipping spins at bits 2 and 5\n";
    std::cout << "  Before flip : " << byte_to_binary(storage) << "\n";
    bit_flip(storage, 2);
    bit_flip(storage, 5);
    std::cout << "  After  flip : " << byte_to_binary(storage) << "\n";
    std::cout << "    bit[2] is now: " << spin_str(bit_get(storage, 2)) << "\n";
    std::cout << "    bit[5] is now: " << spin_str(bit_get(storage, 5)) << "\n";

    // =========================================================================
    // SECTION 2: Pack / Unpack Round-Trip Verification
    //
    // Demonstrates the fundamental invariant:
    //   unpack(pack(spins)) == spins  for all spin configurations
    //
    // This is the correctness test we must run before ANY C++ migration.
    // =========================================================================
    std::cout << "\n------------------------------------------------------------\n";
    std::cout << "  SECTION 2: Pack → Unpack Round-Trip Verification\n";
    std::cout << "------------------------------------------------------------\n";

    // Define 8 "logical" spins (+1 or -1)
    // Convention: +1 stored as bit=1, -1 stored as bit=0
    std::array<int, 8> original_spins = {+1, -1, +1, +1, -1, -1, +1, -1};

    std::cout << "\n  Original spin array (index 0..7):\n  [";
    for (int i = 0; i < 8; ++i)
        std::cout << std::setw(3) << original_spins[i] << (i < 7 ? "," : "");
    std::cout << " ]\n\n";

    // --- PACK: convert array of ±1 spins into a single byte ---
    uint8_t packed = 0;
    for (int k = 0; k < 8; ++k) {
        if (original_spins[k] == +1)
            bit_set(packed, k);
        // bit_clear not needed: packed initialised to 0 (all DOWN)
    }

    std::cout << "  Packed byte     : " << byte_to_binary(packed)
              << "  (hex: 0x" << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(packed) << std::dec << ")\n\n";

    // --- UNPACK: reconstruct array of ±1 from the byte ---
    std::array<int, 8> recovered_spins;
    for (int k = 0; k < 8; ++k)
        recovered_spins[k] = bit_get(packed, k);

    std::cout << "  Recovered spins (index 0..7):\n  [";
    for (int i = 0; i < 8; ++i)
        std::cout << std::setw(3) << recovered_spins[i] << (i < 7 ? "," : "");
    std::cout << " ]\n\n";

    // --- VERIFY: assert round-trip fidelity ---
    bool all_match = true;
    std::cout << "  Round-trip verification:\n";
    for (int k = 0; k < 8; ++k) {
        bool match = (original_spins[k] == recovered_spins[k]);
        if (!match) all_match = false;
        std::cout << "    spin[" << k << "]: original=" << std::setw(2) << original_spins[k]
                  << "  recovered=" << std::setw(2) << recovered_spins[k]
                  << "  " << (match ? "OK" : "MISMATCH!") << "\n";
    }
    std::cout << "\n  Result: " << (all_match ? "ALL OK — bit-packing is lossless." :
                                               "FAILURE — check encoding.") << "\n";
    assert(all_match && "Pack/unpack round-trip failed!");

    // =========================================================================
    // SECTION 3: Memory & Cache-Line Summary
    // =========================================================================
    std::cout << "\n------------------------------------------------------------\n";
    std::cout << "  SECTION 3: Memory & Cache-Line Analysis\n";
    std::cout << "------------------------------------------------------------\n";

    constexpr int L = 1000;
    constexpr long N = static_cast<long>(L) * L;   // 10^6 spins

    long mem_int64    = N * 8;
    long mem_int8     = N * 1;
    long mem_bitpack  = N / 8;

    int spins_per_cacheline_int64   = 64 / 8;
    int spins_per_cacheline_int8    = 64 / 1;
    int spins_per_cacheline_bitpack = 64 * 8;

    std::cout << "\n  Lattice: L=" << L << "  N=" << N << " spins\n\n";
    std::cout << "  Storage Type  | Bytes       | Spins / 64B Cache Line\n";
    std::cout << "  " << std::string(55, '-') << "\n";
    std::cout << "  int64 (Python)| " << std::setw(10) << mem_int64
              << " B | " << spins_per_cacheline_int64 << "  spins\n";
    std::cout << "  int8_t (v1.0) | " << std::setw(10) << mem_int8
              << " B | " << spins_per_cacheline_int8 << "  spins\n";
    std::cout << "  bit-packed    | " << std::setw(10) << mem_bitpack
              << " B | " << spins_per_cacheline_bitpack << " spins\n\n";

    std::cout << "  Compression vs Python int64:\n";
    std::cout << "    int8_t   → " << mem_int64 / mem_int8 << "x reduction\n";
    std::cout << "    bit-pack → " << mem_int64 / mem_bitpack << "x reduction\n\n";

    std::cout << "  Typical cache sizes (reference):\n";
    std::cout << "    L1: ~32 KB  → bit-packed L=16   fits (" << 16*16/8 << " B)\n";
    std::cout << "    L2: ~512 KB → bit-packed L=64   fits (" << 64*64/8 << " B = 512 B)\n";
    std::cout << "    L3: ~8  MB  → bit-packed L=8192 fits (" << 8192*8192/8/1024 << " KB)\n";

    std::cout << "\n  CONCLUSION:\n";
    std::cout << "    With bit-packing, a 1000x1000 lattice (125 KB) fits\n";
    std::cout << "    entirely in L2 cache on most modern CPUs. Every spin\n";
    std::cout << "    read in the Metropolis loop is served from cache,\n";
    std::cout << "    eliminating the Memory Wall for production-scale sims.\n";

    std::cout << "\n============================================================\n";
    return 0;
}
