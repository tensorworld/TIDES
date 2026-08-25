#pragma once

// Backend selection for the Large tier.
//
// The tier historically had exactly one backend (CUTLASS grouped GEMM) and the
// only free parameter was the threadblock tile, so the selector returned a bare
// TileShape.  cuBLAS grouped GEMM is a second backend whose tile is chosen
// internally by the library and is not visible (or selectable) here, so the
// selector's result widens from "which tile" to "which kernel": a backend plus,
// for CUTLASS, the tile within it.
//
// Everything downstream (prepare / fill / run) keys off KernelChoice.

#include "cutlass_tile_shapes.hpp"

namespace bsptc {

enum class LargeBackend {
    CUTLASS,  // our grouped GEMM instantiations; tile is explicit
    CUBLAS,   // cublas{S,D}gemmGroupedBatched; tile is library-internal
};

// A selected Large-tier kernel.  `tile` is meaningful only for CUTLASS; the
// cuBLAS backend ignores it (kept at the queue's fallback for diagnostics).
struct KernelChoice {
    LargeBackend backend = LargeBackend::CUTLASS;
    TileShape tile = TileShape::Tile_128x128;

    KernelChoice() = default;
    KernelChoice(LargeBackend b, TileShape t) : backend(b), tile(t) {}

    // Implicit construction from a bare tile keeps the CUTLASS-only call sites
    // (and the tile-calibration benchmark) unchanged.
    KernelChoice(TileShape t) : backend(LargeBackend::CUTLASS), tile(t) {}

    bool is_cublas() const { return backend == LargeBackend::CUBLAS; }
};

inline const char* kernel_choice_name(KernelChoice c) {
    return c.is_cublas() ? "cublas" : tile_shape_name(c.tile);
}

} // namespace bsptc
