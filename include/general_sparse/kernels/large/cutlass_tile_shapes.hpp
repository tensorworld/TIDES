#pragma once

namespace bsptc {

// Tile shapes for CUTLASS grouped GEMM dispatch.
// Each shape has per-type kernel instantiations in src/cutlass/gemm_*.cu.
enum class TileShape {
    Tile_128x128, // Square: min(M,N) >= 16 (float default)
    Tile_64x64,   // Square: legacy / double SM80
    Tile_32x32,   // Double SM80 fallback
    Tile_64x8,    // SkinnyTall: M > N, min(M,N) < 16 (fused epilogue transpose)
    Tile_8x64,    // SkinnyWide: N >= M, min(M,N) < 16 (direct RowMajor-C)
    Tile_128x16,  // IntermediateTall: M >> N, min(M,N) in [16,32) (fused epilogue transpose)
    Tile_16x128,  // IntermediateWide: N >> M, min(M,N) in [16,32) (direct RowMajor-C)
    Tile_256x32,  // IntermediateTall: M >> N, min(M,N) >= 32 (direct RowMajor-C)
    Tile_32x256,  // IntermediateWide: N >> M, min(M,N) >= 32 (direct RowMajor-C)
};

inline int tile_m(TileShape t) {
    switch (t) {
        case TileShape::Tile_128x128: return 128;
        case TileShape::Tile_64x64:   return 64;
        case TileShape::Tile_32x32:   return 32;
        case TileShape::Tile_64x8:    return 64;
        case TileShape::Tile_8x64:    return 8;
        case TileShape::Tile_128x16:  return 128;
        case TileShape::Tile_16x128:  return 16;
        case TileShape::Tile_256x32:  return 256;
        case TileShape::Tile_32x256:  return 32;
    }
    return 128;
}

inline int tile_n(TileShape t) {
    switch (t) {
        case TileShape::Tile_128x128: return 128;
        case TileShape::Tile_64x64:   return 64;
        case TileShape::Tile_32x32:   return 32;
        case TileShape::Tile_64x8:    return 8;
        case TileShape::Tile_8x64:    return 64;
        case TileShape::Tile_128x16:  return 16;
        case TileShape::Tile_16x128:  return 128;
        case TileShape::Tile_256x32:  return 32;
        case TileShape::Tile_32x256:  return 256;
    }
    return 128;
}

inline const char* tile_shape_name(TileShape t) {
    switch (t) {
        case TileShape::Tile_128x128: return "128x128";
        case TileShape::Tile_64x64:   return "64x64";
        case TileShape::Tile_32x32:   return "32x32";
        case TileShape::Tile_64x8:    return "64x8";
        case TileShape::Tile_8x64:    return "8x64";
        case TileShape::Tile_128x16:  return "128x16";
        case TileShape::Tile_16x128:  return "16x128";
        case TileShape::Tile_256x32:  return "256x32";
        case TileShape::Tile_32x256:  return "32x256";
    }
    return "?";
}

} // namespace bsptc
