#include <torch/extension.h>
#include <cuda_runtime.h>

#include "morton.h"

namespace flex_gemm {
namespace spconv {

namespace {

constexpr int kMortonBlock = 256;

/// Spread lower 16 bits across 48 bits (three interleaved streams). Matches Python
/// ``morton_key._part1by2_16`` / ``meshy_sparse.morton3d_16`` semantics.
__device__ __forceinline__ uint64_t part1by2_16_u64(uint32_t v) {
    uint64_t x = static_cast<uint64_t>(v & 0xFFFFu);
    x = (x | (x << 16)) & 0xFF0000FFull;
    x = (x | (x << 8)) & 0xF00F00F00Full;
    x = (x | (x << 4)) & 0xC30C30C30C3ull;
    x = (x | (x << 2)) & 0x1249249249249ull;
    return x;
}

/// Spatial 48-bit Morton from NCWHD voxel (W,H,D) with meshy column order:
/// x = D, y = H, z = W (same as ``morton_keys_batched_ncwhd`` torch impl).
__device__ __forceinline__ uint64_t spatial_morton_whd_int32(
    int32_t w,
    int32_t h,
    int32_t d,
    int32_t skip_k
) {
    const int64_t xv = static_cast<int64_t>(d) >> skip_k;
    const int64_t yv = static_cast<int64_t>(h) >> skip_k;
    const int64_t zv = static_cast<int64_t>(w) >> skip_k;
    const uint32_t xu = static_cast<uint32_t>(xv & 0xFFFFLL);
    const uint32_t yu = static_cast<uint32_t>(yv & 0xFFFFLL);
    const uint32_t zu = static_cast<uint32_t>(zv & 0xFFFFLL);
    return part1by2_16_u64(xu) | (part1by2_16_u64(yu) << 1) |
           (part1by2_16_u64(zu) << 2);
}

__global__ void morton_keys_3d_16_kernel(
    const int32_t* __restrict__ coords,
    int64_t* __restrict__ keys,
    int64_t L,
    int32_t skip_k
) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= L) {
        return;
    }
    const int32_t* row = coords + i * 3;
    const int32_t w = row[0];
    const int32_t h = row[1];
    const int32_t d = row[2];
    keys[i] = static_cast<int64_t>(spatial_morton_whd_int32(w, h, d, skip_k));
}

__global__ void morton_keys_batched_ncwhd_kernel(
    const int32_t* __restrict__ coords,
    int64_t* __restrict__ keys,
    int64_t L,
    int32_t skip_k
) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= L) {
        return;
    }
    const int32_t* row = coords + i * 4;
    const int32_t b = row[0];
    const int32_t w = row[1];
    const int32_t h = row[2];
    const int32_t d = row[3];
    const uint64_t spatial =
        spatial_morton_whd_int32(w, h, d, skip_k);
    const int64_t batch_part = static_cast<int64_t>(b) << 48;
    keys[i] = static_cast<int64_t>(spatial) + batch_part;
}

void check_coords_2d(const torch::Tensor& coords, int64_t ncols, const char* ctx) {
    TORCH_CHECK(coords.is_cuda(), ctx, ": coords must be CUDA");
    TORCH_CHECK(
        coords.dtype() == torch::kInt32,
        ctx,
        ": coords must be int32"
    );
    TORCH_CHECK(coords.dim() == 2, ctx, ": coords must be 2D");
    TORCH_CHECK(
        coords.size(1) == ncols,
        ctx,
        ": coords must be [L, ",
        ncols,
        "]"
    );
    TORCH_CHECK(coords.is_contiguous(), ctx, ": coords must be contiguous");
}

} // namespace

torch::Tensor morton_keys_3d_16_cuda(const torch::Tensor& coords, int64_t skip_k) {
    check_coords_2d(coords, 3, "morton_keys_3d_16_cuda");
    TORCH_CHECK(skip_k >= 0 && skip_k <= 15, "skip_k must be in [0, 15]");

    const int64_t L = coords.size(0);
    auto opts = coords.options().dtype(torch::kInt64);
    torch::Tensor keys = torch::empty({L}, opts);

    if (L == 0) {
        return keys;
    }

    const dim3 block(kMortonBlock);
    const dim3 grid(static_cast<unsigned>((L + kMortonBlock - 1) / kMortonBlock));
    morton_keys_3d_16_kernel<<<grid, block>>>(
        coords.data_ptr<int32_t>(),
        keys.data_ptr<int64_t>(),
        L,
        static_cast<int32_t>(skip_k)
    );
    return keys;
}

torch::Tensor morton_keys_batched_ncwhd_cuda(const torch::Tensor& coords, int64_t skip_k) {
    check_coords_2d(coords, 4, "morton_keys_batched_ncwhd_cuda");
    TORCH_CHECK(skip_k >= 0 && skip_k <= 15, "skip_k must be in [0, 15]");

    const int64_t L = coords.size(0);
    auto opts = coords.options().dtype(torch::kInt64);
    torch::Tensor keys = torch::empty({L}, opts);

    if (L == 0) {
        return keys;
    }

    const dim3 block(kMortonBlock);
    const dim3 grid(static_cast<unsigned>((L + kMortonBlock - 1) / kMortonBlock));
    morton_keys_batched_ncwhd_kernel<<<grid, block>>>(
        coords.data_ptr<int32_t>(),
        keys.data_ptr<int64_t>(),
        L,
        static_cast<int32_t>(skip_k)
    );
    return keys;
}

} // namespace spconv
} // namespace flex_gemm
