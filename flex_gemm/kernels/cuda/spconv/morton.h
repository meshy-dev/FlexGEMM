/*
 * Batched 48+16 Morton keys (meshy_sparse.morton3d_16-compatible).
 */

#pragma once
#include <torch/extension.h>

namespace flex_gemm {
namespace spconv {

/// [L,3] int32 columns (W, H, D) -> [L] int64 spatial Morton (48 bits used).
torch::Tensor morton_keys_3d_16_cuda(const torch::Tensor& coords, int64_t skip_k);

/// [L,4] int32 columns (batch, W, H, D) -> [L] int64 keys (batch in high bits).
torch::Tensor morton_keys_batched_ncwhd_cuda(const torch::Tensor& coords, int64_t skip_k);

} // namespace spconv
} // namespace flex_gemm
