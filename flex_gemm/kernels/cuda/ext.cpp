#include <torch/extension.h>
#include "hash/api.h"
#include "spconv/api.h"


using namespace flex_gemm;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Hash functions
    m.def("hashmap_insert_cuda", &hash::hashmap_insert_cuda);
    m.def("hashmap_lookup_cuda", &hash::hashmap_lookup_cuda);
    m.def("hashmap_insert_3d_cuda", &hash::hashmap_insert_3d_cuda);
    m.def("hashmap_lookup_3d_cuda", &hash::hashmap_lookup_3d_cuda);
    m.def("hashmap_insert_3d_idx_as_val_cuda", &hash::hashmap_insert_3d_idx_as_val_cuda);

    // Morton keys (meshy_sparse.morton3d_16-compatible)
    m.def("morton_keys_3d_16_cuda", &spconv::morton_keys_3d_16_cuda);
    m.def("morton_keys_batched_ncwhd_cuda", &spconv::morton_keys_batched_ncwhd_cuda);

    // Convolution functions
    m.def("hashmap_build_submanifold_conv_neighbour_map_cuda", &spconv::hashmap_build_submanifold_conv_neighbour_map_cuda);
    m.def("neighbor_map_post_process_for_masked_implicit_gemm_1", &spconv::neighbor_map_post_process_for_masked_implicit_gemm_1);
    m.def("neighbor_map_post_process_for_masked_implicit_gemm_2", &spconv::neighbor_map_post_process_for_masked_implicit_gemm_2);
}
