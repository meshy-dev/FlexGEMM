import dataclasses
import torch
from torch import Tensor
from torch.autograd import Function
from . import Algorithm
from .neighbor_map_searchsorted import build_submanifold_neighbor_map_searchsorted
from .. import spconv, utils
from ... import kernels


# All possible B1 (block size for N dimension) values across autotune configs.
_ALL_BLOCK_SIZES = [32, 64, 128, 256]


@dataclasses.dataclass
class SpConvConfig:
    """Frozen, compile-friendly config for sparse convolution.

    All fields are tensors, lists of tensors, or plain ints/strings --
    no Python objects or callbacks.  Obtain one via
    :meth:`SubMConv3dNeighborCache.freeze`.
    """

    neighbor_map: Tensor                   # [N, V] uint32
    sorted_idx: Tensor                     # [N] int64
    valid_kernels: list[Tensor]            # len = len(block_sizes), per-B1 valid kernel indices
    valid_kernel_segs: list[Tensor]        # len = len(block_sizes), per-B1 valid kernel segments
    block_sizes: list[int]                 # e.g. [32, 64, 128, 256]
    valid_signal_i: Tensor                 # [M] uint32
    valid_signal_o: Tensor                 # [M] uint32
    valid_signal_seg: Tensor               # [V+1] uint32
    splitk: int = 1                        # autotuned SPLITK (1 = non-splitk path)
    algorithm: str = "masked_implicit_gemm_splitk"


class SubMConv3dNeighborCache:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
        
    def compute_kernel_idx(self, block_size: int):
        valid_kernel, valid_kernel_seg = torch.ops.flex_gemm.neighbor_map_post_process_for_masked_implicit_gemm_2(self['gray_code'], self['sorted_idx'], block_size)
        self[f'valid_kernel_{block_size}'] = valid_kernel
        self[f'valid_kernel_seg_{block_size}'] = valid_kernel_seg
        
    def valid_kernel_callback(self, block_size: int) -> torch.Tensor:
        if not hasattr(self, f'valid_kernel_{block_size}'):
            self.compute_kernel_idx(block_size)
        return self[f'valid_kernel_{block_size}']
    
    def valid_kernel_seg_callback(self, block_size: int) -> torch.Tensor:
        if not hasattr(self, f'valid_kernel_seg_{block_size}'):
            self.compute_kernel_idx(block_size)
        return self[f'valid_kernel_seg_{block_size}']

    def freeze(self, splitk: int = 1, algorithm: str = "masked_implicit_gemm_splitk") -> SpConvConfig:
        """Freeze this cache into a :class:`SpConvConfig` for ``torch.compile``.

        Pre-computes ``valid_kernel`` / ``valid_kernel_seg`` for every
        possible autotune block size so the compiled path never needs
        a callback.

        Args:
            splitk: The SPLITK value to use.  Run one eager forward pass
                first to let the ``@autotune`` decorator choose, then pass
                the selected value here.  ``1`` means non-split-K.
            algorithm: ``"masked_implicit_gemm"`` or
                ``"masked_implicit_gemm_splitk"`` (default).

        Returns:
            A :class:`SpConvConfig` containing only tensors and plain ints.
        """
        assert hasattr(self, 'neighbor_map'), "neighbor_map is required"
        assert hasattr(self, 'sorted_idx'), "sorted_idx is required (masked algorithm)"

        block_sizes = _ALL_BLOCK_SIZES
        valid_kernels: list[Tensor] = []
        valid_kernel_segs: list[Tensor] = []
        for bs in block_sizes:
            self.compute_kernel_idx(bs)
            valid_kernels.append(self[f'valid_kernel_{bs}'])
            valid_kernel_segs.append(self[f'valid_kernel_seg_{bs}'])

        return SpConvConfig(
            neighbor_map=self['neighbor_map'],
            sorted_idx=self['sorted_idx'],
            valid_kernels=valid_kernels,
            valid_kernel_segs=valid_kernel_segs,
            block_sizes=block_sizes,
            valid_signal_i=self['valid_signal_i'],
            valid_signal_o=self['valid_signal_o'],
            valid_signal_seg=self['valid_signal_seg'],
            splitk=splitk,
            algorithm=algorithm,
        )


class SubMConv3dFunction(Function):
    @staticmethod
    def _compute_neighbor_cache(
        coords: torch.Tensor,
        shape: torch.Size,
        kernel_size: tuple[int, int, int],
        dilation: tuple[int, int, int],
    ) -> SubMConv3dNeighborCache:
        assert coords.is_contiguous(), "Coords should be contiguous"
        assert coords.dtype in [torch.int32], "Unsupported coords dtype. Expect int32"
        N, C, W, H, D = shape

        if spconv.SPATIAL_INDEX_MODE == "searchsorted":
            neighbor_map = build_submanifold_neighbor_map_searchsorted(
                coords,
                shape,
                kernel_size,
                dilation,
                skip_k=spconv.SEARCHSORTED_SKIP_K,
            )
            if spconv.ALGORITHM in [
                Algorithm.EXPLICIT_GEMM,
                Algorithm.IMPLICIT_GEMM,
                Algorithm.IMPLICIT_GEMM_SPLITK,
            ]:
                return SubMConv3dNeighborCache(**{"neighbor_map": neighbor_map})

            if spconv.ALGORITHM in [
                Algorithm.MASKED_IMPLICIT_GEMM,
                Algorithm.MASKED_IMPLICIT_GEMM_SPLITK,
            ]:
                if not neighbor_map.is_cuda:
                    raise NotImplementedError(
                        "masked implicit GEMM neighbor post-process is CUDA-only; "
                        "use CUDA coords, explicit/implicit algorithms on CPU, or hashmap mode."
                    )
                V = kernel_size[0] * kernel_size[1] * kernel_size[2]
                assert V <= 32, (
                    "Currently, the max kernel volume is 32 because kernel mask is encoded as uint32"
                )
                (
                    gray_code,
                    sorted_idx,
                    valid_signal_i,
                    valid_signal_o,
                    valid_signal_seg,
                ) = torch.ops.flex_gemm.neighbor_map_post_process_for_masked_implicit_gemm_1(
                    neighbor_map
                )
                return SubMConv3dNeighborCache(
                    **{
                        "neighbor_map": neighbor_map,
                        "gray_code": gray_code,
                        "sorted_idx": sorted_idx,
                        "valid_signal_seg": valid_signal_seg,
                        "valid_signal_i": valid_signal_i,
                        "valid_signal_o": valid_signal_o,
                    }
                )

            raise ValueError(f"Unsupported algorithm {spconv.ALGORITHM}")

        if not coords.is_cuda:
            raise NotImplementedError(
                "CUDA coords are required for hashmap neighbor-map build; "
                "use spconv.set_spatial_index_mode('searchsorted') for CPU or JIT-friendly builds."
            )

        hashmap_keys, hashmap_vals = utils.init_hashmap(
            shape, int(spconv.HASHMAP_RATIO * coords.shape[0]), coords.device
        )

        if spconv.ALGORITHM in [
            Algorithm.EXPLICIT_GEMM,
            Algorithm.IMPLICIT_GEMM,
            Algorithm.IMPLICIT_GEMM_SPLITK,
        ]:
            neighbor_map = torch.ops.flex_gemm.hashmap_build_submanifold_conv_neighbour_map_cuda(
                hashmap_keys,
                hashmap_vals,
                coords,
                W,
                H,
                D,
                kernel_size[0],
                kernel_size[1],
                kernel_size[2],
                dilation[0],
                dilation[1],
                dilation[2],
            )
            return SubMConv3dNeighborCache(**{"neighbor_map": neighbor_map})

        if spconv.ALGORITHM in [
            Algorithm.MASKED_IMPLICIT_GEMM,
            Algorithm.MASKED_IMPLICIT_GEMM_SPLITK,
        ]:
            neighbor_map = torch.ops.flex_gemm.hashmap_build_submanifold_conv_neighbour_map_cuda(
                hashmap_keys,
                hashmap_vals,
                coords,
                W,
                H,
                D,
                kernel_size[0],
                kernel_size[1],
                kernel_size[2],
                dilation[0],
                dilation[1],
                dilation[2],
            )
            V = kernel_size[0] * kernel_size[1] * kernel_size[2]
            assert V <= 32, (
                "Currently, the max kernel volume is 32 because kernel mask is encoded as uint32"
            )

            gray_code, sorted_idx, valid_signal_i, valid_signal_o, valid_signal_seg = (
                torch.ops.flex_gemm.neighbor_map_post_process_for_masked_implicit_gemm_1(
                    neighbor_map
                )
            )

            return SubMConv3dNeighborCache(
                **{
                    "neighbor_map": neighbor_map,
                    "gray_code": gray_code,
                    "sorted_idx": sorted_idx,
                    "valid_signal_seg": valid_signal_seg,
                    "valid_signal_i": valid_signal_i,
                    "valid_signal_o": valid_signal_o,
                }
            )

        raise ValueError(f"Unsupported algorithm {spconv.ALGORITHM}")

    @staticmethod
    def _compute_neighbor_cache_torch(
        coords: torch.Tensor,
        shape: torch.Size,
        kernel_size: tuple[int, int, int],
        dilation: tuple[int, int, int],
    ) -> SubMConv3dNeighborCache:
        assert spconv.ALGORITHM == Algorithm.EXPLICIT_GEMM, (
            "Only explicit_gemm is supported for torch implementation"
        )
        neighbor_map = build_submanifold_neighbor_map_searchsorted(
            coords,
            shape,
            kernel_size,
            dilation,
            skip_k=spconv.SEARCHSORTED_SKIP_K,
        )
        return SubMConv3dNeighborCache(**{"neighbor_map": neighbor_map})

    @staticmethod
    def _sparse_submanifold_conv_forward(
        feats: torch.Tensor,
        neighbor_cache: SubMConv3dNeighborCache,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert feats.is_contiguous(), "Input features should be contiguous"
        N = feats.shape[0]
        Co, Kw, Kh, Kd, Ci = weight.shape
        V = Kd * Kh * Kw
        
        if spconv.ALGORITHM == Algorithm.EXPLICIT_GEMM:        
            neighbor_map = neighbor_cache['neighbor_map']
            
            # im2col
            im2col = torch.zeros((N * V, Ci), device=feats.device, dtype=feats.dtype)
            mask = neighbor_map.view(-1) != 0xffffffff
            im2col[mask] = feats[neighbor_map.view(-1).long()[mask]]
            im2col = im2col.view(N, V * Ci)
            
            # addmm
            weight = weight.view(Co, V * Ci).transpose(0, 1)
            if bias is not None:
                output = torch.addmm(bias, im2col, weight)
            else:
                output = torch.mm(im2col, weight)
        
        elif spconv.ALGORITHM == Algorithm.IMPLICIT_GEMM:
            output = kernels.triton.sparse_submanifold_conv_fwd_implicit_gemm(
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map']
            )
            
        elif spconv.ALGORITHM == Algorithm.IMPLICIT_GEMM_SPLITK:
            output = kernels.triton.sparse_submanifold_conv_fwd_implicit_gemm_splitk(
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map']
            )
            
        elif spconv.ALGORITHM == Algorithm.MASKED_IMPLICIT_GEMM:
            output = kernels.triton.sparse_submanifold_conv_fwd_masked_implicit_gemm(
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map'],
                neighbor_cache['sorted_idx'],
                neighbor_cache.valid_kernel_callback,
                neighbor_cache.valid_kernel_seg_callback
            )
            
        elif spconv.ALGORITHM == Algorithm.MASKED_IMPLICIT_GEMM_SPLITK:
            output = kernels.triton.sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk(
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map'],
                neighbor_cache['sorted_idx'],
                neighbor_cache.valid_kernel_callback,
                neighbor_cache.valid_kernel_seg_callback
            )
            
        else:
            raise ValueError(f"Unsupported algorithm {spconv.ALGORITHM}")
        
        return output

    @staticmethod
    def _sparse_submanifold_conv_backward(
        grad_output: torch.Tensor,
        feats: torch.Tensor,
        neighbor_cache: SubMConv3dNeighborCache,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        N = feats.shape[0]
        Co, Kw, Kh, Kd, Ci = weight.shape
        V = Kd * Kh * Kw

        if spconv.ALGORITHM == Algorithm.EXPLICIT_GEMM:
            neighbor_map = neighbor_cache['neighbor_map']
            
            if feats.requires_grad:
                # im2col
                im2col = torch.zeros((N * V, Co), device=feats.device, dtype=feats.dtype)
                inv_neighbor_map = torch.flip(neighbor_map, [1])
                mask = inv_neighbor_map.view(-1) != 0xffffffff
                im2col[mask] = grad_output[inv_neighbor_map.view(-1).long()[mask]]
                im2col = im2col.view(N, V * Co)
                
                # addmm
                grad_input = torch.mm(im2col, weight.view(Co, V, Ci).transpose(0, 1).reshape(V * Co, Ci))
            else:
                grad_input = None
                
            if weight.requires_grad:
                # im2col
                im2col = torch.zeros((N * V, Ci), device=weight.device, dtype=weight.dtype)
                mask = neighbor_map.view(-1) != 0xffffffff
                im2col[mask] = feats[neighbor_map.view(-1).long()[mask]]
                im2col = im2col.view(N, V * Ci)
                
                # addmm
                grad_weight = torch.mm(im2col.t(), grad_output.view(N, -1)).view(V, Ci, Co).permute(2, 0, 1).contiguous().view(Co, Kw, Kh, Kd, Ci)
            else:
                grad_weight = None
            
            if bias is not None and bias.requires_grad:
                grad_bias = grad_output.sum(dim=0)
            else:
                grad_bias = None
            
        elif spconv.ALGORITHM == Algorithm.IMPLICIT_GEMM:
            grad_input, grad_weight, grad_bias = kernels.triton.sparse_submanifold_conv_bwd_implicit_gemm(
                grad_output.contiguous(),
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map']
            )
            grad_weight = grad_weight.reshape(Co, Kw, Kh, Kd, Ci)
            
        elif spconv.ALGORITHM == Algorithm.IMPLICIT_GEMM_SPLITK:
            grad_input, grad_weight, grad_bias = kernels.triton.sparse_submanifold_conv_bwd_implicit_gemm_splitk(
                grad_output.contiguous(),
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map']
            )
            grad_weight = grad_weight.reshape(Co, Kw, Kh, Kd, Ci)
            
        elif spconv.ALGORITHM == Algorithm.MASKED_IMPLICIT_GEMM:
            grad_input, grad_weight, grad_bias = kernels.triton.sparse_submanifold_conv_bwd_masked_implicit_gemm(
                grad_output.contiguous(),
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map'],
                neighbor_cache['sorted_idx'],
                neighbor_cache['valid_kernel_callback'],
                neighbor_cache['valid_kernel_seg_callback'],
                neighbor_cache['valid_signal_i'],
                neighbor_cache['valid_signal_o'],
                neighbor_cache['valid_signal_seg']
            )
            grad_weight = grad_weight.reshape(Co, Kw, Kh, Kd, Ci)
        
        elif spconv.ALGORITHM == Algorithm.MASKED_IMPLICIT_GEMM_SPLITK:
            grad_input, grad_weight, grad_bias = kernels.triton.sparse_submanifold_conv_bwd_masked_implicit_gemm_splitk(
                grad_output.contiguous(),
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map'],
                neighbor_cache['sorted_idx'],
                neighbor_cache['valid_kernel_callback'],
                neighbor_cache['valid_kernel_seg_callback'],
                neighbor_cache['valid_signal_i'],
                neighbor_cache['valid_signal_o'],
                neighbor_cache['valid_signal_seg']
            )
            grad_weight = grad_weight.reshape(Co, Kw, Kh, Kd, Ci)
            
        else:
            raise ValueError(f"Unsupported algorithm {spconv.ALGORITHM}")
        
        return grad_input, grad_weight, grad_bias
    
    @staticmethod
    def forward(
        ctx,
        feats: torch.Tensor,
        coords: torch.Tensor,
        shape: torch.Size,
        neighbor_cache: SubMConv3dNeighborCache | None,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        dilation: tuple[int, int, int] = (1, 1, 1),
    ) -> tuple[torch.Tensor, SubMConv3dNeighborCache]:
        Co, Kw, Kh, Kd, Ci = weight.shape
        assert feats.shape[-1] == Ci, f"Input channels ({feats.shape[-1]}) should match weight channels ({Ci})"
        
        # check if neighbor map is already computed
        if neighbor_cache is None:
            neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, (Kw, Kh, Kd), dilation)
            
        # compute output
        output = SubMConv3dFunction._sparse_submanifold_conv_forward(feats, neighbor_cache, weight, bias)
        
        # save for backward
        ctx.save_for_backward(feats, weight, bias)
        ctx.neighbor_cache = neighbor_cache
        
        return output, neighbor_cache
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, _):
        feats, weight, bias = ctx.saved_tensors
        neighbor_cache = ctx.neighbor_cache
        
        grad_input, grad_weight, grad_bias = SubMConv3dFunction._sparse_submanifold_conv_backward(grad_output, feats, neighbor_cache, weight, bias)
        
        if not feats.requires_grad:
            grad_input = None
        if not weight.requires_grad:
            grad_weight = None
        if not bias.requires_grad:
            grad_bias = None
        return grad_input, None, None, None, grad_weight, grad_bias, None


class SubMConv3dCompiledFunction(Function):
    """Compile-friendly autograd function for masked sparse convolution.

    Takes only tensors and plain-int/str arguments -- no Python objects
    or callbacks.  Use :meth:`SubMConv3dNeighborCache.freeze` to obtain
    a :class:`SpConvConfig` whose fields are passed here.
    """

    @staticmethod
    def forward(
        ctx,
        feats: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        neighbor_map: Tensor,
        sorted_idx: Tensor,
        valid_kernels: list[Tensor],
        valid_kernel_segs: list[Tensor],
        block_sizes: list[int],
        valid_signal_i: Tensor,
        valid_signal_o: Tensor,
        valid_signal_seg: Tensor,
        splitk: int,
        algorithm: str,
    ) -> Tensor:
        Co, V, Ci = weight.shape

        if algorithm == "masked_implicit_gemm":
            out = torch.ops.flex_gemm.sparse_conv_masked_fwd(
                feats, weight, bias, neighbor_map, sorted_idx,
                valid_kernels, valid_kernel_segs, block_sizes,
            )
        else:
            out = torch.ops.flex_gemm.sparse_conv_masked_splitk_fwd(
                feats, weight, bias, neighbor_map, sorted_idx,
                valid_kernels, valid_kernel_segs, block_sizes,
                splitk,
            )

        ctx.save_for_backward(
            feats, weight, bias,
            neighbor_map, sorted_idx,
            *valid_kernels, *valid_kernel_segs,
            valid_signal_i, valid_signal_o, valid_signal_seg,
        )
        ctx.num_block_sizes = len(block_sizes)
        ctx.block_sizes = block_sizes
        ctx.splitk = splitk
        ctx.algorithm = algorithm
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        saved = ctx.saved_tensors
        n = ctx.num_block_sizes
        feats, weight, bias = saved[0], saved[1], saved[2]
        neighbor_map, sorted_idx = saved[3], saved[4]
        valid_kernels = list(saved[5 : 5 + n])
        valid_kernel_segs = list(saved[5 + n : 5 + 2 * n])
        valid_signal_i = saved[5 + 2 * n]
        valid_signal_o = saved[5 + 2 * n + 1]
        valid_signal_seg = saved[5 + 2 * n + 2]
        Co, V, Ci = weight.shape

        if ctx.algorithm == "masked_implicit_gemm":
            grad_input, grad_weight, grad_bias = (
                torch.ops.flex_gemm.sparse_conv_masked_bwd(
                    grad_output, feats, weight, bias,
                    neighbor_map, sorted_idx,
                    valid_kernels, valid_kernel_segs, ctx.block_sizes,
                    valid_signal_i, valid_signal_o, valid_signal_seg,
                )
            )
        else:
            grad_input, grad_weight, grad_bias = (
                torch.ops.flex_gemm.sparse_conv_masked_splitk_bwd(
                    grad_output, feats, weight, bias,
                    neighbor_map, sorted_idx,
                    valid_kernels, valid_kernel_segs, ctx.block_sizes,
                    valid_signal_i, valid_signal_o, valid_signal_seg,
                    ctx.splitk,
                )
            )

        grad_weight = grad_weight.reshape(weight.shape)

        # Return gradients for: feats, weight, bias, neighbor_map, sorted_idx,
        #   valid_kernels..., valid_kernel_segs...,
        #   valid_signal_i, valid_signal_o, valid_signal_seg,
        #   splitk, algorithm
        none_list = [None] * (2 * n)  # for valid_kernels + valid_kernel_segs
        return (
            grad_input, grad_weight, grad_bias,
            None, None,              # neighbor_map, sorted_idx
            *none_list,              # valid_kernels, valid_kernel_segs
            None, None, None,        # valid_signal_i/o/seg
            None, None,              # splitk, algorithm
        )


def sparse_submanifold_conv3d(
    feats: torch.Tensor,
    coords: torch.Tensor = None,
    shape: torch.Size = None,
    weight: torch.Tensor = None,
    bias: torch.Tensor | None = None,
    neighbor_cache: SubMConv3dNeighborCache | None = None,
    dilation: tuple[int, int, int] = (1, 1, 1),
    *,
    config: SpConvConfig | None = None,
) -> tuple[torch.Tensor, SubMConv3dNeighborCache] | torch.Tensor:
    """
    Sparse submanifold convolution for 3D input.

    Supports two modes:

    **Legacy mode** (``config=None``): returns ``(output, neighbor_cache)``.

    **Compiled mode** (``config=SpConvConfig``): returns ``output`` only.
    All neighbor-structure data comes from the frozen config -- no Python
    objects cross the ``torch.compile`` boundary.

    Args:
        feats (torch.Tensor): [N, C] tensor of input features.
        coords (torch.Tensor): [N, 4] tensor of input coordinates (legacy only).
        shape (torch.Size): shape of the input tensor in NCWHD order (legacy only).
        weight (torch.Tensor): [Co, Kw, Kh, Kd, Ci] tensor of weights.
        bias (torch.Tensor | None): [Co] tensor of biases.
        neighbor_cache: neighbor cache for forward (legacy only).
        dilation: dilation rate (legacy only).
        config: frozen :class:`SpConvConfig` for the compiled path.

    Returns:
        Legacy mode: ``(output, neighbor_cache)``
        Compiled mode: ``output``
    """
    if config is not None:
        # ---- Compiled path: pure tensor args, no Python objects ----
        Co, Kw, Kh, Kd, Ci = weight.shape
        V = Kw * Kh * Kd
        return SubMConv3dCompiledFunction.apply(
            feats,
            weight.reshape(Co, V, Ci),
            bias,
            config.neighbor_map,
            config.sorted_idx,
            config.valid_kernels,
            config.valid_kernel_segs,
            config.block_sizes,
            config.valid_signal_i,
            config.valid_signal_o,
            config.valid_signal_seg,
            config.splitk,
            config.algorithm,
        )
    else:
        # ---- Legacy path: unchanged ----
        return SubMConv3dFunction.apply(
            feats, coords, shape, neighbor_cache, weight, bias, dilation
        )
