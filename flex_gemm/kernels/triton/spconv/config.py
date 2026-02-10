import triton
from ..utils import get_autotune_config


allow_tf32 = True


autotune_config = get_autotune_config(
    platform={
        'cuda': [
            triton.Config({'B1': 128, 'B2': 256, 'BK': 64}, num_stages=3, num_warps=8),
            triton.Config({'B1': 64,  'B2': 256, 'BK': 32}, num_stages=4, num_warps=4),
            triton.Config({'B1': 128, 'B2': 128, 'BK': 32}, num_stages=4, num_warps=4),
            triton.Config({'B1': 128, 'B2': 64,  'BK': 32}, num_stages=4, num_warps=4),
            triton.Config({'B1': 64,  'B2': 128, 'BK': 32}, num_stages=4, num_warps=4),
            triton.Config({'B1': 128, 'B2': 32,  'BK': 32}, num_stages=4, num_warps=4),
            triton.Config({'B1': 64,  'B2': 32,  'BK': 32}, num_stages=5, num_warps=2),
            triton.Config({'B1': 32,  'B2': 64,  'BK': 32}, num_stages=5, num_warps=2),
        ],
        'hip': [
            triton.Config({'B1': 128, 'B2': 256, 'BK': 16, 'waves_per_eu': 2}, num_warps=4, num_stages=2),
            triton.Config({'B1': 256, 'B2': 256, 'BK': 16, 'waves_per_eu': 2}, num_warps=8, num_stages=2),
            triton.Config({'B1': 128, 'B2': 128, 'BK': 32, 'waves_per_eu': 2}, num_warps=8, num_stages=2),
            triton.Config({'B1': 64,  'B2': 128, 'BK': 32, 'waves_per_eu': 3}, num_warps=4, num_stages=2),
            triton.Config({'B1': 64,  'B2': 64,  'BK': 32, 'waves_per_eu': 8}, num_warps=4, num_stages=2),
        ]
    },
    device={
        'B200': [
            # Accum regs/thread = B1*B2/(num_warps*32), keep <=64 for 2+ blocks/SM
            # Key findings from profiling (RES=128, C=512, N=64160):
            #   - BK=64 is ~30% faster than BK=32 (fewer loop iters, less index overhead)
            #   - Wider B2 with more warps increases input gather reuse
            #   - num_stages=2 is optimal; stages=3,4 don't improve
            #   - BK=128 slightly slower than BK=64 (diminishing returns)
            # Top configs (accum=64 regs, ~25% occupancy):
            triton.Config({'B1': 64,  'B2': 256, 'BK': 64}, num_stages=2, num_warps=8),  # best: +14% vs prev
            triton.Config({'B1': 64,  'B2': 128, 'BK': 64}, num_stages=2, num_warps=4),
            triton.Config({'B1': 128, 'B2': 256, 'BK': 64}, num_stages=2, num_warps=8),
            triton.Config({'B1': 128, 'B2': 128, 'BK': 64}, num_stages=2, num_warps=8),
            triton.Config({'B1': 64,  'B2': 128, 'BK': 128}, num_stages=2, num_warps=4),
            # Fallback configs for smaller Co:
            triton.Config({'B1': 64,  'B2': 128, 'BK': 32}, num_stages=2, num_warps=4),
            triton.Config({'B1': 128, 'B2': 64,  'BK': 64}, num_stages=2, num_warps=4),
            triton.Config({'B1': 64,  'B2': 64,  'BK': 64}, num_stages=2, num_warps=4),
            # Small problem sizes:
            triton.Config({'B1': 64,  'B2': 64,  'BK': 32}, num_stages=2, num_warps=4),
            triton.Config({'B1': 32,  'B2': 64,  'BK': 32}, num_stages=2, num_warps=2),
        ],
        'A100': [
            triton.Config({'B1': 256, 'B2': 128, 'BK': 64}, num_stages=4, num_warps=8),
            triton.Config({'B1': 256, 'B2': 128, 'BK': 32}, num_stages=4, num_warps=8),
            triton.Config({'B1': 128, 'B2': 256, 'BK': 64}, num_stages=4, num_warps=8),
            triton.Config({'B1': 128, 'B2': 256, 'BK': 32}, num_stages=4, num_warps=8),
            triton.Config({'B1': 256, 'B2': 64,  'BK': 64}, num_stages=4, num_warps=4),
            triton.Config({'B1': 256, 'B2': 64,  'BK': 32}, num_stages=4, num_warps=4),
            triton.Config({'B1': 64,  'B2': 256, 'BK': 64}, num_stages=4, num_warps=4),
            triton.Config({'B1': 64,  'B2': 256, 'BK': 32}, num_stages=4, num_warps=4),
            triton.Config({'B1': 128, 'B2': 128, 'BK': 64}, num_stages=4, num_warps=4),
            triton.Config({'B1': 128, 'B2': 128, 'BK': 32}, num_stages=4, num_warps=4),
            triton.Config({'B1': 128, 'B2': 64,  'BK': 32}, num_stages=4, num_warps=4),
            triton.Config({'B1': 128, 'B2': 64,  'BK': 32}, num_stages=4, num_warps=2),
            triton.Config({'B1': 64,  'B2': 128, 'BK': 32}, num_stages=4, num_warps=4),
            triton.Config({'B1': 64,  'B2': 128, 'BK': 32}, num_stages=4, num_warps=2),
            triton.Config({'B1': 64,  'B2': 64,  'BK': 64}, num_stages=4, num_warps=2),
            triton.Config({'B1': 64,  'B2': 64,  'BK': 32}, num_stages=4, num_warps=2),
        ],
        'MI300X': [
            triton.Config({'B1': 256, 'B2': 256, 'BK': 64, 'waves_per_eu': 2}, num_stages=2, num_warps=16),
            triton.Config({'B1': 256, 'B2': 256, 'BK': 64, 'waves_per_eu': 2, 'kpack': 2, 'matrix_instr_nonkdim': 16}, num_stages=2, num_warps=8),
            triton.Config({'B1': 256, 'B2': 128, 'BK': 64, 'waves_per_eu': 2}, num_stages=2, num_warps=16),
            triton.Config({'B1': 256, 'B2': 128, 'BK': 64, 'waves_per_eu': 2, 'kpack': 2, 'matrix_instr_nonkdim': 16}, num_stages=2, num_warps=8),
            triton.Config({'B1': 128, 'B2': 256, 'BK': 64, 'waves_per_eu': 2}, num_stages=2, num_warps=16),
            triton.Config({'B1': 128, 'B2': 256, 'BK': 64, 'waves_per_eu': 2, 'kpack': 2, 'matrix_instr_nonkdim': 16}, num_stages=2, num_warps=8),
            triton.Config({'B1': 256, 'B2': 64,  'BK': 32, 'waves_per_eu': 2}, num_stages=2, num_warps=8),
            triton.Config({'B1': 256, 'B2': 64,  'BK': 32, 'waves_per_eu': 2, 'kpack': 2, 'matrix_instr_nonkdim': 16}, num_stages=2, num_warps=8),
            triton.Config({'B1': 64,  'B2': 256, 'BK': 32, 'waves_per_eu': 2}, num_stages=2, num_warps=8),
            triton.Config({'B1': 64,  'B2': 256, 'BK': 32, 'waves_per_eu': 2, 'kpack': 2, 'matrix_instr_nonkdim': 16}, num_stages=2, num_warps=8),
            triton.Config({'B1': 128, 'B2': 128, 'BK': 64, 'waves_per_eu': 2}, num_stages=2, num_warps=8),
            triton.Config({'B1': 128, 'B2': 128, 'BK': 64, 'waves_per_eu': 2, 'kpack': 2, 'matrix_instr_nonkdim': 16}, num_stages=2, num_warps=8),
            triton.Config({'B1': 128, 'B2': 64,  'BK': 64, 'waves_per_eu': 2}, num_stages=2, num_warps=4),
            triton.Config({'B1': 128, 'B2': 64,  'BK': 64, 'waves_per_eu': 2, 'kpack': 2, 'matrix_instr_nonkdim': 16}, num_stages=2, num_warps=4),
            triton.Config({'B1': 64,  'B2': 128, 'BK': 64, 'waves_per_eu': 2}, num_stages=2, num_warps=4),
            triton.Config({'B1': 64,  'B2': 128, 'BK': 64, 'waves_per_eu': 2, 'kpack': 2, 'matrix_instr_nonkdim': 16}, num_stages=2, num_warps=4),
            triton.Config({'B1': 64,  'B2': 64,  'BK': 64, 'waves_per_eu': 2}, num_stages=2, num_warps=2),
            triton.Config({'B1': 64,  'B2': 64,  'BK': 64, 'waves_per_eu': 2, 'kpack': 2, 'matrix_instr_nonkdim': 16}, num_stages=2, num_warps=2),
        ],
    }
)
