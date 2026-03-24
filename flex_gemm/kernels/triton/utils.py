import torch
import triton


def get_gpu_name() -> str:
    return torch.cuda.get_device_name()


def get_platform_name() -> str:
    if torch.cuda.is_available():
        if getattr(torch.version, 'hip', None) is not None:
            return 'hip'
        return 'cuda'
    return 'unknown'
    

def get_num_sm() -> int:
    return torch.cuda.get_device_properties("cuda").multi_processor_count
    

def get_autotune_config(
    default: list[triton.Config] | None = None,
    platform: dict[str, list[triton.Config]] | None = None,
    device: dict[str, list[triton.Config]] | None = None,
) -> list[triton.Config]:
    """
    Get the autotune configuration for the current platform and device.
    """
    if device is not None:
        gpu_name = get_gpu_name()
        for key, value in device.items():
            if key.lower() in gpu_name.lower():
                return value
    
    if platform is not None:
        platform_name = get_platform_name()
        for key, value in platform.items():
            if key.lower() in platform_name.lower():
                return value
    
    if default is None:
        raise ValueError("No autotune configuration found for the current platform and device.")
    return default
