import subprocess
import re
from typing import Optional, Tuple


def get_nvml_vram_info() -> Tuple[int, int, int]:
    """
    Query NVML / nvidia-smi for GPU 0 memory (total_mib, used_mib, free_mib).
    Returns (total_mib, used_mib, free_mib).
    """
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,nounits,noheader"],
            capture_output=True, text=True, check=True
        )
        parts = [int(p.strip()) for p in res.stdout.strip().split(",")]
        return parts[0], parts[1], parts[2]
    except Exception as e:
        # Fallback to conservative estimate if nvidia-smi is unavailable
        return 4096, 2048, 2048


def check_vram_headroom(required_free_mib: int = 1500) -> bool:
    """
    Check if at least required_free_mib is currently free on GPU 0.
    """
    _, _, free_mib = get_nvml_vram_info()
    return free_mib >= required_free_mib
