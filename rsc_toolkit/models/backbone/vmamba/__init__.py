"""VMamba VSSM backbone (Lung-SRAD / DASS-medium) — pure-torch selective scan fallback + optional CUDA kernel."""
from .cross_scan import CrossMerge, CrossScan
from .selective_scan import cuda_kernel_available, resolve_backend, selective_scan, selective_scan_torch
from .vssm import (
    DASS_MEDIUM_KWARGS,
    MIX_LAYOUTS,
    MIX_TYPES,
    SS2D,
    VSSM,
    VSSBlock,
    LayerNorm2d,
    Linear2d,
    blur_separable,
    build_dass_medium_vssm,
    gaussian_kernel1d,
    load_dass_checkpoint,
    strip_dass_state_dict,
)

__all__ = [
    "CrossMerge", "CrossScan", "cuda_kernel_available", "resolve_backend", "selective_scan", "selective_scan_torch",
    "DASS_MEDIUM_KWARGS", "MIX_LAYOUTS", "MIX_TYPES", "SS2D", "VSSM", "VSSBlock", "LayerNorm2d", "Linear2d",
    "blur_separable", "build_dass_medium_vssm", "gaussian_kernel1d", "load_dass_checkpoint", "strip_dass_state_dict",
]
