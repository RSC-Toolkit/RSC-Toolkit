"""Toolkit re-implementation of the M2D (Masked Modeling Duo, NTT) encoder — the official code (evaluation-only license) is not vendored."""
from .encoder import M2DEncoder, get_2d_sincos_pos_embed, load_m2d_checkpoint

__all__ = ["M2DEncoder", "get_2d_sincos_pos_embed", "load_m2d_checkpoint"]
