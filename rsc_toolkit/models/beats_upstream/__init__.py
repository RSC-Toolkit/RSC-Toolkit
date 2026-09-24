"""
BEATs (Microsoft, MIT) — the BEATs/ directory of the wa976/PAFA repository copied **verbatim** (only imports made relative).

Upstream: https://github.com/microsoft/unilm/tree/master/beats (BEATs.py, backbone.py, modules.py).
The PAFA paper numbers were obtained with this modified copy, so it is the reference rather than the unilm original.
Differences from the unilm original (checked by diff):
- The `waveform * 2**15` scaling is removed from BEATs.preprocess (commented out). The fbank normalization constants
  (mean 15.41663 / std 6.55582) assume scaled input, so the distribution is off, but the original behavior is kept.
- A spec_transform (SpecAugment) hook is added to BEATs.extract_features (unused, since the PAFA script uses --nospec).
- `x = x.clone() + x_conv` in backbone.TransformerEncoder (avoids in-place ops); everything else is black formatting.
quantizer.py / Tokenizers.py are not copied because fine-tuning does not use them.
"""
from .BEATs import BEATs, BEATsConfig

__all__ = ["BEATs", "BEATsConfig"]
