"""
LoRA-RSC (ICASSP 2025, deegy666/LoRA-RSC d1f0653) — ICBHI classifier adding LoRA to the K/V projections of HF AST (AudioSet 10/10),
faithful port of the official code (exact port of the official code, verified against goldens).

Original src/AST_LoRA_ICBHI.py AST_LoRA / src/AST.py AST (a variant of Cappellazzo PETL_AST):
  ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", max_length=500, ignore_mismatched_sizes=True)
    → 12×49=588 patches + cls/dist; position_embeddings (1,590,768) has a different shape from the ckpt (1214) and is not loaded → **stays 0, frozen** (no interpolation)
  ASTSelfAttention_LoRA: adds lora_up(lora_down(h))·scaling to key/value, lora_down: 768→round(768/96)=8, lora_up: 8→768 (no bias),
    scaling = alpha/rank_arg = 8/96 (the rank argument 96 is actually a reduction rate; the true rank is 8)
  transformers 4.28.1 from_pretrained fast-init quirk: AST._init_weights is re-applied to the new nn.Linear (lora_down/up) layers absent from the ckpt,
    overwriting the zeros_/kaiming_uniform_ of __init__ with **trunc_normal(std 0.02)** → initial LoRA delta ≠ 0 (confirmed by goldens). Kept as the default.
  embeddings/encoder frozen; LoRA + final LayerNorm + Linear(768→4) trained (299,524 parameters). A train() override keeps encoder/embeddings always in eval.
  Pooling: CLS token (hidden[:,0]) only (not the HF pooler's (cls+dist)/2).
  full-FT (src/AST.py): same setup (pos_embed 0, trainable), all parameters trained, lr 5e-5.  linear: frozen backbone + final LN and head only.

The backbone is the .venv transformers 5.15.1 ASTModel (construction only; no from_pretrained), into which the HF ckpt is loaded directly by
mapping 4.28.1 keys → 5.15.1 keys (encoder.layer.i.attention.attention.{query,key,value} → layers.i.attention.{q,k,v}_proj,
attention.output.dense → o_proj, intermediate.dense → mlp.fc1, output.dense → mlp.fc2). LoRA is reproduced by wrapping the k_proj/v_proj
nn.Linear in LoRALinear (no peft).
Input: toolkit fbank (B,128,T) — kaldi 128 mel + AudioSet constant normalization = identical to HF ASTFeatureExtractor (golden); if T>500 only the
first 500 frames are kept (original max_length=500 truncation = first 5 s of the 8 s cyclic-padded waveform).
Goldens (tests/golden/lorarsc, 4.28.1 reference venv): fast-init quirk (LoRA down std 0.020), pos_embed 0, parameters 299,524/85,711,108, logits/loss match.
Waveform noise (unknown): the public code's (1,N) array + randn(len) = scalar DC offset is removed by the fbank and has no effect, but that call itself
raises AssertionError in the 4.28.1 extractor (golden orig_2d_call) → the noise form in the authors' runs (scalar vs 1D per-sample N(0,0.3²)) is unknown.
The default config uses no noise; the 1D interpretation is data=icbhi_lorarsc_noise (datasets/lorarsc_noise.py).
"""
from __future__ import annotations

import logging
import pathlib
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseRSCModel
from .factory import ModelFactory

logger = logging.getLogger(__name__)

HF_AST_CANDIDATES = (
    "pretrained_models/upstream/hf/ast-finetuned-audioset-10-10-0.4593",   # relative to cwd (repo root); use the pretrained_dir argument for other locations
)
HF_PREFIX = "audio_spectrogram_transformer."


class LoRALinear(nn.Module):
    """nn.Linear + lora_up(lora_down(x))·scaling — key/value path of the original ASTSelfAttention_LoRA (down/up without bias)."""

    def __init__(self, base: nn.Linear, rank: int, scaling: float):
        super().__init__()
        self.base = base
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        self.scaling = float(scaling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_up(self.lora_down(x)) * self.scaling


def map_hf_ast_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """HF ckpt (4.28.1-era keys, ASTForAudioClassification) → transformers 5.x ASTModel keys. classifier.* is dropped."""
    out = {}
    for k, v in state.items():
        if not k.startswith(HF_PREFIX):
            continue  # classifier.*
        k = k[len(HF_PREFIX):]
        if k.startswith("encoder.layer."):
            parts = k.split(".")
            i, rest = parts[2], ".".join(parts[3:])
            rest = (rest.replace("attention.attention.query.", "attention.q_proj.")
                        .replace("attention.attention.key.", "attention.k_proj.")
                        .replace("attention.attention.value.", "attention.v_proj.")
                        .replace("attention.output.dense.", "attention.o_proj.")
                        .replace("intermediate.dense.", "mlp.fc1.")
                        .replace("output.dense.", "mlp.fc2."))
            k = f"layers.{i}.{rest}"
        out[k] = v
    return out


def _load_hf_state(hf_dir: pathlib.Path) -> Dict[str, torch.Tensor]:
    st = hf_dir / "model.safetensors"
    if st.exists():
        from safetensors.torch import load_file
        return load_file(str(st))
    return torch.load(hf_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)


@ModelFactory.register_model_class("ast_lora_rsc")
class ASTLoRARSC(BaseRSCModel, nn.Module):
    def __init__(
        self,
        n_cls: int = 4,
        method: str = "lora",                     # lora | full_ft | linear (original --method LoRA / full-FT / linear)
        max_length: int = 500,                    # train.yaml max_len_AST_ICBHI (number of feature frames; first 5 s of the 8 s waveform)
        num_mel_bins: int = 128,
        pretrained_dir: Optional[str] = None,     # HF snapshot directory (config.json + model.safetensors)
        pretrained: bool = True,
        lora_reduction: int = 96,                 # --reduction_rate_lora (the value the original passes as 'rank'; actual rank = round(768/96) = 8)
        lora_alpha: float = 8.0,                  # --alpha_lora; scaling = alpha / reduction (= 8/96)
        lora_targets: Sequence[str] = ("k", "v"), # original K,V (the PETL_AST prototype uses Q,V)
        lora_init: str = "hf_fast_init",          # hf_fast_init: both down/up trunc_normal(0.02) (4.28.1 quirk) | paper: down 0 / up kaiming_uniform(√5)
        initializer_range: float = 0.02,
        pos_embed: str = "zeros",                 # zeros: original quirk (not loaded, 0) | interpolate: bilinear-interpolate the AudioSet pos_embed to 12×49
        final_output: str = "CLS",                # CLS | ALL (train.yaml final_output)
        attn_implementation: str = "eager",       # eager attention as in 4.28.1 (sdpa is mathematically identical)
        input_layout: str = "freq_time",          # toolkit fbank (B,128,T)
        freeze_backbone_bn_eval: bool = True,     # original train() override: encoder/embeddings always in eval (no effect since dropout is 0)
    ):
        super().__init__()
        from transformers import ASTConfig, ASTModel

        assert method in ("lora", "full_ft", "linear"), method
        assert final_output in ("CLS", "ALL"), final_output
        self.method, self.max_length, self.final_output, self.input_layout = method, int(max_length), final_output, input_layout
        self.freeze_backbone_bn_eval = bool(freeze_backbone_bn_eval)
        self.final_feat_dim = 768

        hf_dir = self._resolve_dir(pretrained_dir)
        config = ASTConfig.from_pretrained(str(hf_dir), max_length=self.max_length, num_mel_bins=num_mel_bins)
        config._attn_implementation = attn_implementation
        self.ast = ASTModel(config)               # position_embeddings (1, 12*49+2, 768) = 0 (5.x _init_weights zeros)
        self.hidden = config.hidden_size
        self.load_info = None
        if pretrained:
            self.load_info = self._load_pretrained(hf_dir, pos_embed)
        self.pos_embed_mode = pos_embed

        # LoRA (K,V) — new Linear layers created after loading, matching the original order (load → init quirk)
        self.lora_rank = round(self.hidden / lora_reduction)
        self.lora_scaling = float(lora_alpha) / float(lora_reduction)
        self.lora_targets = tuple(lora_targets)
        if method == "lora":
            for layer in self.ast.layers:
                for t in self.lora_targets:
                    name = f"{t}_proj"
                    setattr(layer.attention, name, LoRALinear(getattr(layer.attention, name), self.lora_rank, self.lora_scaling))
            self._init_lora(lora_init, initializer_range)

        self.classification_head = nn.Linear(self.hidden, n_cls)  # nn.Linear default init (as in the original)
        self._set_trainable()

    # ----------------------------------------------------------------- construction
    @staticmethod
    def _resolve_dir(pretrained_dir: Optional[str]) -> pathlib.Path:
        cands = [pretrained_dir] if pretrained_dir else list(HF_AST_CANDIDATES)
        for c in cands:
            if c and (pathlib.Path(c) / "config.json").exists():
                return pathlib.Path(c)
        raise FileNotFoundError(f"HF AST snapshot not found: {cands}")

    def _load_pretrained(self, hf_dir: pathlib.Path, pos_embed: str) -> dict:
        state = map_hf_ast_state_dict(_load_hf_state(hf_dir))
        pe_key = "embeddings.position_embeddings"
        ckpt_pe = state.pop(pe_key)
        missing, unexpected = self.ast.load_state_dict(state, strict=False)
        assert not unexpected and set(missing) == {pe_key}, (missing, unexpected)
        if pos_embed == "interpolate":
            # toolkit extension (not in the original): AudioSet 12×101 grid → 12×49 bilinear (YuanGong ast_models.py style)
            fo, to = self.ast.embeddings.get_shape(self.ast.config)
            pe = ckpt_pe[:, 2:, :].transpose(1, 2).reshape(1, self.hidden, 12, 101)
            pe = F.interpolate(pe, size=(fo, to), mode="bilinear").reshape(1, self.hidden, fo * to).transpose(1, 2)
            with torch.no_grad():
                self.ast.embeddings.position_embeddings.copy_(torch.cat([ckpt_pe[:, :2, :], pe], dim=1))
        elif pos_embed != "zeros":
            raise ValueError(pos_embed)
        logger.info("LoRA-RSC HF AST load: %d keys, pos_embed=%s (ckpt %s not loaded)", len(state), pos_embed, tuple(ckpt_pe.shape))
        return {"loaded": len(state), "pos_embed": pos_embed, "ckpt_pos_embed_shape": list(ckpt_pe.shape)}

    def lora_modules(self):
        for layer in self.ast.layers:
            for t in self.lora_targets:
                m = getattr(layer.attention, f"{t}_proj")
                if isinstance(m, LoRALinear):
                    yield m

    def _init_lora(self, lora_init: str, std: float):
        for m in self.lora_modules():
            if lora_init == "hf_fast_init":
                for w in (m.lora_down.weight, m.lora_up.weight):
                    with torch.no_grad():
                        nn.init.trunc_normal_(w, mean=0.0, std=std)
            elif lora_init == "paper":
                nn.init.zeros_(m.lora_down.weight)
                nn.init.kaiming_uniform_(m.lora_up.weight, a=5 ** 0.5)
            else:
                raise ValueError(lora_init)

    def _set_trainable(self):
        if self.method == "full_ft":
            self.requires_grad_(True)
            return
        self.ast.requires_grad_(False)
        self.ast.layernorm.requires_grad_(True)
        self.classification_head.requires_grad_(True)
        if self.method == "lora":
            for m in self.lora_modules():
                m.lora_down.requires_grad_(True)
                m.lora_up.requires_grad_(True)

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.method != "full_ft" and self.freeze_backbone_bn_eval:
            self.ast.embeddings.eval()
            self.ast.layers.eval()
        return self

    # ------------------------------------------------------------------ execution
    def to_model_input(self, x: torch.Tensor) -> torch.Tensor:
        """(B,128,T) or (B,1,128,T) → (B, max_length, 128); if T>max_length keep the first frames (HF max_length truncation), if shorter zero-pad.

        Note (unreachable path): T<max_length padding is 0 **after** normalization, whereas HF ASTFeatureExtractor applies ZeroPad2d before
        normalization, so its padded frames are (0-mean)/(2std)=+0.467. Toolkit data configs use 8 s cyclic padding (798 frames ≥ 500), so this branch never occurs."""
        if x.dim() == 4:
            x = x.squeeze(1)
        if self.input_layout == "freq_time":
            x = x.transpose(-2, -1)
        T = x.shape[1]
        if T > self.max_length:
            x = x[:, : self.max_length]
        elif T < self.max_length:
            x = F.pad(x, (0, 0, 0, self.max_length - T))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_model_input(x)
        hidden = self.ast(input_values=x).last_hidden_state  # includes the final layernorm
        feat = hidden[:, 0] if self.final_output == "CLS" else hidden.mean(dim=1)
        return self.classification_head(feat)

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        logits = self.forward(audio_batch)
        return logits, loss_func(logits, labels)
