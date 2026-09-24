# Portions of this file (TfmEncHead, MLPHead) are adapted from EVAR (https://github.com/nttcslab/eval-audio-repr); that code is under the NTT Software License Agreement for Evaluation.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
OPERA-CT classification model — reproduces the frozen evaluation path of NTT EMBC'25 (arXiv 2504.18004) (eval-audio-repr app/icbhi_sprs/solve.py + evar/ar_opera.py).

TaskNetwork(solve.py): ar.encode_frames(waveform batch) → [B, 768, 1] → transpose → [B, 1, 768] → head.
  - AR_OPERA_CT.encode_frames: get_entire_signal_librosa(batch trim·mel64·dB·batch min-max) → augment_if_training(evar SpecAugment:
    torchaudio FrequencyMasking(20)·TimeMasking(40), iid_masks=False → **the whole batch shares one mask**) → (B,T,64) → Cola.extract_feature(768)
    = HTSAT latent_output.
  - Head 'tfm' = TfmEncHead(sem token + timm Block×4, heads 1, mlp_ratio 1, Linear gets default init after the xavier init — original order kept),
    'mlp' = MLP(BatchNorm1d(affine=False) → 768-128-n_cls, Dropout 0.5).
  - freeze_body=True (EMBC --freeze_body): only sets requires_grad=False on the backbone. model.train() is unchanged, so bn0 uses batch statistics and updates running statistics,
    and DropPath stays active (quirk of the original). freeze_body=False is full fine-tuning on the toolkit's standard task (no paper-reported value).
Training loop (scl ce.py, split_iter 1): transform the whole batch → forward → CE; best-on-test Score is handled by a callback.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torchaudio

from .base import BaseRSCModel
from .factory import ModelFactory
from .opera import HTSATSwin, load_opera_ct_checkpoint
from rsc_toolkit.processors.opera_mel import OperaBatchMel

logger = logging.getLogger(__name__)


class TfmEncHead(nn.Module):
    """solve.py TfmEncHead — unchanged."""

    def __init__(self, embed_dim=768, output_dim=3, depth=2, heads=1, mlp_ratio=1):
        super().__init__()
        from timm.models.vision_transformer import Block

        self.sem_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.sem_blocks = nn.ModuleList([Block(embed_dim, heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)
        self.linear = nn.Linear(embed_dim, output_dim)  # created after apply → default init (original order)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):  # x: (B, T, D)
        x = torch.cat((self.sem_token.expand(x.shape[0], -1, -1), x), dim=1)
        for blk in self.sem_blocks:
            x = blk(x)
        return self.linear(self.norm(x[:, 0, :]))


class MLPHead(nn.Module):
    """solve.py MLP — unchanged (input [B,T,D] averaged over T, then BatchNorm1d(affine=False))."""

    def __init__(self, input_size, hidden_sizes, output_size, hidden_dropout=0.5, mean=0.0, std=0.01, bias=0.0):
        super().__init__()
        self.norm = nn.BatchNorm1d(input_size, affine=False)
        sizes = [input_size] + list(hidden_sizes) + [output_size]
        fcs = []
        for l, (i, o) in enumerate(zip(sizes[:-1], sizes[1:])):
            if l > 0:
                fcs.append(nn.Dropout(hidden_dropout))
            linear = nn.Linear(i, o)
            nn.init.normal_(linear.weight, mean=mean, std=std)
            nn.init.constant_(linear.bias, bias)
            fcs.append(linear)
            fcs.append(nn.ReLU())
        self.mlp = nn.Sequential(*fcs[:-1])

    def forward(self, x):
        x = torch.mean(x, dim=-2)
        x = self.norm(x.unsqueeze(-1)).squeeze(-1)
        return self.mlp(x)


@ModelFactory.register_model_class("opera_ct")
class OperaCT(BaseRSCModel, nn.Module):
    def __init__(
        self,
        n_cls: int = 4,
        pretrained_path: Optional[str] = None,   # encoder-operaCT.ckpt (random init if None)
        freeze_body: bool = True,                # EMBC --freeze_body
        head: str = "tfm",                       # tfm | mlp
        head_depth: int = 4, head_heads: int = 1, head_mlp_ratio: int = 1,
        freq_mask: int = 20, time_mask: int = 40,  # scl args defaults (evar SpecAugment, mask shared across the batch)
        input_sec: float = 8.0, sample_rate: int = 16000,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.htsat = HTSATSwin(drop_path_rate=drop_path_rate)
        if pretrained_path is not None:
            from rsc_toolkit.paths import resolve_pretrained

            dropped = load_opera_ct_checkpoint(self.htsat, str(resolve_pretrained(explicit=pretrained_path)))  # honors RSCTK_PRETRAINED_ROOT
            logger.info("Loaded OPERA-CT ckpt %s (dropped %d keys: torchlibrosa front end, COLA projector)", pretrained_path, len(dropped))
        self.freeze_body = bool(freeze_body)
        if self.freeze_body:
            for p in self.htsat.parameters():
                p.requires_grad = False
        self.feature_d = self.htsat.num_features  # 768
        if head == "tfm":
            self.head = TfmEncHead(embed_dim=self.feature_d, output_dim=n_cls, depth=head_depth, heads=head_heads, mlp_ratio=head_mlp_ratio)
        elif head == "mlp":
            self.head = MLPHead(self.feature_d, (128,), n_cls)
        else:
            raise ValueError(f"head must be tfm|mlp: {head!r}")
        self.mel = OperaBatchMel(sample_rate=sample_rate, input_sec=input_sec)
        self.freqmask = torchaudio.transforms.FrequencyMasking(freq_mask) if freq_mask > 0 else None
        self.timemask = torchaudio.transforms.TimeMasking(time_mask) if time_mask > 0 else None

    # ---- transform (per batch) ----
    def val_transform(self, wave: torch.Tensor) -> torch.Tensor:
        return self.mel(wave)  # (B, 64, T)

    def train_transform(self, wave: torch.Tensor) -> torch.Tensor:
        x = self.mel(wave)
        with torch.no_grad():
            if self.freqmask is not None:
                x = self.freqmask(x)
            if self.timemask is not None:
                x = self.timemask(x)
        return x

    def encode(self, feats: torch.Tensor) -> torch.Tensor:
        """(B, 64, T) → HTSAT latent (B, 768) — backbone part of AR_OPERA_CT.encode_frames."""
        return self.htsat(feats.transpose(-2, -1).unsqueeze(1))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(feats).unsqueeze(1))  # [B,D] → [B,1,D] single token

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        x = self.train_transform(audio_batch) if self.training else self.val_transform(audio_batch)
        logits = self.forward(x)
        return logits, loss_func(logits, labels)
