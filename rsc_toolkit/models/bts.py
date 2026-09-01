"""
BTS 논문(Interspeech 2024)의 메타데이터-텍스트 융합 CLAP — 공식 코드 충실 이식 (P0).

출처:
- 논문: BTS: Bridging Text and Sound Modalities for Metadata-Aided Respiratory Sound
  Classification (https://arxiv.org/abs/2406.06786)
- 공식 코드: https://github.com/kaen2891/bts models/clap.py PretrainedCLAP + main.py set_model()
  (--model_type ClapModel --clap_final concat --meta_mode all)

구조: HF ClapModel(laion/clap-htsat-unfused; 오디오 HTSAT + 텍스트 RoBERTa, 각 512-d
투영·L2 정규화) → text_embeds ⊕ audio_embeds (concat 1024-d, 또는 add 가중합)
→ nn.Linear(1024, n_cls). 텍스트는 환자 메타(나이·성별·청진 위치·장비)를 문장으로 만든 것
(datasets/icbhi/meta_description.py)으로, 원본은 dataset __getitem__ 에서 토크나이즈
(padding="max_length", max_length=64)한다 — 여기서는 모델이 배치의 meta 벡터를 받아
같은 토크나이저로 같은 방식으로 인코딩한다 (수치 동일).

원본 유지 사항:
- 속성 이름 audio_features (원본 PretrainedCLAP.audio_features) 를 그대로 둬서 저자
  체크포인트(best.pth: model=audio_features.*, classifier=weight/bias)를 키 변환 없이 로드.
- 학습 시 텍스트 인코더도 함께 미세조정된다 (원본 optimizer 에 model.parameters() 전부).
- 손실은 비가중 CE (원본 set_model: nn.CrossEntropyLoss()).

버전 차이: transformers 5.x 의 ClapModel 은 input_features 를 (B, 1, T, F) 4D 로만
받는다(원본은 구버전에서 (B, T, F) 를 넘김). 3D 가 오면 채널 축을 붙인다.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .base import BaseRSCModel
from .factory import ModelFactory
from rsc_toolkit.datasets.icbhi.meta_description import (
    describe_meta7,
    generate_meta_description,
    meta_fields_from_meta7,
    select_meta_fields,
)

logger = logging.getLogger(__name__)


def _as_4d(input_features: torch.Tensor) -> torch.Tensor:
    return input_features.unsqueeze(1) if input_features.dim() == 3 else input_features


@ModelFactory.register_model_class("bts")
class BTSModel(BaseRSCModel, nn.Module):
    def __init__(
        self,
        n_cls: int = 4,
        pretrained_name: str = "laion/clap-htsat-unfused",
        final_feat_dim: int = 512,
        clap_final: str = "concat",     # "concat" | "add"
        te_alpha: float = 0.5,          # add 일 때 텍스트 가중 (원본 기본값)
        meta_mode: str = "all",
        text_max_length: int = 64,      # 원본 icbhi_dataset.py:189
        # 메타 문장 어휘. "icbhi" = 원본(흉부 위치 7종 + 청진기 4종).
        # "sprsound" = 저자 util/sprsound_util.py 의 SPRSound 판(위치 p1~p4, 청진기 없음) —
        # SPRSound 로 직접 학습할 때만 쓴다. 어휘를 안 바꾸면 없는 청진 위치를 지어내게 된다.
        meta_dialect: str = "icbhi",
    ):
        super().__init__()
        from transformers import ClapModel, ClapProcessor

        if clap_final not in {"concat", "add"}:
            raise ValueError(f"clap_final must be concat|add, got {clap_final!r}")
        self.audio_features = ClapModel.from_pretrained(pretrained_name)
        self.final_feat_dim = final_feat_dim
        self.clap_final = clap_final
        self.te_alpha = te_alpha
        self.meta_mode = meta_mode
        if meta_dialect not in {"icbhi", "sprsound"}:
            raise ValueError(f"meta_dialect must be icbhi|sprsound, got {meta_dialect!r}")
        self.meta_dialect = meta_dialect
        self.text_max_length = int(text_max_length)
        head_dim = final_feat_dim * 2 if clap_final == "concat" else final_feat_dim
        self.classifier = nn.Linear(head_dim, n_cls)  # 원본 main.py:191
        # 원본은 ClapProcessor(...).tokenizer 를 쓴다 — 같은 객체로 맞춘다
        self.tokenizer = ClapProcessor.from_pretrained(pretrained_name).tokenizer

    # ----------------------------------------------------------------- 텍스트
    def meta_field_dicts(self, meta7_batch: torch.Tensor) -> list:
        """meta7 → 문장 필드 dict. 방언에 따라 어휘와 문장 틀이 갈린다."""
        if self.meta_dialect == "sprsound":
            from rsc_toolkit.datasets.sprsound_ood import meta_fields_from_meta7_sprsound

            return [meta_fields_from_meta7_sprsound(m.tolist()) for m in meta7_batch]
        return [select_meta_fields(meta_fields_from_meta7(m.tolist()), self.meta_mode) for m in meta7_batch]

    def render_meta_sentence(self, fields: dict) -> str:
        """문장 생성기. SPRSound 는 저자의 util/sprsound_util.py 판을 쓴다(청진기 항이 없다)."""
        if self.meta_dialect == "sprsound":
            from rsc_toolkit.datasets.sprsound_ood import generate_meta_description_sprsound

            return generate_meta_description_sprsound(
                age=fields["age"] if fields["age"] is not None else "unknown",
                sex=fields["sex"] if fields["sex"] is not None else "unknown",
                loc=fields["loc"] if fields["loc"] is not None else "unknown",
            )
        return generate_meta_description(**fields)

    def describe(self, meta7_batch: torch.Tensor) -> list:
        if self.meta_dialect == "sprsound":
            return [self.render_meta_sentence(d) for d in self.meta_field_dicts(meta7_batch)]
        return [describe_meta7(m.tolist(), self.meta_mode) for m in meta7_batch]

    def encode_meta(self, meta7_batch: torch.Tensor, device) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            text=self.describe(meta7_batch), return_tensors="pt",
            padding="max_length", max_length=self.text_max_length,
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    # ------------------------------------------------------------------ 모델
    def forward_features(self, input_features, input_ids, attention_mask) -> torch.Tensor:
        """원본 PretrainedCLAP.forward (models/clap.py:26-36)."""
        out = self.audio_features(
            input_ids=input_ids, attention_mask=attention_mask, input_features=_as_4d(input_features)
        )
        text_embeds, audio_embeds = out.text_embeds, out.audio_embeds
        if self.clap_final == "concat":
            return torch.cat((text_embeds, audio_embeds), dim=-1)
        return (text_embeds * self.te_alpha) + (audio_embeds * (1 - self.te_alpha))

    def forward(self, x: torch.Tensor, meta7: torch.Tensor) -> torch.Tensor:
        ids, mask = self.encode_meta(meta7, x.device)
        return self.classifier(self.forward_features(x, ids, mask))

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        if meta_labels is None:
            raise ValueError("BTS 는 배치에 meta(7필드) 가 필요하다 — data=bts_precomputed 를 쓸 것")
        logits = self.forward(audio_batch, meta_labels)
        return logits, loss_func(logits, labels)

    # ----------------------------------------------------- 저자 체크포인트
    def load_reference_checkpoint(self, path) -> None:
        """
        원본 save_model 산출물(best.pth: {'model': audio_features.*, 'classifier': {...}})을
        키 변환 없이 strict 로드한다 (구조 동일성의 증명이기도 하다).
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = dict(ckpt["model"])
        state.update({f"classifier.{k}": v for k, v in ckpt["classifier"].items()})
        self.load_state_dict(state, strict=True)
        logger.info("BTS 저자 체크포인트 로드: %s (epoch %s)", path, ckpt.get("epoch"))
