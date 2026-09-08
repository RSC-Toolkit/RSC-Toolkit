"""데이터셋 ↔ 모델 배치 계약 — 한 곳에 적어 둔 것 (리팩토링 P2-7).

예전에는 이 계약이 네 곳에 흩어져 있었다: IndexDataModule docstring(dict 항목), IndexFeatureDataset(위치 튜플),
collate(4번째 원소 = meta), PLAdapter(`batch[0].squeeze(1)`, `batch[2] = meta_labels`). 커스텀 데이터셋을 만들려면
전부 역공학해야 했다. 여기서는 **바꾸지 않고** 정리만 한다 — 배치 레이아웃 자체는 20여 편 이식 모델이 걸려 있어
불변이다.

1. 인덱스 데이터셋 항목 (`IndexDataset.__getitem__`): dict
       {"audio": Tensor(1, L) 또는 (L,), "label": int | None, "sample_id": str, "meta": dict(옵션), ...}
   필수 인덱스 CSV 열은 `rsc_toolkit.datasets.sample_index.REQUIRED_INDEX_COLUMNS`.
   `sample_level` 어휘는 `SAMPLE_LEVELS` (cycle | clip).

2. 피처 래퍼 항목 (`IndexFeatureDataset.__getitem__`): 위치 튜플
       (feats, label, sample_id[, meta])
   meta 는 datamodule 이 `meta_label_fn` 을 줬을 때만 4번째 원소로 붙는다.

3. 배치 (`rsc_toolkit.datasets.collate`): 위치 튜플
       (x, y[, meta])
   - x = feats stack (B, ...) — PLAdapter 가 `batch[0].squeeze(1)` 로 채널 축을 뗀다
   - y = long (B,)  (PC-MCL 학습 배치만 float (B, 3) 멀티라벨 — 아래 META_KINDS 의 pcmcl 참조)
   - meta = 3번째 원소. **의미는 datamodule 마다 다르다** — 그래서 datamodule 은 `meta_kind` 를, 모델은
     `accepts_meta` 를 선언하고 PLAdapter.setup 이 대조한다 (rsc_toolkit.trainer.adapters.pl_adapter.check_meta_contract).

4. 모델 (`BaseRSCModel.training_forward(audio_batch, labels, loss_func, **kwargs)`): `(logits, loss)`.
   meta 는 `meta_labels=` 키워드로 들어온다. 쓰지 않는 모델도 `**kwargs` 를 열어 둔다.
"""
from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional, Tuple

import torch

SAMPLE_LEVELS: Tuple[str, ...] = ("cycle", "clip")

# 트리 안의 meta_kind 표 — datamodule 이 batch[2] 로 무엇을 보내는지. 새 datamodule 은 자기 kind 를 클래스 속성으로
# 선언하면 되고(여기 등록은 문서용), 받는 모델은 accepts_meta 에 그 이름을 적는다.
META_KINDS: Dict[str, Dict[str, Any]] = {
    "dev": {"dtype": "long (B,)", "meaning": "ICBHI 청진기 id 0..3", "emitter": "IndexDataModule(meta_label_mode=dev)", "receivers": ("ast_sgscl", "resnet18/34_sft", "efficientnet_b0_sft", "cnn6_sft")},
    "patient": {"dtype": "long (B,)", "meaning": "환자 번호", "emitter": "IndexDataModule(meta_label_mode=patient)", "receivers": ("beats",)},
    "sa": {"dtype": "long (B,)", "meaning": "성별×연령군 0..3 (ICBHI)", "emitter": "IndexDataModule(meta_label_mode=sa)", "receivers": ("cnn6_sclmeta",)},
    "sa_sprs": {"dtype": "long (B,)", "meaning": "gender×age_group 0..3 (SPRSound)", "emitter": "IndexDataModule(meta_label_mode=sa_sprs)", "receivers": ("cnn6_sclmeta",)},
    "qlung_aqs": {"dtype": "float64 (B,)", "meaning": "AQS 음향 품질 점수", "emitter": "QLungDataModule", "receivers": ("ast_qlung", "clap_qlung")},
    "pcmcl_ssl_label": {"dtype": "long (B,) ∈ {-1,0,1}", "meaning": "PC-MCL 쌍 라벨 (학습 y 는 float (B,3) 멀티라벨)", "emitter": "PCMCLDataModule", "receivers": ("ast_pcmcl", "beats_pcmcl")},
    "bts_meta7": {"dtype": "float (B, 7)", "meaning": "BTS 메타 7필드 벡터 (train·val 양쪽)", "emitter": "BTSPrecomputedDataModule, BTSCARDSprsDataModule", "receivers": ("bts", "bts_card", "audioclap")},
    "amsd_sample_index": {"dtype": "long (B,)", "meaning": "학습 샘플 인덱스 (per-sample 버퍼)", "emitter": "AMSDDataModule", "receivers": ("ast_amsd",)},
    "respirenet_chunk_index": {"dtype": "long (B,)", "meaning": "청크 인덱스 (사이클 투표 콜백용)", "emitter": "RespireNetDataModule", "receivers": ("respirenet",)},
    "lungmix_valid_len": {"dtype": "long (B,)", "meaning": "패딩 전 원 구간 길이(샘플 수)", "emitter": "LungmixDataModule", "receivers": ("ast_lungmix",)},
}


class Batch(NamedTuple):
    """배치 튜플의 이름 붙은 뷰 — `Batch.from_tuple(batch)` 로 감싸 읽는다 (레이아웃 자체는 튜플 그대로)."""

    x: torch.Tensor
    y: torch.Tensor
    meta: Optional[torch.Tensor] = None

    @classmethod
    def from_tuple(cls, batch) -> "Batch":
        if len(batch) < 2 or len(batch) > 3:
            raise ValueError(f"배치는 (x, y[, meta]) 2~3 원소여야 한다: len={len(batch)}")
        return cls(*batch)


def describe_batch(batch) -> Dict[str, Any]:
    """디버그용: 각 원소의 모양·dtype (meta 가 있으면 함께)."""
    b = Batch.from_tuple(batch)
    out = {"x": (tuple(b.x.shape), str(b.x.dtype)), "y": (tuple(b.y.shape), str(b.y.dtype))}
    if b.meta is not None:
        out["meta"] = (tuple(b.meta.shape), str(b.meta.dtype))
    return out


__all__ = ["SAMPLE_LEVELS", "META_KINDS", "Batch", "describe_batch"]
