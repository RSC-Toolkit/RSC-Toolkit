"""전 컴포넌트 registry 허브 — 여기 lazy 선언된 것이 '내장 목록'의 단일 진실이다.

이 모듈은 무거운 것을 아무것도 import 하지 않는다. 실제 import 는 첫 조회 때 일어나므로
한 논문 모듈의 선택 의존성(librosa 등)이 빠져도 그 항목만 실패하고 나머지는 산다.

서드파티는 같은 registry 에 데코레이터로 등록한다:

    from rsc_toolkit.registries import MODELS, LOSSES
    @MODELS.register("my_model")
    class MyModel(BaseRSCModel, nn.Module): ...
    @LOSSES.register("focal")
    class FocalLoss(nn.Module): ...

등록 모듈은 엔트리포인트 그룹 `rsc_toolkit.plugins` 또는 `RSC_TOOLKIT_PLUGINS` 환경변수로
불러온다 (rsc_toolkit.registry 참조).
"""
from __future__ import annotations

from rsc_toolkit.registry import Registry

# 모델은 패키지 스캔으로 등록된다 (각 모듈의 데코레이터). base/factory 는 모델이 아니다.
MODELS = Registry("model", scan_packages=("rsc_toolkit.models",), scan_exclude=("base", "factory"))
DATASETS = Registry("dataset")            # 다운로드·인덱스·CLI (DatasetFactory 가 위임한다)
DATAMODULES = Registry("datamodule")
LOSSES = Registry("loss")
METRICS = Registry("metric")
OPTIMIZERS = Registry("optimizer")
ADAPTERS = Registry("adapter")
PROCESSORS = Registry("processor")        # data.params.features.kind
AUGMENTATIONS = Registry("augmentation")  # 피처 공간 transform
META_LABELS = Registry("meta_label")      # data.params.meta_label_mode

# ── datamodule (구 trainer/registries/data_registry.py 의 수동 dict) ──────────────────────
DATAMODULES.register_lazy("index", "rsc_toolkit.trainer.data.index_datamodule:IndexDataModule")
# 저자 배포 training.pt/test.pt (BTS·Audio-CLAP·BTS-CARD)
DATAMODULES.register_lazy("bts_precomputed", "rsc_toolkit.datasets.bts_precomputed:BTSPrecomputedDataModule")
# RespireNet 8초 청킹 + magma 이미지 (+ 선택 CBA) (microsoft/RespireNet)
DATAMODULES.register_lazy("respirenet", "rsc_toolkit.datasets.respirenet_chunks:RespireNetDataModule")
# PC-MCL 동적 연결·PM 쌍 학습셋 + IndexDataset 평가 (wa976/PC-MCL)
DATAMODULES.register_lazy("pcmcl", "rsc_toolkit.datasets.pcmcl_concat:PCMCLDataModule")
# AMS-D raw fbank 캐시 + 샘플별 torchaudio 마스킹/정규화 + 샘플 인덱스 meta (CcnNnn/AMS-D)
DATAMODULES.register_lazy("amsd", "rsc_toolkit.datasets.amsd_data:AMSDDataModule")
# GaP-Aug 클래스 균형 배치 샘플러(4/클래스, 500 step epoch) — 논문 명세 재구현
DATAMODULES.register_lazy("gapaug", "rsc_toolkit.datasets.gapaug_data:GaPAugDataModule")
# Lungmix 밴드패스 + SPRSound 7→4 파생 라벨 (Tigerdwgth/ICASSP-2025-Lungmix, 논문 명세 재구현)
DATAMODULES.register_lazy("lungmix", "rsc_toolkit.datasets.lungmix_data:LungmixDataModule")
# Prototype(ICASSP 2022) 4 kHz 밴드패스·4 s 랜덤/센터 크롭 + balanced class weight
DATAMODULES.register_lazy("protonet", "rsc_toolkit.datasets.protonet_data:ProtoNetDataModule")
# Domain Transfer(ICASSP 2022) 8 kHz·circular padding·splice·파형 증강
DATAMODULES.register_lazy("domaintransfer", "rsc_toolkit.datasets.domaintransfer_data:DomainTransferDataModule")
# QLung AQS float 를 배치 3번째 원소로 (RSC-Toolkit/QLung 46408dd)
DATAMODULES.register_lazy("qlung", "rsc_toolkit.datasets.qlung_data:QLungDataModule")
# BTS 계열 SPR→SPR 학습: meta7 모양 벡터를 train·val 양쪽 배치에 붙인다 (index 경로는 val 에 meta 를 안 붙인다)
DATAMODULES.register_lazy("btscard_sprs", "rsc_toolkit.datasets.btscard_sprs:BTSCARDSprsDataModule")

# ── loss / metric / optimizer (구 trainer/registries/{loss,metric,optimizer}_registry.py) ──
LOSSES.register_lazy("cross_entropy", "torch.nn:CrossEntropyLoss")
LOSSES.register_lazy("cross_entropy_from_counts", "rsc_toolkit.trainer.losses:WeightedCrossEntropyFromCounts")
LOSSES.register_lazy("cross_entropy_from_prior", "rsc_toolkit.trainer.losses:WeightedCrossEntropyFromPrior")
LOSSES.register_lazy("bce_with_logits", "torch.nn:BCEWithLogitsLoss")  # PC-MCL 3-label 주 손실 (wa976/PC-MCL main.py:421)

METRICS.register_lazy("topk_accuracy", "rsc_toolkit.metrics.topk_accuracy:TopKAccuracy")
METRICS.register_lazy("icbhi_score", "rsc_toolkit.metrics.icbhi.score:ICBHIScore")
# AMS-D 논문 이진(비표준) Se/Sp/Score (CcnNnn/AMS-D utils/metric.py)
METRICS.register_lazy("icbhi_paper_binary", "rsc_toolkit.metrics.icbhi.binary_score:PaperBinaryScore")

OPTIMIZERS.register_lazy("adam", "torch.optim:Adam")
OPTIMIZERS.register_lazy("adamw", "torch.optim:AdamW")
OPTIMIZERS.register_lazy("sgd", "torch.optim:SGD")

# ── adapter (구 train.py 의 if 문) ─────────────────────────────────────────────────────
ADAPTERS.register_lazy("pl", "rsc_toolkit.trainer.adapters.pl_adapter:PLAdapter")
# M2D 미세조정 quirk: 배치를 K 분할해 forward·backward 누적 (nttcslab/m2d ce.py)
ADAPTERS.register_lazy("split_iter", "rsc_toolkit.trainer.adapters.split_iter_adapter:SplitIterAdapter")

# ── processor (구 IndexDataModule._build_feature_processor 의 if-elif) — features.kind ─────────
# 조회된 클래스의 from_options(opts, sampling_rate=, feature_cache=) 로 만든다. get_processor_config()
# (= 캐시 키)는 생성 경로와 무관하므로 기존 캐시가 그대로 적중한다.
PROCESSORS.register_lazy("clap", "rsc_toolkit.processors.clap:CLAPProcessor")
PROCESSORS.register_lazy("mel", "rsc_toolkit.processors.mel:MelBankAudioProcessor")
PROCESSORS.register_lazy("fbank", "rsc_toolkit.processors.mel:MelBankAudioProcessor")  # 별칭 (AST 계열 표기)
PROCESSORS.register_lazy("sclmeta_mel", "rsc_toolkit.processors.sclmeta_mel:SCLMetaMelProcessor")
PROCESSORS.register_lazy("waveform", "rsc_toolkit.processors.waveform:WaveformProcessor")  # 캐시 없음

# ── augmentation (피처 공간 transform; 구 IndexDataModule._build_feature_transform 하드코딩) ──
AUGMENTATIONS.register_lazy("specaugment", "rsc_toolkit.augmentations.specaugment:SpecAugment")

# ── meta label (data.params.meta_label_mode; 구 datasets/meta_labels.build_meta_label_fn 의 if 체인) ──
# 등록 객체는 팩토리: 인자 없이 불러 `item -> int` 함수를 돌려준다.
META_LABELS.register_lazy("dev", "rsc_toolkit.datasets.meta_labels:build_dev")          # ICBHI 청진기 4종 (sgscl 계열)
META_LABELS.register_lazy("patient", "rsc_toolkit.datasets.meta_labels:build_patient")  # 환자 번호 (PAFA)
META_LABELS.register_lazy("sa", "rsc_toolkit.datasets.meta_labels:build_sa")            # 성별×연령군 (SCL-metadata ICBHI)
META_LABELS.register_lazy("sa_sprs", "rsc_toolkit.datasets.meta_labels:build_sa_sprs")  # gender×age_group (SCL-metadata SPRS)

# ── dataset (raw 데이터셋: 다운로드·인덱스·CLI; 구 DatasetFactory 의 '_dataset' 파일명 스캔) ────────
DATASETS.register_lazy("icbhi", "rsc_toolkit.datasets.icbhi.icbhi_dataset:ICBHIDataset")
DATASETS.register_lazy("sprsound", "rsc_toolkit.datasets.sprsound.sprsound_dataset:SPRSoundDataset")
DATASETS.register_lazy("hflungv1", "rsc_toolkit.datasets.hflungv1.hflungv1_dataset:HFLungV1Dataset")
DATASETS.register_lazy("toy", "rsc_toolkit.datasets.toy_dataset:ToyDataset")  # 번들 fixture (sample_data)

ALL_REGISTRIES = {
    r.kind: r
    for r in (MODELS, DATASETS, DATAMODULES, LOSSES, METRICS, OPTIMIZERS, ADAPTERS, PROCESSORS, AUGMENTATIONS, META_LABELS)
}

__all__ = [
    "MODELS", "DATASETS", "DATAMODULES", "LOSSES", "METRICS", "OPTIMIZERS",
    "ADAPTERS", "PROCESSORS", "AUGMENTATIONS", "META_LABELS", "ALL_REGISTRIES",
]
