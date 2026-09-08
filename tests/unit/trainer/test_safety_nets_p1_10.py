"""리팩토링 P1 10단계 — 잔여 안전망.

(a) PLAdapter 의 학습 메트릭 키는 `train/` 접두사를 붙인다 — 검증 키는 그대로(best-checkpoint·검증 스크립트·
    best.json 스키마 불변). 예전엔 train/val 의 같은 키(acc@1)가 callback_metrics 와 TensorBoard 한 시리즈에 섞였고,
    train 목록에 icbhi_score 를 넣으면 best 선정이 train 값으로 오염될 수 있었다.
(b) FeatureCache.configure 는 audio_chain 없이 부르면 경고한다 (다음 릴리스에 필수) — 체인 대장 없는 캐시는
    다른 전처리의 피처를 조용히 읽을 수 있다. SPRSound OOD 경로도 이제 체인을 남긴다.
"""
from __future__ import annotations

import json
import logging

import pytest
import torch
from torch import nn

from rsc_toolkit.datasets.feature_cache import FeatureCache
from rsc_toolkit.trainer.adapters.pl_adapter import PLAdapter
from rsc_toolkit.trainer.factories.build_metrics import build_train_val_metric_factories


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 3)

    def training_forward(self, audio_batch, labels, loss_func, **kwargs):
        logits = self.head(audio_batch)
        return logits, loss_func(logits, labels)


def _adapter():
    train_fns, val_fns = build_train_val_metric_factories({
        "train": [{"name": "topk_accuracy", "params": {"topk": [1]}}],
        "val": [{"name": "topk_accuracy", "params": {"topk": [1]}},
                {"name": "icbhi_score", "params": {"num_classes": 3, "normal_index": 0, "topk": 1}}],
    })
    return PLAdapter(rsc_model=_Model(), loss_func=nn.CrossEntropyLoss(),
                     optimizer_cfg={"name": "sgd", "params": {"lr": 0.1}},
                     train_metric_fns=train_fns, val_metric_fns=val_fns)


def test_train_metric_keys_are_prefixed_and_val_keys_are_not(monkeypatch):
    adapter = _adapter()
    logged = {}
    monkeypatch.setattr(adapter, "log_dict", lambda d, **kw: logged.update(d))
    logits, labels = torch.randn(6, 3), torch.tensor([0, 1, 2, 0, 1, 2])
    for m in adapter.train_metrics:
        m.update(logits, labels)
    for m in adapter.val_metrics:
        m.update(logits, labels)
    adapter.on_train_epoch_end()
    adapter.on_validation_epoch_end()
    assert "train/acc@1" in logged
    assert "acc@1" in logged and "icbhi_score" in logged and "S_p" in logged and "S_e" in logged
    assert not any(k.startswith("val/") for k in logged)  # 검증 키는 무접두사 유지 (콜백·스크립트 호환)


def test_feature_cache_configure_warns_without_audio_chain(tmp_path, caplog):
    cache = FeatureCache(cache_dir=tmp_path)
    with caplog.at_level(logging.WARNING, logger="rsc_toolkit.datasets.feature_cache"):
        cache.configure({"n_mels": 64}, processor_name="Mel")
    assert any("audio_chain" in r.getMessage() for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="rsc_toolkit.datasets.feature_cache"):
        cache.configure({"n_mels": 64}, processor_name="Mel", audio_chain={"dataset_name": "d", "sampling_rate": 16000})
    assert not any("audio_chain" in r.getMessage() for r in caplog.records)
    assert json.loads((cache.features_dir / "_chain.json").read_text()) == {"d": {"sampling_rate": 16000}}


def test_sprsound_ood_feature_dataset_records_its_audio_chain(tmp_path, monkeypatch):
    import rsc_toolkit.datasets.sprsound_ood as ood

    monkeypatch.setattr(ood, "_load_clap_processor", lambda model_name, sampling_rate: None, raising=False)
    from transformers import ClapProcessor

    monkeypatch.setattr(ClapProcessor, "from_pretrained", staticmethod(lambda *a, **k: None))
    chain = {"dataset_name": "sprsound_ood", "sampling_rate": 48000, "sample_level": "cycle",
             "desired_length_sec": 8.0, "pad_type": "repeat_fade", "fade": True, "time_unit": "ood_ms_round"}
    ds = ood._CLAPFeatureDataset(index_ds=[], sampling_rate=48000, model_name="laion/clap-htsat-unfused",
                                 cache_dir=tmp_path, audio_chain=dict(chain))
    ledger = json.loads((ds.cache.features_dir / "_chain.json").read_text())
    assert ledger == {"sprsound_ood": {k: v for k, v in chain.items() if k != "dataset_name"}}
