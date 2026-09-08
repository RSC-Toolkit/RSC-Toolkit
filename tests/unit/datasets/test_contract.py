"""P2-7 — 배치 계약 문서 모듈: META_KINDS 표가 트리의 선언과 어긋나지 않는지, Batch 뷰."""
from __future__ import annotations

import pytest
import torch

from rsc_toolkit.datasets import contract
from rsc_toolkit.registries import MODELS


def test_meta_kinds_table_matches_datamodule_and_model_declarations():
    from rsc_toolkit.datasets.amsd_data import AMSDDataModule
    from rsc_toolkit.datasets.bts_precomputed import BTSPrecomputedDataModule
    from rsc_toolkit.datasets.btscard_sprs import BTSCARDSprsDataModule
    from rsc_toolkit.datasets.lungmix_data import LungmixDataModule
    from rsc_toolkit.datasets.pcmcl_concat import PCMCLDataModule
    from rsc_toolkit.datasets.qlung_data import QLungDataModule
    from rsc_toolkit.datasets.respirenet_chunks import RespireNetDataModule

    declared = {QLungDataModule.meta_kind, PCMCLDataModule.meta_kind, BTSPrecomputedDataModule.meta_kind,
                BTSCARDSprsDataModule.meta_kind, AMSDDataModule.meta_kind, RespireNetDataModule.meta_kind,
                LungmixDataModule.meta_kind}
    assert declared <= set(contract.META_KINDS)
    # 표의 receivers 가 실제로 그 kind 를 accepts_meta 에 선언했는가 (SFT 4종 표기는 묶음이라 건너뛴다)
    for kind, row in contract.META_KINDS.items():
        for name in row["receivers"]:
            if "/" in name:
                continue
            assert kind in MODELS.get(name).accepts_meta, (kind, name)


def test_batch_view_and_describe():
    b = contract.Batch.from_tuple((torch.zeros(2, 1, 4), torch.tensor([0, 1])))
    assert b.meta is None
    d = contract.describe_batch((torch.zeros(2, 1, 4), torch.tensor([0, 1]), torch.tensor([0.5, 0.25], dtype=torch.float64)))
    assert d["x"] == ((2, 1, 4), "torch.float32") and d["meta"] == ((2,), "torch.float64")
    with pytest.raises(ValueError):
        contract.Batch.from_tuple((torch.zeros(1),))
    assert contract.SAMPLE_LEVELS == ("cycle", "clip")
