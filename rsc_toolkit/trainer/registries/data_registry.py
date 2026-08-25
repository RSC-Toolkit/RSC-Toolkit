from typing import Dict, Type

import lightning as L

from ..data.index_datamodule import IndexDataModule
from rsc_toolkit.datasets.bts_precomputed import BTSPrecomputedDataModule
from rsc_toolkit.datasets.respirenet_chunks import RespireNetDataModule

DATAMODULES: Dict[str, Type[L.LightningDataModule]] = {
    "index": IndexDataModule,
    "bts_precomputed": BTSPrecomputedDataModule,  # 저자 배포 training.pt/test.pt (BTS·Audio-CLAP·BTS-CARD)
    "respirenet": RespireNetDataModule,           # RespireNet 8초 청킹 + magma 이미지 (microsoft/RespireNet)
}
