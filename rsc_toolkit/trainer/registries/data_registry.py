from typing import Dict, Type

import lightning as L

from ..data.index_datamodule import IndexDataModule
from rsc_toolkit.datasets.bts_precomputed import BTSPrecomputedDataModule
from rsc_toolkit.datasets.respirenet_chunks import RespireNetDataModule
from rsc_toolkit.datasets.pcmcl_concat import PCMCLDataModule

DATAMODULES: Dict[str, Type[L.LightningDataModule]] = {
    "index": IndexDataModule,
    "bts_precomputed": BTSPrecomputedDataModule,  # 저자 배포 training.pt/test.pt (BTS·Audio-CLAP·BTS-CARD)
    "respirenet": RespireNetDataModule,           # RespireNet 8초 청킹 + magma 이미지 (microsoft/RespireNet)
    "pcmcl": PCMCLDataModule,                     # PC-MCL 동적 연결·PM 쌍 학습셋 + IndexDataset 평가 (wa976/PC-MCL)
}
