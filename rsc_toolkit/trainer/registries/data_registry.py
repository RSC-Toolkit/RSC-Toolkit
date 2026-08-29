from typing import Dict, Type

import lightning as L

from ..data.index_datamodule import IndexDataModule
from rsc_toolkit.datasets.bts_precomputed import BTSPrecomputedDataModule
from rsc_toolkit.datasets.respirenet_chunks import RespireNetDataModule
from rsc_toolkit.datasets.pcmcl_concat import PCMCLDataModule
from rsc_toolkit.datasets.amsd_data import AMSDDataModule
from rsc_toolkit.datasets.gapaug_data import GaPAugDataModule
from rsc_toolkit.datasets.lungmix_data import LungmixDataModule
from rsc_toolkit.datasets.protonet_data import ProtoNetDataModule
from rsc_toolkit.datasets.qlung_data import QLungDataModule

DATAMODULES: Dict[str, Type[L.LightningDataModule]] = {
    "index": IndexDataModule,
    "bts_precomputed": BTSPrecomputedDataModule,  # 저자 배포 training.pt/test.pt (BTS·Audio-CLAP·BTS-CARD)
    "respirenet": RespireNetDataModule,           # RespireNet 8초 청킹 + magma 이미지 (microsoft/RespireNet)
    "pcmcl": PCMCLDataModule,                     # PC-MCL 동적 연결·PM 쌍 학습셋 + IndexDataset 평가 (wa976/PC-MCL)
    "amsd": AMSDDataModule,                       # AMS-D raw fbank 캐시 + 샘플별 torchaudio 마스킹/정규화 + 샘플 인덱스 meta (CcnNnn/AMS-D)
    "gapaug": GaPAugDataModule,                   # GaP-Aug 클래스 균형 배치 샘플러(4/클래스, 500 step epoch) — 논문 명세 재구현
    "lungmix": LungmixDataModule,
    "protonet": ProtoNetDataModule,               # Prototype(ICASSP 2022) 4 kHz 밴드패스·4 s 랜덤/센터 크롭 + balanced class weight                 # Lungmix 밴드패스·SPRSound 7→4 파생 라벨 (Tigerdwgth/ICASSP-2025-Lungmix, 논문 명세 재구현)
    "qlung": QLungDataModule,                     # QLung AQS float 를 배치 3번째 원소로 (RSC-Toolkit/QLung 46408dd)
}
