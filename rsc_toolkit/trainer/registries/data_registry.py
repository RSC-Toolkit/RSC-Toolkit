from typing import Dict, Type

import lightning as L

from ..data.index_datamodule import IndexDataModule

DATAMODULES: Dict[str, Type[L.LightningDataModule]] = {
    "index": IndexDataModule,
}
