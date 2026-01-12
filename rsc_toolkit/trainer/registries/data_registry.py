from typing import Dict, Type

import lightning as L

from ..data.icbhi_index_datamodule import ICBHIIndexDataModule

DATAMODULES: Dict[str, Type[L.LightningDataModule]] = {
    "icbhi_index": ICBHIIndexDataModule,
}
