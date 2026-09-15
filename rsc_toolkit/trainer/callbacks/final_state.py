"""학습 종료 시 모델 state_dict 저장 — SCL/M-SCL 사전학습처럼 평가 없이 마지막 epoch 상태를 넘기는 프로토콜용."""
import logging
import pathlib

import lightning as L

from rsc_toolkit.trainer.callbacks.atomic_io import atomic_torch_save

logger = logging.getLogger(__name__)


class FinalStateCheckpoint(L.Callback):
    def __init__(self, dirpath: str, filename: str = "final_state.pth", model_attr: str = "rsc_model"):
        super().__init__()
        self.dirpath, self.filename, self.model_attr = pathlib.Path(dirpath), filename, model_attr

    def on_train_end(self, trainer, pl_module):
        model = getattr(pl_module, self.model_attr, pl_module)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        atomic_torch_save({"model": model.state_dict(), "epoch": trainer.current_epoch}, self.dirpath / self.filename)
        logger.info("최종 상태 저장: %s (epoch %d)", self.dirpath / self.filename, trainer.current_epoch)
