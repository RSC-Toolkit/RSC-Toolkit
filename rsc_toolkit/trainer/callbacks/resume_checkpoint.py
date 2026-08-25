"""
epoch 단위 재개(resume)용 체크포인트 — 학습 프로세스가 외부 요인(Slurm 잡 종료 등)으로
죽어도 마지막 완료 epoch 부터 이어갈 수 있게 한다.

매 epoch 끝(그 epoch 의 검증·best 갱신까지 끝난 뒤)에 Lightning 전체 상태
(모델·옵티마이저·AMP scaler·loop 진행·콜백 상태)를 `last.ckpt` 로 저장한다.
train.py 는 같은 run 을 다시 띄울 때 이 파일이 있으면 자동으로 이어간다 (resume: auto).
정상 완료되면 파일을 지운다 (산출물은 best.pth 뿐이어야 한다).

논문 프로토콜(수치)에는 영향이 없다: 저장만 추가되며, 재개 시 달라지는 것은
난수 스트림(SpecAugment 마스크·셔플 순서)뿐이다 — Lightning 은 전역 RNG 를 복원하지
않는다. 중단 없는 실행과 비트 단위로 같지는 않지만 같은 분포의 학습이다.
"""
from __future__ import annotations

import logging
import os
import pathlib

import lightning as L

logger = logging.getLogger(__name__)


class EpochResumeCheckpoint(L.Callback):
    def __init__(self, dirpath: str, filename: str = "last.ckpt", delete_on_finish: bool = True):
        super().__init__()
        self.dirpath = pathlib.Path(dirpath)
        self.filename = filename
        self.delete_on_finish = bool(delete_on_finish)

    @property
    def last_path(self) -> pathlib.Path:
        return self.dirpath / self.filename

    def on_train_epoch_end(self, trainer, pl_module):
        # 검증(on_validation_end → best 갱신)은 이 훅보다 먼저 끝난다.
        if trainer.sanity_checking:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        tmp = self.last_path.with_name(self.last_path.name + ".tmp")
        trainer.save_checkpoint(str(tmp))
        os.replace(tmp, self.last_path)  # 저장 도중 죽어도 이전 last.ckpt 가 온전히 남는다

    def on_fit_end(self, trainer, pl_module):
        # trainer.state.finished 는 이 훅 *뒤에* 세팅되므로 loop 완료 여부로 판단한다
        # (max_epochs 도달 또는 should_stop). 예외/중단 시엔 이 훅이 불리지 않는다.
        if self.delete_on_finish and trainer.fit_loop.done and self.last_path.exists():
            self.last_path.unlink()
            logger.info("학습 정상 종료 — 재개용 %s 삭제", self.last_path)


def find_resume_path(trainer: L.Trainer) -> pathlib.Path | None:
    """trainer 에 EpochResumeCheckpoint 가 있고 그 last.ckpt 가 존재하면 경로를 돌려준다."""
    for cb in trainer.callbacks:
        if isinstance(cb, EpochResumeCheckpoint) and cb.last_path.exists():
            return cb.last_path
    return None
