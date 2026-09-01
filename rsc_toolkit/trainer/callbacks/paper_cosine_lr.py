"""
patch-mix 계열 논문의 epoch 단위 cosine LR — util/misc.py adjust_learning_rate 재현.

원본은 스케줄러 객체 없이 매 epoch 시작 때 lr 을 직접 계산해 덮어쓴다:
  eta_min = lr * lr_decay_rate**3
  lr(e)   = eta_min + (lr - eta_min) * (1 + cos(pi * e / epochs)) / 2,  e = 1..epochs
같은 방식(콜백에서 직접 설정)으로 재현한다. warmup 은 AST 스크립트에서 쓰지 않으므로
구현하지 않았다 (--warm 미사용).
"""
import logging
import math

import lightning as L

logger = logging.getLogger(__name__)


class PaperCosineLR(L.Callback):
    def __init__(self, learning_rate: float, epochs: int, lr_decay_rate: float = 0.1):
        super().__init__()
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.lr_decay_rate = float(lr_decay_rate)

    def _check_single_source(self, trainer) -> None:
        """
        lr·epochs 가 optimizer/trainer 그룹과 어긋나면 경고한다.

        이 콜백은 매 epoch lr 을 직접 덮어쓰므로, 값이 어긋나면 CLI 로 준
        optimizer.params.lr 스윕이 조용히 무효가 되고 결과가 엉뚱한 하이퍼파라미터에
        귀속된다 (callbacks yaml 은 ${optimizer.params.lr} 보간이 표준).
        fresh run(epoch 0)에서만 검사한다 — 재개 시 optimizer 의 lr 은 스케줄
        중간값이라 비교 대상이 아니다.
        """
        base_lr = float(trainer.optimizers[0].param_groups[0]["lr"])
        if not math.isclose(base_lr, self.learning_rate, rel_tol=1e-9):
            logger.warning(
                "PaperCosineLR.learning_rate=%.3g 가 optimizer 의 lr=%.3g 와 다르다 — "
                "이 콜백이 매 epoch lr 을 덮어쓰므로 실효 lr 은 %.3g 다. "
                "callbacks yaml 의 ${optimizer.params.lr} 보간이 깨졌는지 확인하라.",
                self.learning_rate, base_lr, self.learning_rate,
            )
        max_epochs = getattr(trainer, "max_epochs", None)  # 테스트 스텁 trainer 에는 없을 수 있다
        if max_epochs is not None and self.epochs != int(max_epochs):
            logger.warning(
                "PaperCosineLR.epochs=%d 가 trainer.max_epochs=%d 와 다르다 — "
                "epochs 를 넘는 구간에서 cos 이 되돌아 lr 이 다시 상승한다.",
                self.epochs, int(max_epochs),
            )

    def lr_at(self, epoch_1based: int) -> float:
        lr = self.learning_rate
        eta_min = lr * (self.lr_decay_rate ** 3)
        return eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch_1based / self.epochs)) / 2

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == 0:
            self._check_single_source(trainer)
        new_lr = self.lr_at(trainer.current_epoch + 1)  # 원본 epoch 는 1부터
        for optimizer in trainer.optimizers:
            for group in optimizer.param_groups:
                group["lr"] = new_lr
