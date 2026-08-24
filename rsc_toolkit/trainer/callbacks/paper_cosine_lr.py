"""
patch-mix 계열 논문의 epoch 단위 cosine LR — util/misc.py adjust_learning_rate 재현.

원본은 스케줄러 객체 없이 매 epoch 시작 때 lr 을 직접 계산해 덮어쓴다:
  eta_min = lr * lr_decay_rate**3
  lr(e)   = eta_min + (lr - eta_min) * (1 + cos(pi * e / epochs)) / 2,  e = 1..epochs
같은 방식(콜백에서 직접 설정)으로 재현한다. warmup 은 AST 스크립트에서 쓰지 않으므로
구현하지 않았다 (--warm 미사용).
"""
import math

import lightning as L


class PaperCosineLR(L.Callback):
    def __init__(self, learning_rate: float, epochs: int, lr_decay_rate: float = 0.1):
        super().__init__()
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.lr_decay_rate = float(lr_decay_rate)

    def lr_at(self, epoch_1based: int) -> float:
        lr = self.learning_rate
        eta_min = lr * (self.lr_decay_rate ** 3)
        return eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch_1based / self.epochs)) / 2

    def on_train_epoch_start(self, trainer, pl_module):
        new_lr = self.lr_at(trainer.current_epoch + 1)  # 원본 epoch 는 1부터
        for optimizer in trainer.optimizers:
            for group in optimizer.param_groups:
                group["lr"] = new_lr
