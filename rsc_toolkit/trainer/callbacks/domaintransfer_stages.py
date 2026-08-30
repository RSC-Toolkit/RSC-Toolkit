"""
Domain Transfer(Wang & Wang, ICASSP 2022) 의 **2단계 학습**과 warm-up·클래스 가중 주입.

논문이 명시한 것(§3.2):
- "Adam optimizer with learning rate 6e-4, batch size 64, weight decay 1e-3, and **warm-up**"
- 증강 데이터로 학습한 뒤 **원본 데이터만으로 작은 learning rate 로 다시 미세조정**하는 단계가 따로 있다.

가정(논문 미기재 — 전부 인자로 노출, `docs/REPRODUCTION_TARGETS.md` 신규 24 의 공백 (3)(4)(7)):
- **epoch 수 미기재** → 1단계 50 epoch, 2단계 10 epoch.
- **warm-up 스케줄 미기재** → 1단계 앞 5 epoch 을 0 → lr 로 선형 증가시킨 뒤 cosine 으로 감쇠.
- **2단계 lr 미기재** → 1단계 lr 의 0.1배(6e-5) 고정.

2단계에서 끄는 것(= "원본 데이터"의 해석):
- splice 로 만든 추가 샘플 제거(`extra = []`)
- 파형 증강 3종(잡음·time stretch·pitch shift) 해제
- mixup 해제(`model.mixup_alpha = 0`)
- circular padding 은 **끄지 않는다** — 논문이 test set 에도 적용한다고 못박은 전처리이지 증강이 아니다.

데이터셋 길이가 2단계에서 줄어들므로 `trainer.params.reload_dataloaders_every_n_epochs: 1` 이 필요하다
(config 에 함께 넣어 두었다).
"""
from __future__ import annotations

import logging
import math

import lightning as L

logger = logging.getLogger(__name__)


class DomainTransferStages(L.Callback):
    def __init__(
        self,
        learning_rate: float = 6.0e-4,
        stage1_epochs: int = 50,
        warmup_epochs: int = 5,
        stage2_lr_scale: float = 0.1,
        eta_min: float = 0.0,
    ):
        super().__init__()
        self.lr = float(learning_rate)
        self.stage1_epochs = int(stage1_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.stage2_lr = self.lr * float(stage2_lr_scale)
        self.eta_min = float(eta_min)
        self._switched = False

    # ------------------------------------------------------------------ 가중치 주입
    def setup(self, trainer, pl_module, stage=None):
        dm = getattr(trainer, "datamodule", None)
        w = getattr(dm, "train_class_weight", None)
        model = getattr(pl_module, "rsc_model", pl_module)
        if w is None or not hasattr(model, "set_class_weight"):
            logger.warning("DomainTransferStages: class weight 주입 생략(datamodule 또는 모델에 없음)")
            return
        model.set_class_weight(w)
        logger.info("DomainTransferStages: class weight 주입 %s", [round(float(x), 4) for x in w])

    # ------------------------------------------------------------------ lr
    def lr_at(self, epoch: int) -> float:
        """epoch 는 0-based. 1단계는 warm-up→cosine, 2단계는 낮은 고정 lr."""
        if epoch >= self.stage1_epochs:
            return self.stage2_lr
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            return self.lr * float(epoch + 1) / float(self.warmup_epochs)
        span = max(1, self.stage1_epochs - self.warmup_epochs)
        prog = (epoch - self.warmup_epochs) / span
        return self.eta_min + (self.lr - self.eta_min) * (1 + math.cos(math.pi * prog)) / 2

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch >= self.stage1_epochs and not self._switched:
            self._enter_stage2(trainer, pl_module)
        new_lr = self.lr_at(epoch)
        for optimizer in trainer.optimizers:
            for group in optimizer.param_groups:
                group["lr"] = new_lr

    # ------------------------------------------------------------------ 2단계 전환
    def _enter_stage2(self, trainer, pl_module) -> None:
        model = getattr(pl_module, "rsc_model", pl_module)
        if hasattr(model, "mixup_alpha"):
            model.mixup_alpha = 0.0
        dm = getattr(trainer, "datamodule", None)
        ds = getattr(dm, "train_ds", None)
        base = getattr(ds, "index_ds", ds)
        n_before = len(base) if base is not None else -1
        if base is not None:
            base.extra = []
            for flag in ("use_noise", "use_stretch", "use_pitch"):
                if hasattr(base, flag):
                    setattr(base, flag, False)
        self._switched = True
        logger.info(
            "Domain Transfer 2단계 진입(epoch %d): mixup·파형증강·splice 해제, lr %.2e — 학습 샘플 %d → %d",
            trainer.current_epoch, self.stage2_lr, n_before, len(base) if base is not None else -1,
        )
