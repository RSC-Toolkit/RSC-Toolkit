"""
split_iter(K) 학습 어댑터 — nttcslab/m2d app/icbhi_sprs 패치본 ce.py 의 저메모리 분할 배치 동작 재현.

train: 배치 전체를 model.train_transform (배치 min-max·SpecAug 가 B 기준) → D = B // K 개씩 K 번
       forward → 손실(부분배치 평균) backward 누적 → optimizer.step 1회. 나머지 B - K·D 개는 버린다(원본 quirk; 학습 레시피의 일부라 유지).
val:   D = B // K 개씩 model.val_transform (min-max 가 D 기준) → forward. 원본 ce.py 는 나머지 B - K·D 개를 평가에서 빠뜨렸지만
       (SPRSound test 1,429 = 64×22 + 21 → 1 건 탈락) 툴킷은 나머지를 (K+1) 번째 서브배치로 평가한다 — 2026-09-16 사용자 지시
       ("테스트 데이터 누락은 말이 안 된다"). 앞 K·D 건의 서브배치 구성·min-max 는 원본과 같아 예측이 비트 동일하고 나머지 건만 더해진다.
스케줄러는 epoch 끝에 수동 step (원본 scheduler.step() 위치와 동일).
"""
from typing import Any, Dict, List, Optional

import torch

from .pl_adapter import PLAdapter


class SplitIterAdapter(PLAdapter):
    # BaseRSCModel 계약(training_forward) 밖의 M2D 계열 전용 프로토콜. 생성 시점에 검사해
    # 다른 모델과 잘못 조합했을 때 학습 도중 AttributeError 로 발견되는 일을 막는다.
    REQUIRED_MODEL_ATTRS = ("train_transform", "val_transform")

    def __init__(self, *args, rsc_model=None, split_iter: int = 4, **kwargs):
        if rsc_model is None and args:
            rsc_model, args = args[0], args[1:]
        missing = [a for a in self.REQUIRED_MODEL_ATTRS if not callable(getattr(rsc_model, a, None))]
        if missing:
            raise TypeError(
                f"adapter=split_iter 는 모델에 {missing} 메서드를 요구한다 (M2D 계열 전용 프로토콜). "
                f"{type(rsc_model).__name__} 은 BaseRSCModel 계약(training_forward)만 구현했다 — "
                "adapter=pl 을 쓰거나 해당 메서드를 구현하라."
            )
        if int(split_iter) < 1:
            raise ValueError(f"split_iter 는 1 이상이어야 한다: {split_iter!r}")
        super().__init__(*args, rsc_model=rsc_model, **kwargs)
        self.K = int(split_iter)
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        x, y = batch[0].squeeze(1), batch[1]
        opt = self.optimizers()
        opt.zero_grad()
        x_t = self.rsc_model.train_transform(x)
        D = x_t.shape[0] // self.K
        total = 0.0
        logits_all, y_all = [], []
        for i in range(self.K):
            xs, ys = x_t[i * D:(i + 1) * D], y[i * D:(i + 1) * D]
            logits = self.rsc_model(xs)
            loss = self.loss_func(logits, ys)
            self.manual_backward(loss)
            total = total + loss.detach()
            logits_all.append(logits.detach()); y_all.append(ys)
        opt.step()
        self.log("train_loss", total, prog_bar=True, on_step=True, on_epoch=True)
        for metric in self.train_metrics:
            metric.update(torch.cat(logits_all), torch.cat(y_all))
        return total

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

    def val_chunks(self, B: int) -> list:
        """검증 서브배치 경계 [(start, end), …]: D=B//K 개씩 K 개 + 나머지(있으면) 1 개. B < K 면 통째로 1 개. 모든 샘플이 정확히 한 번 평가된다."""
        D = B // self.K
        bounds = [(i * D, (i + 1) * D) for i in range(self.K)] if D > 0 else []
        if self.K * D < B:
            bounds.append((self.K * D, B))
        return bounds

    def validation_step(self, batch, batch_idx):
        x, y = batch[0].squeeze(1), batch[1]
        losses, logits_all, y_all = [], [], []
        for s, e in self.val_chunks(int(x.shape[0])):
            xs, ys = x[s:e], y[s:e]
            logits = self.rsc_model(self.rsc_model.val_transform(xs))
            losses.append(self.loss_func(logits, ys))
            logits_all.append(logits); y_all.append(ys)
        self.log("val_loss", torch.stack(losses).mean(), prog_bar=True, on_step=False, on_epoch=True)
        for metric in self.val_metrics:
            metric.update(torch.cat(logits_all), torch.cat(y_all))
