from typing import Callable, List, Optional, Any, Dict

import lightning as L
import torch.nn as nn
from torchmetrics import Metric

from ..factories.build_optimizer import build_optimizer


MetricFactory = Callable[[], Metric]


def check_meta_contract(rsc_model: Any, datamodule: Any) -> None:
    """
    batch[2](meta) 사이드채널의 조합 검사 — 학습 시작 전에 1회.

    datamodule 이 `meta_kind`(무엇을 보내는지) 를, 모델이 `accepts_meta`(무엇을 받는지) 를 선언한
    경우에만 대조한다. 한쪽이라도 선언하지 않았으면 예전과 똑같이 통과한다(opt-in). 어긋나면
    TypeError — AQS float 를 청진기 id 로 해석하는 식으로 **조용히 잘못 학습되는 조합**을 차단한다.
    """
    if datamodule is None:
        return
    kind = getattr(datamodule, "meta_kind", None)
    accepts = getattr(rsc_model, "accepts_meta", None)
    if kind is None or accepts is None:
        return
    accepts = (accepts,) if isinstance(accepts, str) else tuple(accepts)
    if kind not in accepts:
        raise TypeError(
            f"datamodule {type(datamodule).__name__} 은 batch[2] 로 meta_kind={kind!r} 를 보내는데 "
            f"모델 {type(rsc_model).__name__} 의 accepts_meta={accepts} 에 없다 — 데이터/모델 조합 오류. "
            "(meta 를 쓰지 않는 모델이면 accepts_meta 선언을 빼거나 그 kind 를 추가하라.)"
        )


class PLAdapter(L.LightningModule):
    def __init__(
        self,
        rsc_model: nn.Module,
        loss_func: nn.Module,
        optimizer_cfg: Dict[str, Any],
        train_metric_fns: Optional[List[MetricFactory]] = None,
        val_metric_fns: Optional[List[MetricFactory]] = None,
    ):
        super().__init__()
        self.rsc_model = rsc_model
        self.loss_func = loss_func
        self.optimizer_cfg = optimizer_cfg

        # Must instantiate inside to avoid shared state and ensure ownership
        self.train_metrics = nn.ModuleList([fn() for fn in (train_metric_fns or [])])
        self.val_metrics = nn.ModuleList([fn() for fn in (val_metric_fns or [])])

    def setup(self, stage: Optional[str] = None) -> None:
        # trainer 에 datamodule 이 붙어 있으면 meta 사이드채널 조합을 1회 검사한다.
        # (trainer 미부착·dataloader 직접 전달 경로에서는 조용히 건너뛴다.)
        trainer = getattr(self, "_trainer", None)
        datamodule = getattr(trainer, "datamodule", None) if trainer is not None else None
        check_meta_contract(self.rsc_model, datamodule)

    def training_step(self, batch, batch_idx):
        audio_batch = batch[0].squeeze(1)
        labels = batch[1]

        extra = {}
        if len(batch) > 2:
            # datamodule 의 meta_label_mode 가 켜진 경우에만 존재 (sgscl 계열 DAT·SG-SCL).
            # 받는 모델의 training_forward 는 meta_labels 인자를 지원해야 한다.
            extra["meta_labels"] = batch[2]

        logits, loss = self.rsc_model.training_forward(
            audio_batch=audio_batch,
            labels=labels,
            loss_func=self.loss_func,
            **extra,
        )
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        for metric in self.train_metrics:
            metric.update(logits, labels)

        return loss

    def validation_step(self, batch, batch_idx):
        audio_batch = batch[0].squeeze(1)
        labels = batch[1]

        extra = {}
        if len(batch) > 2:
            # BTS 처럼 평가에도 meta 가 필요한 모델 (bts_precomputed datamodule 은 val 에도 붙인다).
            # index datamodule 은 val 배치에 meta 를 붙이지 않으므로 sgscl 계열은 여기 안 온다.
            extra["meta_labels"] = batch[2]

        logits, loss = self.rsc_model.training_forward(
            audio_batch=audio_batch,
            labels=labels,
            loss_func=self.loss_func,
            **extra,
        )

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        for metric in self.val_metrics:
            metric.update(logits, labels)

    def on_train_start(self):
        self.rsc_model.train()

    def on_train_epoch_end(self):
        # 학습 메트릭은 `train/` 접두사 — 검증 키(acc@1·icbhi_score·S_p·S_e)는 그대로 둔다.
        # 예전엔 같은 키가 callback_metrics·TensorBoard 한 시리즈에 섞였고, train 목록에 icbhi_score 를
        # 넣으면 best-checkpoint 콜백이 train 값을 읽을 수 있었다. 검증 키를 안 바꾸므로 best.json 스키마·
        # 검증 스크립트·기존 run 과의 호환은 그대로다.
        for metric in self.train_metrics:
            self.log_dict({f"train/{k}": v for k, v in metric.compute().items()}, prog_bar=True)
            metric.reset()

    def on_validation_epoch_end(self):
        for metric in self.val_metrics:
            self.log_dict(metric.compute(), prog_bar=True)
            metric.reset()

    def configure_optimizers(self):
        cfg = dict(self.optimizer_cfg)
        sched_cfg = cfg.pop("lr_scheduler", None)
        optimizer = build_optimizer(cfg, self.parameters())
        if not sched_cfg:
            return optimizer
        # epoch 단위 torch 스케줄러 (예: SCL-metadata 의 CosineAnnealingLR(T_max=40, eta_min=1e-6) —
        # 원본은 매 epoch 끝에 scheduler.step(); Lightning 도 interval="epoch" 면 같은 시점에 step 한다)
        import torch.optim.lr_scheduler as S

        name = sched_cfg["name"]
        params = dict(sched_cfg.get("params", {}))
        cls = {"cosine_annealing": S.CosineAnnealingLR, "step": S.StepLR, "multistep": S.MultiStepLR}[name]  # multistep: AMS-D main.py adjust_learning_rate
        # interval: 기본 "epoch". Prototype(ICASSP 2022)처럼 원본이 매 iteration scheduler.step() 하는 논문은 "step".
        interval = str(sched_cfg.get("interval", "epoch"))
        if interval not in {"epoch", "step"}:
            raise ValueError(f"lr_scheduler.interval must be epoch|step, got {interval!r}")
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": cls(optimizer, **params), "interval": interval}}
