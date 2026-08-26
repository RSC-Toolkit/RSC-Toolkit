"""DDE-MAE 1단계 손실 분해(recon / siamese / mi) 로깅 — 모델이 training_forward 에서 남긴 last_losses 를 step·epoch 단위로 기록한다."""
import lightning as L


class DDEMAELossLogger(L.Callback):
    def __init__(self, model_attr: str = "rsc_model", prefix: str = "stage1/"):
        super().__init__()
        self.model_attr, self.prefix = model_attr, prefix

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        model = getattr(pl_module, self.model_attr, pl_module)
        losses = getattr(model, "last_losses", None) or {}
        for k, v in losses.items():
            pl_module.log(f"{self.prefix}{k}", float(v), on_step=True, on_epoch=True, prog_bar=False)
