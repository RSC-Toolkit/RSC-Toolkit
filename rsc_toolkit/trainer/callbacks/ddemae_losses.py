"""Logs the DDE-MAE stage-1 loss breakdown (recon / siamese / mi): records the last_losses left by the model's training_forward per step and per epoch."""
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
