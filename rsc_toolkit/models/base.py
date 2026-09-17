# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
The contract the toolkit requires of a model.

This file is the only thing a third party must follow to plug in their own model.
Inherit `BaseRSCModel` together with `nn.Module`, implement `training_forward`, and
register with `ModelFactory.register_model_class(...)`; the training/validation loop then works unchanged.

    from torch import nn
    from rsc_toolkit.models.base import BaseRSCModel
    from rsc_toolkit.models.factory import ModelFactory

    @ModelFactory.register_model_class("my_model")
    class MyModel(BaseRSCModel, nn.Module):
        def __init__(self, n_cls: int = 4, **params):
            super().__init__()
            self.net = ...          # n_cls is injected from model.params.n_cls in the configs

        def training_forward(self, audio_batch, labels, loss_func, **kwargs):
            logits = self.net(audio_batch)
            return logits, loss_func(logits, labels)

(See `factory.py` for registration paths — modules inside the package are scanned automatically;
external packages attach via the `rsc_toolkit.plugins` entry point (the legacy `rsc_toolkit.models` group is also read) or `RSC_TOOLKIT_PLUGINS`.)
"""
from abc import ABC, abstractmethod
import logging
from typing import Any, Tuple


logger = logging.getLogger(__name__)


class BaseRSCModel(ABC):
    """
    Respiratory sound classification model interface.

    The training loop (`rsc_toolkit.trainer.adapters.pl_adapter.PLAdapter`) calls
    **only `training_forward`** at every step. The toolkit never calls `forward`, so
    it may be defined freely as needed.
    """

    @abstractmethod
    def training_forward(
        self,
        audio_batch: Any,
        labels: Any,
        loss_func: Any,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """
        Return logits and loss together for one batch.

        The training step and the validation step call **the same method**. If the two must
        differ, branch on `self.training` (managed by `nn.Module`) — models with terms that are
        active only during training (contrastive learning, masking) do this.

        Parameters
        ----------
        audio_batch
            Input batch. `PLAdapter` passes `batch[0].squeeze(1)`, so it arrives without a
            channel axis. The concrete shape (waveform or spectrogram, number of frames) is
            **set by the data config and the model adapts to it** — the toolkit does not enforce it.
        labels
            Integer class labels `(B,)`. ICBHI is 4-class, SPRSound is 7-class.
        loss_func
            Loss object built from `configs/loss/*.yaml`. Losses that depend on the dataset
            distribution (e.g. weighted CE) are injected here, so **use the one passed in instead
            of building a new loss inside the model.** Add model-specific terms (contrastive loss, etc.) to it.
        **kwargs
            Optional arguments. Currently the only one the toolkit may pass is `meta_labels`,
            which arrives as the third batch element only when the datamodule's `meta_label_mode` is on
            (SG-SCL and DAT receive it in training; the BTS family also in validation).
            **Models that do not use it must still accept `**kwargs`** — so the same model runs
            with data configs that attach meta and with those that do not.
            The meaning of `meta_labels` differs per datamodule (stethoscope id, patient number, AQS float,
            ssl_label, meta7 vector, ...). Models that consume it should declare which datamodule
            `meta_kind` values they understand via the class attribute `accepts_meta = ("dev", ...)` — the adapter
            checks the combination before training starts to prevent silent mistraining (no check if undeclared).

        Returns
        -------
        `(logits, loss)`
            `logits` is `(B, n_cls)` — metric computation assumes this shape.
            `loss` is a 0-dim tensor that supports backprop.
        """
        raise NotImplementedError
