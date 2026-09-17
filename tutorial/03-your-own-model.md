# 3. Your own model (CPU)

[Tutorial index](README.md) · previous: [2. Toy data](02-toy-data.md) · next: [4. ICBHI 2017](04-icbhi.md)

Uses `WORK`, `ROOT` and `SANDBOX` from [part 1](01-setup.md#working-directory) and the toy index from [part 2](02-toy-data.md).

A model is one Python file plus a small config. The toolkit does not need to be forked; both live outside the repository.
The full contract is documented in [`rsc_toolkit/models/base.py`](../rsc_toolkit/models/base.py).

```
$WORK/plugins/
  my_model.py
  configs/model/tutorial_cnn.yaml
```

```bash
mkdir -p "$WORK/plugins/configs/model"
```

## 1. Model

The model file has three parts.

**(a) Registration.** The decorator registers the class under a name:

```python
@ModelFactory.register_model_class("tutorial_cnn")
class TutorialCNN(BaseRSCModel, nn.Module):
```

- The name is how configs refer to the model: the `name:` in the model config (step 2) must be the same string.
- The registered name is independent of the file name: the file `my_model.py` is the module `my_model`, which
  `RSC_TOOLKIT_PLUGINS` imports in step 3, while configs use the registered name `tutorial_cnn`.
- Inherit `BaseRSCModel` (the toolkit's model interface) and `nn.Module`.

**(b) Constructor.** Its keyword arguments are filled from `params:` in the model config:

```python
    def __init__(self, n_cls: int = 4, hidden: int = 16, **params):
```

- `n_cls` must equal the number of classes of the dataset (4 for ICBHI, 7 for SPRSound).
- Keep `**params` so that extra keys in the config do not break the constructor.

**(c) `training_forward`.** The toolkit calls only this method, for training and for validation:

```python
    def training_forward(self, audio_batch, labels, loss_func, **kwargs):
```

- `audio_batch`: the input batch without a channel axis; its shape is set by the data config (`[B, F, T]` for fbank
  features).
- `labels`: class indices of shape `[B]`.
- `loss_func`: the loss built from the config (for example class-weighted cross-entropy); use it instead of creating
  a loss inside the model.
- Return `(logits, loss)` with logits of shape `[B, n_cls]`.
- Keep `**kwargs`: some datamodules pass extra inputs such as `meta_labels`. Branch on `self.training` if training and
  validation must differ.
- `forward` is optional; the toolkit never calls it. This example defines it for convenience and calls it from
  `training_forward`.

The complete file — save as `$WORK/plugins/my_model.py`:

```python
from torch import nn
from rsc_toolkit.models.base import BaseRSCModel
from rsc_toolkit.models.factory import ModelFactory


@ModelFactory.register_model_class("tutorial_cnn")
class TutorialCNN(BaseRSCModel, nn.Module):
    def __init__(self, n_cls: int = 4, hidden: int = 16, **params):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(), nn.MaxPool2d(4),
                                 nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Linear(hidden, n_cls))

    def forward(self, x):
        return self.net(x.unsqueeze(1) if x.dim() == 3 else x)      # [B, F, T] -> [B, 1, F, T] for Conv2d

    def training_forward(self, audio_batch, labels, loss_func, **kwargs):
        logits = self.forward(audio_batch)
        return logits, loss_func(logits, labels)
```

## 2. Model config

Save as `$WORK/plugins/configs/model/tutorial_cnn.yaml`:

```yaml
name: tutorial_cnn
params:
  n_cls: 4
  hidden: 16
```

## 3. Train and re-evaluate

`PYTHONPATH` makes the module importable, `RSC_TOOLKIT_PLUGINS` imports it so the decorator registers the model, and
`--config-dir` adds the model config:

```bash
export PYTHONPATH="$WORK/plugins" RSC_TOOLKIT_PLUGINS=my_model
```

```bash
rsc-toolkit-train --config-name default --config-dir "$WORK/plugins/configs" data=toy_fbank model=tutorial_cnn callbacks=best_checkpoint \
  data.params.project_root="$ROOT" data.params.data_dir="$WORK/toy" data.params.features.cache_dir="$WORK/cache_toy" \
  data.params.num_workers=0 trainer.params.max_epochs=1 trainer.params.accelerator=cpu \
  run.name=toy-mymodel run.group=tutorial hydra.run.dir="$WORK/hydra/toy-mymodel" +run.overwrite=true
```

```bash
rsc-toolkit-eval "$WORK/outputs/tutorial/toy-mymodel" --project-root "$ROOT" --hydra-dir "$WORK/hydra/toy-mymodel" --sandbox "$SANDBOX"
```

Keep `PYTHONPATH` and `RSC_TOOLKIT_PLUGINS` set for re-evaluation. An installed package can register itself instead through
the entry point `rsc_toolkit.plugins`. Losses, metrics, datamodules and feature processors use the same registry mechanism —
see [Other extension points](../README.md#other-extension-points) in the README.

---
previous: [2. Toy data](02-toy-data.md) · next: [4. ICBHI 2017](04-icbhi.md)
