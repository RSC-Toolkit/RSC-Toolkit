# 3. Your own model (CPU)

[Tutorial index](README.md) · previous: [2. Toy data](02-toy-data.md) · next: [4. ICBHI 2017](04-icbhi.md)

Uses `WORK`, `ROOT` and `SANDBOX` from [part 1](01-setup.md#working-directory) and the toy index from [part 2](02-toy-data.md).

A model is one Python file: subclass `BaseRSCModel`, implement `training_forward`, and register the class by name
(contract: [`rsc_toolkit/models/base.py`](../rsc_toolkit/models/base.py)). The toolkit does not need to be forked; the model
and its config live outside the repository.

```
$WORK/plugins/
  my_model.py
  configs/model/tutorial_cnn.yaml
```

```bash
mkdir -p "$WORK/plugins/configs/model"
```

## 1. Model

Save as `$WORK/plugins/my_model.py`:

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
        return self.net(x.unsqueeze(1) if x.dim() == 3 else x)      # [B, F, T] -> [B, 1, F, T]

    def training_forward(self, audio_batch, labels, loss_func, **kwargs):
        logits = self.forward(audio_batch)
        return logits, loss_func(logits, labels)
```

- `audio_batch` has the shape defined by the data config (`[B, F, T]` for fbank features).
- `loss_func` is the loss from the config. Return `(logits [B, n_cls], loss)`.

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
