from typing import Any, Dict

import lightning as L


def build_trainer(cfg: Dict[str, Any]) -> L.Trainer:
    """
    cfg example:
      {
        "params": {
          "max_epochs": 50,
          "accelerator": "gpu",
          "devices": 1,
          "log_every_n_steps": 50
        }
      }
    """
    params = cfg.get("params", {})
    return L.Trainer(**params)
