from __future__ import annotations

import logging
from typing import Any, Dict, List

from hydra.utils import instantiate

import lightning as L

logger = logging.getLogger(__name__)


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

    logger_cfg = params.get("logger")
    if logger_cfg:
        if isinstance(logger_cfg, list):
            loggers: List[Any] = []
            for log_cfg in logger_cfg:
                if isinstance(log_cfg, dict) and "_target_" in log_cfg:
                    logger.info("Instantiating trainer logger: %s", log_cfg.get("_target_"))
                    log_cfg = instantiate(log_cfg)
                loggers.append(log_cfg)
            params["logger"] = loggers
            logger.info("Trainer loggers: %s", [type(l).__name__ for l in loggers])
        elif isinstance(logger_cfg, dict) and "_target_" in logger_cfg:
            logger.info("Instantiating trainer logger: %s", logger_cfg.get("_target_"))
            params["logger"] = instantiate(logger_cfg)
            logger.info("Trainer logger: %s", type(params["logger"]).__name__)

    callbacks_cfg = params.get("callbacks")
    if callbacks_cfg:
        callbacks: List[L.Callback] = []
        for cb in callbacks_cfg:
            if isinstance(cb, dict) and "_target_" in cb:
                logger.info("Instantiating trainer callback: %s", cb.get("_target_"))
                cb = instantiate(cb)
            if not isinstance(cb, L.Callback):
                raise TypeError(f"trainer.params.callbacks must be Lightning Callbacks, got {type(cb)}")
            callbacks.append(cb)
        params["callbacks"] = callbacks
        logger.info("Trainer callbacks: %s", [f"{type(c).__module__}.{type(c).__name__}" for c in callbacks])
    else:
        logger.info("Trainer callbacks: [] (no trainer.params.callbacks configured)")
    return L.Trainer(**params)
