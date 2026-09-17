import inspect
from typing import Any, Dict, List

import lightning as L

from rsc_toolkit.registries import DATAMODULES

# Keys injected by configs/data_defaults/base.yaml via `data_defaults@data: base`; they are arguments of the index-family datamodules.
# They are left over when combined via a `data=<name>` override with a datamodule that has a strict signature (bts_precomputed, respirenet, etc.).
_DATA_DEFAULTS_KEYS = ("desired_length_sec", "pad_type", "specaugment")


def unknown_datamodule_kwargs(cls: type, params: Dict[str, Any]) -> List[str]:
    """Keys that `cls(**params)` would not accept. Empty list if the class takes **kwargs (its own responsibility)."""
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):      # C extensions, callables without a signature
        return []
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return []
    accepted = {n for n, p in sig.parameters.items()
                if n != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    return sorted(k for k in params if k not in accepted)


def build_datamodule(cfg: Dict[str, Any]) -> L.LightningDataModule:
    """
    cfg example (configs/data/*.yaml):
      {"name": "icbhi",                       # dataset name (for logs and run names; not the lookup key)
       "params": {"datamodule_name": "index",  # lookup key, or top-level "datamodule": "index"
                  "batch_size": 32, "num_workers": 4, ...}}

    The datamodule is specified by `params.datamodule_name` (existing convention) or a top-level `datamodule`.
    params is copied before use, so cfg is not modified.
    Missing/unregistered errors name the key that actually failed (datamodule_name).
    """
    params = dict(cfg.get("params") or {})
    name = params.pop("datamodule_name", None) or cfg.get("datamodule")
    if name is None:
        raise KeyError(
            "data config does not specify a datamodule: set data.params.datamodule_name "
            f"(or data.datamodule). Available: {sorted(DATAMODULES)}"
        )
    try:
        cls = DATAMODULES.get(name)
    except KeyError:
        raise KeyError(
            f"Unknown datamodule '{name}' (data.params.datamodule_name). Available: {sorted(DATAMODULES)}"
        ) from None
    unknown = unknown_datamodule_kwargs(cls, params)
    if unknown:
        # A bare TypeError("unexpected keyword argument ...") from cls(**params) would not say which config
        # injected the key, so report it explicitly.
        injected = [k for k in unknown if k in _DATA_DEFAULTS_KEYS]
        hint = ""
        if injected:
            hint = (f" {injected} are index-family arguments injected by `data_defaults@data: base`; to use this datamodule, "
                    f"remove data_defaults from the defaults of the top-level config or remove them on the CLI: "
                    + " ".join(f"~data.params.{k}" for k in injected) + ".")
        raise TypeError(
            f"datamodule '{name}' ({cls.__module__}.{cls.__qualname__}) does not accept the data.params keys {unknown}."
            f"{hint} (If this is a typo, check the key names in the config.)"
        )
    return cls(**params)
