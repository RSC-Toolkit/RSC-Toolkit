"""
QLung SPRSound OOD evaluation view -- 7-class events -> merged ICBHI 4 classes (Rhonchi and Stridor excluded), 16 kHz kaldi fbank 128x798 (only for zero-shot evaluation of ICBHI-trained models).

Paper Sec. 3.1: SPRSound merged into Normal 6,199 / Crackle 1,044 (Fine+Coarse) / Wheeze 811 (Wheeze+Rhonchi+Stridor) / Both 31 = 8,085
  -> matches the toolkit index (.datasets/sprsound/index/index_cycle.csv) train 6,656 + inter-patient test 1,429 exactly per class.
  The evaluation scope (all 8,085 vs inter-test 1,429) is not stated in the paper -> both are aggregated (scripts/reproduction/eval_qlung_ood.py).
  **The toolkit 4-class view excludes Rhonchi 39 and Stridor 15 instead of adding them to wheeze** -> 6,199 / 1,044 / 757 / 31 = 8,031;
  inter-test 1,429 is identical because both classes have 0 events there (1,040 / 83 / 305 / 1). An all-8,085 aggregate (ood.json 'all') uses the merged labels and is not comparable.
The repository has no SPRSound code, so preprocessing is **assumed** (flags, configs/data/sprsound_qlung_ood.yaml):
  - 16 kHz resampling (whole recording, 8 kHz source audio), recording fade in/out sr/16, segment crop (seconds, int(t*sr)), 8 s repeat_fade -- same chain as ICBHI
    (paper Sec. 3.1 states '8 s standardization for all experiments'; resampling, padding and normalization constants are not stated -> assumed identical to ICBHI)
  - kaldi fbank 128 mel / 10 ms / fixed AudioSet mean/std normalization (same as ICBHI AST)
  - the feature cache uses a separate directory (downloaded_datasets/features_cache_qlung_ood) -- the cache key reflects only the processor config, so it must not mix with other preprocessing chains.
The label merge reuses lungmix_data.SPRSOUND_7_TO_4 (based on index ids; verified against the bundled metadata.csv counts).
"""
from __future__ import annotations

import pathlib
from typing import Dict, List, Optional, Sequence

from omegaconf import OmegaConf

from rsc_toolkit.datasets.lungmix_data import SPRSOUND_7_TO_4, SPRSOUND_EVENT_ID_TO_NAME, SPRSOUND_NAME_TO_4CLASS  # noqa: F401
from rsc_toolkit.datasets.qlung_data import QLungDataModule

DATA_CONFIG_DEFAULT = "rsc_toolkit/configs/data/sprsound_qlung_ood.yaml"
CLASS_NAMES_4 = ("normal", "crackle", "wheeze", "both")
# Paper Sec. 3.1 merged distribution (train 6,656 + inter-test 1,429; Rhonchi/Stridor -> wheeze) -- recorded paper-reported value, differs from the toolkit view
PAPER_MERGED_COUNTS = (6199, 1044, 811, 31)
# Toolkit 4-class view (Rhonchi 39 and Stridor 15 excluded -> 8,031 events)
TOOLKIT_4CLASS_COUNTS = (6199, 1044, 757, 31)


def sprsound_label7_to_4(label_id: int) -> Optional[int]:
    """Toolkit SPRSound index label id (0 Normal/1 Fine/2 Wheeze/3 Coarse/4 Both/5 Rhonchi/6 Stridor) -> 4 classes. Excluded classes (5, 6) give None."""
    return SPRSOUND_7_TO_4.get(int(label_id))


def build_sprsound_ood_datamodule(splits: Sequence[str], data_config: str = DATA_CONFIG_DEFAULT,
                                  project_root: Optional[str] = None, overrides: Optional[Dict] = None) -> QLungDataModule:
    """
    Build a QLungDataModule from the params of configs/data/sprsound_qlung_ood.yaml, replacing the val/test splits with `splits`
    (e.g. ["test"] = inter-test 1,429, ["train", "test"] = 8,085). After setup("test"), test_dataloader() yields (x, y4, aqs) batches.
    """
    from rsc_toolkit.paths import project_root_from

    root = project_root_from(project_root)  # argument -> RSCTK_PROJECT_ROOT -> cwd (repository marker) -> development checkout
    cfg = OmegaConf.to_container(OmegaConf.load(root / data_config), resolve=False)["params"]
    cfg = dict(cfg)
    cfg.pop("datamodule_name", None)
    cfg["project_root"] = str(root)
    cfg["val_splits"] = list(splits)
    cfg["test_splits"] = list(splits)
    if overrides:
        cfg.update(overrides)
    return QLungDataModule(**cfg)


def class_counts_4(ds) -> List[int]:
    """4-class label counts of a QLungAQSIndexDataset (or ConcatDataset)."""
    from torch.utils.data import ConcatDataset

    if isinstance(ds, ConcatDataset):
        out = [0, 0, 0, 0]
        for sub in ds.datasets:
            for i, c in enumerate(class_counts_4(sub)):
                out[i] += c
        return out
    counts = ds.df["label"].value_counts().to_dict()
    return [int(counts.get(i, 0)) for i in range(4)]
