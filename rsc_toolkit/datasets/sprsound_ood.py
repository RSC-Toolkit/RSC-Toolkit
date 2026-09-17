"""
SPRSound -> ICBHI 4-class loader for OOD evaluation only (BTS-CARD; not used for training).

**Non-standard use**: SPRSound is intended for 7-class event classification (2022 inter-patient test). The BTS-CARD paper
merges the 2022 inter-patient valid set (1,429 events) into ICBHI 4 classes and uses it only as an OOD test. What is built here is
that derived view; the original data and index are unchanged.

Label merge (official notebook data/Preprocessing_SPRSound.ipynb cells 2/5):
  Normal -> 0, Fine Crackle / Coarse Crackle -> 1 (Crackle), Wheeze -> 2, Both (Wheeze+Crackle) -> 3,
  Rhonchi / Stridor -> **dropped** (notebook map_annotation_type -> None). Toolkit 4-class rule (excluded even after reviewing the case for merging):
                       the paper's Sec. 3.1 merge into wheeze is not supported -- passing rhonchi_stridor="wheeze" raises ValueError.
  The 2022 inter-test has 0 Rhonchi/Stridor events, so the OOD set is 1040/83/305/1 either way.
  (Notebook quirk: the 'wheeze + crackle' branch does not match the original string 'Wheeze+Crackle', falls through to else, and
   _get_lungsound_label('Wheeze+Crackle') -> 3 -- the result is the same.)

3 meta fields (notebook cell 7): [age (float, from file name), gender (int; 0=male, 1=female), loc p1..p4 -> 0..3]
Sentence template (util/sprsound_util.py): "This sound was recorded from the {loc} of a {sex} pediatric patient."
  loc: 0 left posterior / 1 left lateral / 2 right posterior / 3 right lateral (matches the SPRSound README definition;
  the 'L/R Anterior' notation in the paper's Table 1 disagrees with the dataset documentation). Only age >= 19 is 'adult' -- the valid set is all pediatric.
  CLAP and tiny-bert use the same sentence, no masking (sprsound_dataset.py:35-41).

Two paths:
  (A) valid_spr.pt released by the authors (Google Drive 1zGR…; pretrained_models/upstream/btscard/valid_spr.pt)
      item = (log-mel (1,1001,64), label 0-3, meta3, file path string)
  (B) The toolkit's own pipeline: builds the test set of IndexDataset (48 kHz, fade in/out sr/16 over the whole recording
      -> segment crop -> 8 s repeat padding + fade_out (pad_type repeat_fade) -> CLAPProcessor log-mel) from the configs/data/sprsound_btscard.yaml arguments and
      converts it to 4 classes. Same order as notebook cell 5. Differs from the existing sprsound.yaml (pad_type repeat, no fade_out).
  Feature/label/sentence agreement between the two is recorded by scripts/reproduction/check_btscard_sprsound_parity.py (tests/golden/btscard/sprsound_parity.json):
  labels, meta3 and sentences match for all 1,429 events; log-mel max|diff| 6.7e-3 (24 events > 1e-3; not bit-identical -- **cause unconfirmed** (presumably numeric differences in
  Resample/numpy in the authors' torch 2.0.1/Windows environment; not measured). Segment arithmetic matches the notebook via _NotebookSliceIndexDataset).
"""
from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

AUTHOR_PT_DEFAULT = "pretrained_models/upstream/btscard/valid_spr.pt"
DATA_CONFIG_DEFAULT = "rsc_toolkit/configs/data/sprsound_btscard.yaml"

SPRSOUND_LABELS_7 = ("Normal", "Fine Crackle", "Coarse Crackle", "Wheeze", "Both", "Rhonchi", "Stridor")
LOC_ID_TO_STR = {0: "left posterior", 1: "left lateral", 2: "right posterior", 3: "right lateral"}
LOC_CODE_TO_ID = {"p1": 0, "p2": 1, "p3": 2, "p4": 3}
CLASS_NAMES_4 = ("normal", "crackle", "wheeze", "both")


def sprsound_label_to_4class(label: str, rhonchi_stridor: str = "drop") -> Optional[int]:
    """7-class name -> ICBHI 4-class id. Rhonchi/Stridor are excluded (None). rhonchi_stridor only accepts "drop"."""
    name = str(label).strip().lower()
    if name == "normal":
        return 0
    if name in ("fine crackle", "coarse crackle", "crackle"):
        return 1
    if name == "wheeze":
        return 2
    if name in ("both", "wheeze+crackle", "wheeze + crackle"):
        return 3
    if name in ("rhonchi", "stridor"):
        if rhonchi_stridor == "drop":
            return None
        raise ValueError(f"rhonchi_stridor={rhonchi_stridor!r}: Rhonchi/Stridor can only be excluded in the 4-class view "
                         "(the paper's Sec. 3.1 merge into wheeze is not supported)")
    raise KeyError(f"unknown SPRSound label {label!r}")


def generate_meta_description_sprsound(age: str, sex: str, loc: str) -> str:
    """Same as util/sprsound_util.py:1-24."""
    loc_phrase = "from an unknown region" if loc == "unknown" else f"from the {loc}"
    if age == "unknown" and sex == "unknown":
        age_sex_phrase = "a patient of unknown age and sex"
    elif age == "unknown":
        age_sex_phrase = f"a {sex} patient of unknown age"
    elif sex == "unknown":
        age_sex_phrase = f"a {age} patient of unknown sex"
    elif age == "pediatric":
        age_sex_phrase = f"a {sex} pediatric patient"
    else:
        article = "an" if age[0].lower() in "aeiou" else "a"
        age_sex_phrase = f"{article} {age} {sex} patient"
    return f"This sound was recorded {loc_phrase} of {age_sex_phrase}."


def describe_meta3_sprsound(meta3) -> str:
    """util/sprsound_util.py get_meta_text_descriptions_sprsound: [age, sex, loc] -> sentence."""
    age, sex, loc = int(meta3[0]), int(meta3[1]), int(meta3[2])
    age_str = "adult" if age >= 19 else "pediatric"
    sex_str = "male" if sex == 0 else "female"
    return generate_meta_description_sprsound(age_str, sex_str, LOC_ID_TO_STR[loc])


def meta_fields_from_meta7_sprsound(meta7):
    """
    SPRSound dialect: reads only age/sex/loc from the BTS-family 7-field vector and turns them into sentence fields.

    Two differences from the ICBHI version (`meta_fields_from_meta7`):
      · the loc vocabulary is SPRSound's p1~p4 (LOC_ID_TO_STR) -- using ICBHI's 7 chest locations would invent locations that do not exist.
      · dev is not used (None). All 8,085 SPRSound events come from a single stethoscope (Yunting2), so it carries no information.
    The other three slots (BMI, weight, height) are -1 (missing) and are not read here either.
    """
    age, sex, loc = int(meta7[0]), int(meta7[1]), int(meta7[5])
    return {
        "age": "adult" if age >= 19 else "pediatric",
        "sex": "male" if sex == 0 else "female",
        "loc": LOC_ID_TO_STR[loc],
        "dev": None,
    }


def meta3_from_index_meta(meta: Dict[str, Any]) -> List[float]:
    """Toolkit SPRSound index meta dict -> [age, gender, loc_id] of notebook cell 7."""
    loc = meta.get("location_id")
    if loc is None:
        loc = LOC_CODE_TO_ID[str(meta["recording_location"])]
    return [float(meta["age"]), float(int(float(meta["gender"]))), float(int(loc))]


from rsc_toolkit.datasets import collate as _collate  # single collate implementation

# (x, y, meta7 stack)
sprsound_ood_collate = _collate.stack_meta_collate


# --------------------------------------------------------------------------- (A) released by the authors
class SPRSoundOODAuthorDataset(Dataset):
    """valid_spr.pt loader. item -> (feat (1,1001,64), label, meta3 float(3), recording_id)."""

    def __init__(self, path: str = AUTHOR_PT_DEFAULT):
        self.path = pathlib.Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(f"authors' valid_spr.pt not found: {self.path} (README Google Drive 1zGR…)")
        self.items = torch.load(self.path, map_location="cpu", weights_only=False)
        logger.info("Loaded the authors' SPRSound OOD tensor: %s (%d events)", self.path, len(self.items))

    def __len__(self):
        return len(self.items)

    @staticmethod
    def recording_id(path_str: str) -> str:
        # Windows path from the notebook ('...\\40512331_8.1_1_p1_3548.wav')
        return path_str.replace("\\", "/").rsplit("/", 1)[-1].replace(".wav", "")

    def __getitem__(self, i):
        feat, label, meta, rid = self.items[i]
        return feat, int(label), meta.float(), self.recording_id(rid)


# --------------------------------------------------------------------------- (B) toolkit pipeline
class SPRSoundOOD4ClassDataset(Dataset):
    """
    IndexFeatureDataset (toolkit SPRSound test split, CLAP features) -> derived 4-class view.
    Label names are looked up by sample_id in the bundled metadata.csv (Filename, Start, End -> Label), not from the index label id
    (so as not to depend on the LabelEncoder id order).
    """

    def __init__(self, feature_ds, rhonchi_stridor: str = "drop", metadata_csv: Optional[str] = None):
        import pandas as pd

        self.feature_ds = feature_ds
        index_ds = getattr(feature_ds, "index_ds", feature_ds)
        df = index_ds.df
        csv = pathlib.Path(metadata_csv) if metadata_csv else (
            pathlib.Path(__file__).parent / "sprsound" / "data" / "metadata.csv")
        meta_df = pd.read_csv(csv)
        name_by_sid = {f"{r.Filename}_cycle_{float(r.Start):.3f}_{float(r.End):.3f}": r.Label
                       for r in meta_df.itertuples()}
        self.rows: List[Tuple[int, int, List[float], str, str]] = []  # (idx, label4, meta3, sample_id, label7)
        dropped = 0
        import json

        for i, row in enumerate(df.itertuples()):
            sid = str(row.sample_id)
            if sid not in name_by_sid:
                raise KeyError(f"label for {sid} not found in metadata.csv")
            label7 = name_by_sid[sid]
            y4 = sprsound_label_to_4class(label7, rhonchi_stridor)
            if y4 is None:
                dropped += 1
                continue
            meta = row.meta
            if isinstance(meta, str):
                meta = json.loads(meta)
            self.rows.append((i, y4, meta3_from_index_meta(meta), sid, label7))
        logger.info("SPRSound OOD 4-class view: %d events (Rhonchi/Stridor %s: %d)", len(self.rows), rhonchi_stridor, dropped)

    def __len__(self):
        return len(self.rows)

    def class_counts(self) -> List[int]:
        c = [0, 0, 0, 0]
        for _, y, _, _, _ in self.rows:
            c[y] += 1
        return c

    def __getitem__(self, i):
        idx, y4, meta3, sid, _ = self.rows[i]
        feats = self.feature_ds[idx][0]
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        return feats, y4, torch.tensor(meta3, dtype=torch.float32), sid


class _NotebookSliceIndexDataset(Dataset):
    """
    Reproduces the segment arithmetic of notebook cell 5 on top of IndexDataset:
      start = int(float(ms) * 0.001 * sr) -- using only the index CSV t0 (seconds, float) gives off-by-one-sample events such as
      int(9.274 * 48000) = 445151 (73/1429 events with log-mel max|diff| > 1e-3 in the comparison). Integer ms is recovered as round(t0 * 1000).
    """

    def __init__(self, index_ds, sampling_rate: int, preprocessor):
        self.index_ds = index_ds
        self.df = index_ds.df
        self.sr = int(sampling_rate)
        self.pre = preprocessor

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, i):
        from rsc_toolkit.datasets.icbhi.processing import load_audio

        row = self.df.iloc[int(i)]
        wav = load_audio(self.index_ds._resolve_recording_path(str(row["recording_path"])), target_sampling_rate=self.sr,
                         fade=self.index_ds.fade)
        max_ind = wav.shape[1]
        s = float(round(float(row["t0"]) * 1000)) * 0.001
        e = float(round(float(row["t1"]) * 1000)) * 0.001
        audio = wav[:, min(int(s * self.sr), max_ind): min(int(e * self.sr), max_ind)]
        label = row["label"]
        return {"audio": self.pre(audio), "label": None if label != label else int(label), "sample_id": str(row["sample_id"])}


class _CLAPFeatureDataset(Dataset):
    """
    IndexDataset (48 kHz, fade, repeat_fade) item -> HF ClapProcessor log-mel (1, 1001, 64), cached with FeatureCache.
    Uses the same processor and cache key (model_name, sampling_rate) as processors/clap.py CLAPProcessor but passes the input as a numpy 1-D array
    (ClapFeatureExtractor in transformers 5.15.1 treats a torch 1-D tensor as 2-D and raises MemoryError -- the shared processor file is left unchanged).
    """

    def __init__(self, index_ds, sampling_rate: int, model_name: str, cache_dir: pathlib.Path, audio_chain=None):
        from transformers import ClapProcessor

        from rsc_toolkit.datasets.feature_cache import FeatureCache

        self.index_ds = index_ds
        self.processor = ClapProcessor.from_pretrained(model_name, sampling_rate=sampling_rate)
        self.cache = FeatureCache(cache_dir)
        # audio_chain: records the upstream preprocessing of this path (48 kHz, fade, repeat_fade 8 s, notebook-style ms-rounded crop) in the chain ledger
        # (dataset_name="sprsound_ood" -- both the cache dir and the ledger key differ from the toolkit sprsound_bts path).
        self.cache.configure({"model_name": model_name, "sampling_rate": sampling_rate}, processor_name="CLAPProcessor",
                             audio_chain=audio_chain)

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, i):
        item = self.index_ds[i]
        sid = item["sample_id"]
        feats = self.cache.get(sid) if self.cache.is_cached(sid) else None
        if feats is None:
            audio = item["audio"].squeeze().numpy()
            feats = self.processor(audio=audio, return_tensors="pt", padding=True)["input_features"][0]
            self.cache.put(sid, feats)
        return feats, item["label"], sid


def build_toolkit_ood_dataset(data_config: str = DATA_CONFIG_DEFAULT, rhonchi_stridor: str = "drop",
                              project_root: Optional[str] = None) -> SPRSoundOOD4ClassDataset:
    """Build IndexDataset (test = 2022 inter-patient) with the preprocessing arguments of configs/data/sprsound_btscard.yaml and wrap it in the 4-class view."""
    from omegaconf import OmegaConf

    from rsc_toolkit.datasets.index_dataset import IndexDataset
    from rsc_toolkit.processors.audio import AudioPreprocessConfig, AudioPreprocessor

    from rsc_toolkit.paths import project_root_from

    root = project_root_from(project_root)  # argument -> RSCTK_PROJECT_ROOT -> cwd (repository marker) -> development checkout (replaces the parents[2] guess for installed copies)
    cfg = OmegaConf.to_container(OmegaConf.load(root / data_config), resolve=False)["params"]
    data_dir = root / cfg["data_dir"]
    sr = int(cfg["sampling_rate"])
    pre = AudioPreprocessor(sr, AudioPreprocessConfig(float(cfg["desired_length_sec"]), str(cfg["pad_type"])))
    splits = list(cfg["test_splits"])
    assert len(splits) == 1, "SPRSound OOD supports a single test split only"
    index_ds = IndexDataset(
        index_csv=data_dir / "index" / cfg.get("index_filename", "index_cycle.csv"),
        data_root=data_dir / cfg.get("raw_data_subdir", ""), sampling_rate=sr, sample_level=cfg["sample_level"],
        split=str(splits[0]), audio_preprocessor=None, fade=bool(cfg.get("fade", True)),
    )
    feats = cfg["features"]
    sliced = _NotebookSliceIndexDataset(index_ds, sr, pre)
    audio_chain = {
        "dataset_name": "sprsound_ood", "sampling_rate": sr, "sample_level": str(cfg["sample_level"]),
        "desired_length_sec": float(cfg["desired_length_sec"]), "pad_type": str(cfg["pad_type"]),
        "fade": bool(cfg.get("fade", True)), "time_unit": "ood_ms_round",  # crop rule of _NotebookSliceIndexDataset
    }
    feature_ds = _CLAPFeatureDataset(sliced, sr, feats["clap_model_name"], root / feats["cache_dir"], audio_chain=audio_chain)
    return SPRSoundOOD4ClassDataset(feature_ds, rhonchi_stridor=rhonchi_stridor)


def build_ood_dataset(source: str = "author", **kw) -> Dataset:
    if source == "author":
        return SPRSoundOODAuthorDataset(kw.get("path", AUTHOR_PT_DEFAULT))
    if source == "toolkit":
        return build_toolkit_ood_dataset(kw.get("data_config", DATA_CONFIG_DEFAULT),
                                         kw.get("rhonchi_stridor", "drop"), kw.get("project_root"))
    raise ValueError(f"source must be author|toolkit, got {source!r}")
