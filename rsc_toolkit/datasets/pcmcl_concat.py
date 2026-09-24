# Portions of this file are ported from PC-MCL (https://github.com/wa976/PC-MCL), which does not include a license file.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
Training dataset of PC-MCL (arXiv 2601.17080; official code wa976/PC-MCL util/dataset_pcmcl.py ConcatenationMultilabelDataset)
-- keeps respiratory cycles at original length + dynamic concatenation at __getitem__ time (11 combinations) + PM (patient matching) pair generation.

Behaviour of the official code (exact port verified against goldens; all kept as defaults):
- Iterates the train files of the official 60/40 split in sorted stem order and extracts cycles **at original length**
  (torchaudio.load -> 16 kHz resample -> fade in/out sr/16 over the whole recording -> crop [start,end); icbhi_util.py get_individual_cycles_torchaudio_beats)
  -> identical to the toolkit's datasets/icbhi/processing.py load_audio(fade 1/16) + slice_audio.
- Per-class lists (normal/crackle/wheeze/both; both also goes into the crackle and wheeze lists),
  a 'short cycle' (length < T/2) list, and a per-patient cycle dict (insertion order).
- Size per epoch = N + int(N*mixing_prob) + int(N*ssl_prob). index < N original, next range concatenated samples, last range PM samples.
- Concatenation: uniform choice among 11 additive combinations -> randomly draw only **short** cycles of each class (regardless of patient), torch.cat ->
  T s cut_pad (crop from the front / repeat + fade_out sr/16). If the condition is not met (too few short cycles), a random original cycle is returned.
- PM: 50% positive = two cycles of the same patient_id (70% non-consecutive gap>=2 / 30% consecutive, no length limit, labels OR-ed),
  50% hard negative = two 'patients' with the same pathology profile -- quirk: returns immediately from the first group with >=2 in dict insertion order (starting from normal_only).
  The main label (OR) of PM samples also enters the main BCE loss (no masking).
- **patient_id quirk**: in official split mode, patient_id = the whole file name (= recording). The paper's intent (patient number) is patient_id_mode="patient".
- Test (train=False) uses original cycles only (single input after cut_pad) -- the toolkit reuses the existing IndexDataset (repeat_fade 10 s).

RNG: the original uses the python `random` module in __getitem__ (num_workers=0 by default -> a single stream). Here a dedicated
random.Random instance is used with the same call order as the original, so with rng_seed set and num_workers=0
the same seed yields the same sample sequence (pinned by goldens). With num_workers>0 each worker gets a separate stream seeded from torch.initial_seed() (derived by Lightning
seed_everything(workers=True)) -> 'same distribution' as the original, not the same sequence.
"""
from __future__ import annotations
import os

import hashlib
import logging
import pathlib
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

from rsc_toolkit.datasets.icbhi.processing import load_audio, slice_audio
from rsc_toolkit.datasets.index_dataset import IndexDataset
from rsc_toolkit.processors.audio import AudioPreprocessConfig, AudioPreprocessor

logger = logging.getLogger(__name__)

CLASS_NAMES = ("normal", "crackle", "wheeze", "both")


def convert_to_multilabel(original_label: int) -> List[int]:
    """Original _convert_to_multilabel: 0->[1,0,0], 1->[0,1,0], 2->[0,0,1], 3->[0,1,1]."""
    multilabel = [0, 0, 0]
    if original_label == 0:
        multilabel[0] = 1
    elif original_label == 1:
        multilabel[1] = 1
    elif original_label == 2:
        multilabel[2] = 1
    elif original_label == 3:
        multilabel[1] = 1
        multilabel[2] = 1
    return multilabel


def multilabel_mixing_combinations(strategy: str = "additive") -> List[tuple]:
    """Same order and content as the original _get_multilabel_mixing_combinations (random.choice indices must match)."""
    combinations: List[tuple] = []
    if strategy == "additive":
        combinations.extend([(0, 1, 1, 1, 0, 1.0), (0, 2, 1, 0, 1, 1.0), (1, 2, 0, 1, 1, 1.0)])
    elif strategy == "dominant":
        combinations.extend([(0, 1, 0, 1, 0, 1.0), (0, 2, 0, 0, 1, 1.0), (1, 2, 0, 1, 1, 1.0)])
    else:
        raise ValueError(f"mixing_label_strategy must be additive|dominant, got {strategy!r}")
    combinations.extend([(0, 0, 1, 0, 0, 1.0), (1, 1, 0, 1, 0, 1.0), (2, 2, 0, 0, 1, 1.0)])
    if strategy == "additive":
        combinations.extend([(0, 3, 1, 1, 1, 1.0), (3, 1, 0, 1, 1, 1.0), (3, 2, 0, 1, 1, 1.0), (3, 3, 0, 1, 1, 1.0)])
        combinations.append((0, 1, 2, 1, 1, 1, 1.0))
    else:
        combinations.extend([(0, 3, 0, 1, 1, 1.0), (3, 1, 0, 1, 1, 1.0), (3, 2, 0, 1, 1, 1.0), (3, 3, 0, 1, 1, 1.0)])
        combinations.append((0, 1, 2, 0, 1, 1, 1.0))
    return combinations


def single_label_mixing_combinations() -> List[tuple]:
    """ablation concat_single (ConcatenationSingleLabelDataset._get_single_label_mixing_combinations): dominant 4-class."""
    return [
        (0, 0, 0, 1.0), (1, 1, 1, 1.0), (2, 2, 2, 1.0), (3, 3, 3, 1.0),
        (0, 1, 1, 1.0), (0, 2, 2, 1.0), (0, 3, 3, 1.0), (1, 2, 3, 1.0), (1, 3, 3, 1.0), (2, 3, 3, 1.0),
    ]


def _cat(*audios: torch.Tensor) -> torch.Tensor:
    """Original _concatenate_two/three_audios: shape to (1, L) and concatenate along time (first channel if multi-channel)."""
    parts = []
    for a in audios:
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if a.shape[0] > 1:
            a = a[0:1]
        parts.append(a)
    return torch.cat(parts, dim=-1)


class PCMCLConcatDataset(Dataset):
    def __init__(
        self,
        index_csv: Union[str, pathlib.Path],
        data_root: Union[str, pathlib.Path],
        split_file: Optional[Union[str, pathlib.Path]] = None,
        split: str = "train",
        train: bool = True,
        sampling_rate: int = 16000,
        desired_length_sec: float = 10.0,
        output_kind: str = "waveform",          # waveform (BEATs) | fbank (AST: (n_mels, frames) freq_time)
        n_mels: int = 128,
        # ---- original arguments ----
        label_space: str = "icbhi4",            # icbhi4 (3 attributes defined by the authors) | sprsound7 (6 attributes defined by the toolkit)
        mode: str = "default",                  # default(concat+multi[+PM]) | concat_single | noconcat_multi
        mixing_prob: float = 1.0,               # paper 'one augmentation per original' = 1.0 (argparse default 0.5)
        conditional_mixing: bool = False,       # original --conditional_mixing (store_true; when on, consumes one extra random.random(), all probabilities 1.0)
        mixing_label_strategy: str = "additive",
        enable_ssl: bool = True,
        ssl_prob: float = 0.3,
        ssl_method: str = "coherence",
        # ---- paper-vs-code mismatch flags (default = code) ----
        patient_id_mode: str = "recording",     # recording (official mode quirk: whole file name) | patient (patient number split('_')[0])
        hard_negative_mode: str = "dict_order", # dict_order (quirk: first group with >=2) | random_profile (random among groups with >=2) | basic (any other patient)
        # ---- toolkit ----
        rng_seed: Optional[int] = None,         # None: derived per worker from torch.initial_seed() (training). Integer: fixed stream (goldens/tests)
        cache_dir: Optional[Union[str, pathlib.Path]] = "outputs/pcmcl/cache",
        fade: bool = True,
    ):
        super().__init__()
        if mode not in {"default", "concat_single", "noconcat_multi"}:
            raise ValueError(f"mode must be default|concat_single|noconcat_multi, got {mode!r}")
        if patient_id_mode not in {"recording", "patient"}:
            raise ValueError(f"patient_id_mode must be recording|patient, got {patient_id_mode!r}")
        if hard_negative_mode not in {"dict_order", "random_profile", "basic"}:
            raise ValueError(f"hard_negative_mode must be dict_order|random_profile|basic, got {hard_negative_mode!r}")
        if output_kind not in {"waveform", "fbank"}:
            raise ValueError(f"output_kind must be waveform|fbank, got {output_kind!r}")
        from rsc_toolkit.methods.pcmcl_labelspace import get_label_space

        self.ls = get_label_space(label_space)
        if mode == "concat_single" and self.ls.name != "icbhi4":
            raise NotImplementedError(
                "concat_single is an ablation by the authors using 4-class dominant labels and is defined only for icbhi4"
            )
        if ssl_method != "coherence":
            raise NotImplementedError("only the 'coherence' (PM) ssl_method used by the paper is ported (concat_single SSL is out of scope)")

        self.index_csv = pathlib.Path(index_csv)
        self.data_root = pathlib.Path(data_root)
        self.split_file = pathlib.Path(split_file) if split_file is not None else None
        self.split = split
        self.train_flag = bool(train)
        self.sample_rate = int(sampling_rate)
        self.desired_length = float(desired_length_sec)
        self.target_samples = int(self.desired_length * self.sample_rate)
        self.half_target_samples = self.target_samples // 2
        self.output_kind = output_kind
        self.n_mels = int(n_mels)
        self.mode = mode
        self.mixing_prob = float(mixing_prob) if mode != "noconcat_multi" else 0.0
        self.conditional_mixing = bool(conditional_mixing)
        self.mixing_label_strategy = mixing_label_strategy
        self.enable_ssl = bool(enable_ssl) and mode == "default"
        self.ssl_prob = float(ssl_prob)
        self.patient_id_mode = patient_id_mode
        self.hard_negative_mode = hard_negative_mode
        self.rng_seed = rng_seed
        self._rng: Optional[random.Random] = random.Random(rng_seed) if rng_seed is not None else None
        self.fade = bool(fade)
        self.cache_dir = pathlib.Path(cache_dir) if cache_dir else None
        # original cut_pad_sample_torchaudio(pad_types='repeat') == toolkit repeat_fade (confirmed with the PAFA/patch-mix goldens)
        self.cut_pad = AudioPreprocessor(self.sample_rate, AudioPreprocessConfig(desired_length_sec=self.desired_length, pad_type="repeat_fade"))

        self.all_cycles: List[Dict[str, Any]] = self._load_cycles()
        self._build_lists()

        if self.mode == "concat_single":
            self.mixing_combinations = single_label_mixing_combinations()
        else:
            self.mixing_combinations = self.ls.mixing_combinations(self.mixing_label_strategy)

        n = len(self.all_cycles)
        if self.train_flag:
            self.n_mixed = int(n * self.mixing_prob)
            self.n_ssl = int(n * self.ssl_prob) if self.enable_ssl else 0
            self.dataset_size = n + self.n_mixed + self.n_ssl
        else:
            self.n_mixed, self.n_ssl = 0, 0
            self.dataset_size = n
        logger.info(
            "PCMCLConcatDataset(%s, mode=%s, label_space=%s): cycles=%d short(<T/2)=%d patients(%s)=%d mixed=%d ssl=%d size=%d",
            split, mode, self.ls.name, n, sum(1 for c in self.all_cycles if c["length"] < self.half_target_samples),
            patient_id_mode, len(self.patient_cycles), self.n_mixed, self.n_ssl, self.dataset_size,
        )

    # ------------------------------------------------------------------ loading
    def _cache_path(self) -> Optional[pathlib.Path]:
        if self.cache_dir is None:
            return None
        key = hashlib.sha1(f"{self.index_csv.resolve()}|{self.split_file.resolve() if self.split_file else None}|{self.split}|{self.sample_rate}|{self.fade}".encode()).hexdigest()[:10]
        return self.cache_dir / f"cycles_{self.split}_{self.sample_rate}_{'fade' if self.fade else 'nofade'}_{key}.pt"

    def _load_cycles(self) -> List[Dict[str, Any]]:
        # Split filtering is delegated to IndexDataset (same split_file priority rules)
        ds = IndexDataset(index_csv=self.index_csv, data_root=self.data_root, sampling_rate=self.sample_rate,
                          sample_level="cycle", split=self.split, split_file=self.split_file, fade=self.fade)
        df = ds.df.copy()
        df["_stem"] = df["recording_path"].map(lambda p: pathlib.Path(str(p)).stem)
        df["_row"] = range(len(df))
        # Original: iterate filenames = sorted(stem), annotation row order within a file
        df = df.sort_values(["_stem", "_row"], kind="stable").reset_index(drop=True)
        sample_ids = df["sample_id"].astype(str).tolist()

        cache = self._cache_path()
        if cache is not None and cache.exists():
            try:
                blob = torch.load(cache, weights_only=False)
                if blob.get("sample_ids") == sample_ids:
                    cycles = blob["cycles"]
                    logger.info("Loaded cycle cache %s (%d)", cache, len(cycles))
                    return self._attach_ids(cycles)
                logger.warning("Cycle cache %s has sample_ids that differ from the index; rebuilding", cache)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to load cycle cache %s (%s) -- rebuilding", cache, exc)

        cycles: List[Dict[str, Any]] = []
        for stem, g in df.groupby("_stem", sort=False):
            wav_path = ds._resolve_recording_path(str(g.iloc[0]["recording_path"]))
            wav = load_audio(wav_path, target_sampling_rate=self.sample_rate, fade=self.fade)
            if wav.shape[0] > 1:
                wav = wav[0:1]
            for _, r in g.iterrows():
                audio = slice_audio(wav, float(r["t0"]), float(r["t1"]), self.sample_rate).clone()
                label4 = int(r["label"])
                cycles.append({
                    "audio": audio,                       # (1, L) original length
                    "original_label": label4,
                    "filename": stem,
                    "sample_id": str(r["sample_id"]),
                    "length": int(audio.shape[-1]),
                })
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")  # avoids races between concurrent first runs creating the cache (per-pid tmp -> atomic replace)
            torch.save({"sample_ids": sample_ids, "cycles": cycles}, tmp)
            tmp.replace(cache)
            logger.info("Saved cycle cache %s (%d)", cache, len(cycles))
        return self._attach_ids(cycles)

    def _attach_ids(self, cycles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for c in cycles:
            if self.mode == "concat_single":
                c["label"] = c["original_label"]                      # 4-class single label
            else:
                c["label"] = self.ls.to_multilabel(c["original_label"])
            # Original dataset_pcmcl.py:112 -- official mode: filename.split('.')[0] == whole file name (per recording)
            c["patient_id"] = c["filename"].split("_")[0] if self.patient_id_mode == "patient" else c["filename"]
        return cycles

    def _build_lists(self) -> None:
        n_buckets = self.ls.n_classes
        self.classwise_cycle_list: List[List[Dict[str, Any]]] = [[] for _ in range(n_buckets)]
        self.short_classwise_cycle_list: List[List[Dict[str, Any]]] = [[] for _ in range(n_buckets)]
        self.patient_cycles: Dict[str, List[Dict[str, Any]]] = {}
        for c in self.all_cycles:
            self.patient_cycles.setdefault(c["patient_id"], []).append(c)
            short = c["length"] < self.half_target_samples
            if self.mode == "concat_single":
                # ConcatenationSingleLabelDataset: each of the 4 classes goes into exactly one list
                self.classwise_cycle_list[c["original_label"]].append(c)
                if short:
                    self.short_classwise_cycle_list[c["original_label"]].append(c)
                continue
            # Containment (⊇) rule -- for ICBHI this is bit-identical to the original "0/1/2 by bit, 3 if original_label==3"
            # (only both `[0,1,1]` contains class 3's `[0,1,1]`). See the pcmcl_labelspace module docstring.
            for k in self.ls.buckets_for(c["label"]):
                self.classwise_cycle_list[k].append(c)
                if short:
                    self.short_classwise_cycle_list[k].append(c)

    # ------------------------------------------------------------------ RNG
    @property
    def rng(self) -> random.Random:
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            seed = int(torch.initial_seed()) % (2 ** 32)
            if info is not None:
                seed = (seed + 1000003 * info.id) % (2 ** 32)
            self._rng = random.Random(seed)
        return self._rng

    def reseed(self, seed: int) -> None:
        self._rng = random.Random(seed)

    # ---------------------------------------------------------------- concatenation
    def _should_apply_mixing(self, combination) -> bool:
        if not self.conditional_mixing:
            return True
        return self.rng.random() < combination[-1]

    def _generate_mixed_sample(self) -> Dict[str, Any]:
        rng = self.rng
        combination = rng.choice(self.mixing_combinations)
        if self.mode == "concat_single":
            return self._generate_mixed_single(combination)
        if not self._should_apply_mixing(combination):
            return rng.choice(self.all_cycles)
        short = self.short_classwise_cycle_list
        n_src = self.ls.sources_in(combination)
        result_label = self.ls.label_in(combination)
        if n_src == 2:
            c1, c2 = combination[0], combination[1]
            if c1 == c2:
                if len(short[c1]) >= 2:
                    a, b = rng.sample(short[c1], 2)
                    audio = _cat(a["audio"], b["audio"])
                    return {"audio": audio, "label": result_label, "is_mixed": True, "mix_info": f"concat_self_{c1}",
                            "concat_classes": [c1, c1], "concat_ids": [a["sample_id"], b["sample_id"]], "length": audio.shape[-1]}
                return rng.choice(self.all_cycles)
            if len(short[c1]) > 0 and len(short[c2]) > 0:
                a = rng.choice(short[c1])
                b = rng.choice(short[c2])
                audio = _cat(a["audio"], b["audio"])
                return {"audio": audio, "label": result_label, "is_mixed": True, "mix_info": f"concat_{c1}+{c2}",
                        "concat_classes": [c1, c2], "concat_ids": [a["sample_id"], b["sample_id"]], "length": audio.shape[-1]}
        elif n_src == 3:
            c1, c2, c3 = combination[0], combination[1], combination[2]
            if len(short[c1]) > 0 and len(short[c2]) > 0 and len(short[c3]) > 0:
                a = rng.choice(short[c1])
                b = rng.choice(short[c2])
                c = rng.choice(short[c3])
                audio = _cat(a["audio"], b["audio"], c["audio"])
                return {"audio": audio, "label": result_label, "is_mixed": True, "mix_info": f"concat_{c1}+{c2}+{c3}",
                        "concat_classes": [c1, c2, c3], "concat_ids": [a["sample_id"], b["sample_id"], c["sample_id"]], "length": audio.shape[-1]}
        return rng.choice(self.all_cycles)

    def _generate_mixed_single(self, combination) -> Dict[str, Any]:
        """ablation concat_single: 4-class dominant label (original ConcatenationSingleLabelDataset._generate_mixed_sample)."""
        rng = self.rng
        c1, c2, result, _ = combination
        short = self.short_classwise_cycle_list
        if c1 == c2:
            if len(short[c1]) >= 2:
                a, b = rng.sample(short[c1], 2)
                audio = _cat(a["audio"], b["audio"])
                return {"audio": audio, "label": result, "is_mixed": True, "mix_info": f"concat_self_{c1}",
                        "concat_ids": [a["sample_id"], b["sample_id"]], "length": audio.shape[-1]}
            return rng.choice(self.all_cycles)
        if len(short[c1]) > 0 and len(short[c2]) > 0:
            a = rng.choice(short[c1])
            b = rng.choice(short[c2])
            audio = _cat(a["audio"], b["audio"])
            return {"audio": audio, "label": result, "is_mixed": True, "mix_info": f"concat_{c1}+{c2}",
                    "concat_ids": [a["sample_id"], b["sample_id"]], "length": audio.shape[-1]}
        return rng.choice(self.all_cycles)

    # ------------------------------------------------------------------ PM
    def _or_label(self, a: Dict[str, Any], b: Dict[str, Any]) -> List[int]:
        return [max(a["label"][k], b["label"][k]) for k in range(self.ls.n_attrs)]

    def _generate_ssl_sample(self) -> Optional[Dict[str, Any]]:
        if not self.enable_ssl:
            return None
        if self.rng.random() < 0.5:
            return self._generate_positive_coherence_sample()
        if self.hard_negative_mode == "basic":
            return self._generate_basic_negative_coherence_sample()
        return self._generate_hard_negative_coherence_sample()

    def _generate_positive_coherence_sample(self) -> Optional[Dict[str, Any]]:
        rng = self.rng
        valid = [pid for pid, cs in self.patient_cycles.items() if len(cs) >= 3]
        if not valid:
            valid = [pid for pid, cs in self.patient_cycles.items() if len(cs) >= 2]
            if not valid:
                return None
        pid = rng.choice(valid)
        cs = self.patient_cycles[pid]
        # Original: `random.random() < 0.7 and len(cs) >= 3` -- random() is consumed first
        if rng.random() < 0.7 and len(cs) >= 3:
            gap_options = [(i, j) for i in range(len(cs)) for j in range(i + 2, len(cs))]
            if gap_options:
                i, j = rng.choice(gap_options)
                c1, c2 = cs[i], cs[j]
                mix_info = f"non_consecutive_{i}+{j}"
            else:
                i = rng.randint(0, len(cs) - 2)
                c1, c2 = cs[i], cs[i + 1]
                mix_info = f"consecutive_{i}+{i + 1}"
        else:
            i = rng.randint(0, len(cs) - 2)
            c1, c2 = cs[i], cs[i + 1]
            mix_info = f"consecutive_{i}+{i + 1}"
        audio = _cat(c1["audio"], c2["audio"])
        return {"audio": audio, "label": self._or_label(c1, c2), "ssl_label": 1, "ssl_task": "coherence", "is_mixed": True,
                "patient_id": pid, "mix_info": mix_info, "concat_classes": [c1["original_label"], c2["original_label"]],
                "concat_ids": [c1["sample_id"], c2["sample_id"]], "length": audio.shape[-1]}

    def _pathology_groups(self) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {k: [] for k in ("normal_only", "crackle_only", "wheeze_only", "crackle_wheeze",
                                                         "normal_crackle", "normal_wheeze", "all_types")}
        for pid, cs in self.patient_cycles.items():
            hn = any(c["label"][0] > 0 for c in cs)
            hc = any(c["label"][1] > 0 for c in cs)
            hw = any(c["label"][2] > 0 for c in cs)
            if hn and hc and hw:
                groups["all_types"].append(pid)
            elif hc and hw:
                groups["crackle_wheeze"].append(pid)
            elif hn and hc:
                groups["normal_crackle"].append(pid)
            elif hn and hw:
                groups["normal_wheeze"].append(pid)
            elif hn:
                groups["normal_only"].append(pid)
            elif hc:
                groups["crackle_only"].append(pid)
            elif hw:
                groups["wheeze_only"].append(pid)
        return groups

    def _cycles_for_profile(self, pid: str, ptype: str) -> List[Dict[str, Any]]:
        cs = self.patient_cycles[pid]
        if ptype == "normal_only":
            return [c for c in cs if c["label"][0] > 0 and c["label"][1] == 0 and c["label"][2] == 0]
        if ptype == "crackle_only":
            return [c for c in cs if c["label"][1] > 0 and c["label"][0] == 0 and c["label"][2] == 0]
        if ptype == "wheeze_only":
            return [c for c in cs if c["label"][2] > 0 and c["label"][0] == 0 and c["label"][1] == 0]
        if ptype == "crackle_wheeze":
            return [c for c in cs if c["label"][1] > 0 and c["label"][2] > 0]
        return list(cs)

    def _generate_hard_negative_coherence_sample(self) -> Optional[Dict[str, Any]]:
        rng = self.rng
        if len(self.patient_cycles) < 3:
            return self._generate_basic_negative_coherence_sample()
        groups = self._pathology_groups()
        if self.hard_negative_mode == "dict_order":
            order = list(groups.items())                         # quirk: insertion order -> effectively normal_only
        else:  # random_profile (presumed paper intent -- not confirmed by the authors)
            eligible = [(k, v) for k, v in groups.items() if len(v) >= 2]
            order = [rng.choice(eligible)] if eligible else []
        for ptype, patients in order:
            if len(patients) >= 2:
                p1, p2 = rng.sample(patients, 2)
                cs1, cs2 = self._cycles_for_profile(p1, ptype), self._cycles_for_profile(p2, ptype)
                if cs1 and cs2:
                    c1, c2 = rng.choice(cs1), rng.choice(cs2)
                    audio = _cat(c1["audio"], c2["audio"])
                    return {"audio": audio, "label": self._or_label(c1, c2), "ssl_label": 0, "ssl_task": "coherence", "is_mixed": True,
                            "patient_id": f"{p1}+{p2}", "mix_info": f"hard_negative_{ptype}",
                            "concat_classes": [c1["original_label"], c2["original_label"]],
                            "concat_ids": [c1["sample_id"], c2["sample_id"]], "length": audio.shape[-1]}
        return self._generate_basic_negative_coherence_sample()

    def _generate_basic_negative_coherence_sample(self) -> Optional[Dict[str, Any]]:
        rng = self.rng
        if len(self.patient_cycles) < 2:
            return None
        p1, p2 = rng.sample(list(self.patient_cycles.keys()), 2)
        c1, c2 = rng.choice(self.patient_cycles[p1]), rng.choice(self.patient_cycles[p2])
        audio = _cat(c1["audio"], c2["audio"])
        return {"audio": audio, "label": self._or_label(c1, c2), "ssl_label": 0, "ssl_task": "coherence", "is_mixed": True,
                "patient_id": f"{p1}+{p2}", "mix_info": "basic_negative", "concat_classes": [c1["original_label"], c2["original_label"]],
                "concat_ids": [c1["sample_id"], c2["sample_id"]], "length": audio.shape[-1]}

    # -------------------------------------------------------------- output
    def _process_audio(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.dim() > 2:
            audio = audio[0].unsqueeze(0)
        audio = self.cut_pad(audio)                       # (1, T)
        if self.output_kind == "waveform":
            return audio.squeeze(0).float()               # (T,)
        from rsc_toolkit.processors.mel import generate_fbank

        # Original generate_fbank (kaldi 128 mel, mean -4.2677393 / std 4.5689974, /(2*std)) -> (frames, n_mels); toolkit batch convention (n_mels, frames)
        return generate_fbank(audio, sampling_rate=self.sample_rate, n_mels=self.n_mels).transpose(0, 1).contiguous()

    def get_sample(self, index: int) -> Dict[str, Any]:
        """Branching of the original __getitem__ (raw sample dict + ssl info). Public for golden comparison."""
        n = len(self.all_cycles)
        ssl_label, ssl_task = -1, "none"
        if index < n:
            sample, is_mixed = self.all_cycles[index], False
        elif index < n + self.n_mixed:
            sample, is_mixed = self._generate_mixed_sample(), True
        else:
            sample = self._generate_ssl_sample()
            if sample is None:
                sample, is_mixed = self.all_cycles[index % n], False
            else:
                is_mixed = True
                ssl_label = sample.get("ssl_label", -1)
                ssl_task = sample.get("ssl_task", "none")
        return {"sample": sample, "is_mixed": is_mixed, "ssl_label": ssl_label, "ssl_task": ssl_task}

    def __getitem__(self, index: int):
        info = self.get_sample(int(index))
        sample = info["sample"]
        x = self._process_audio(sample["audio"])
        if self.mode == "concat_single":
            y = torch.tensor(int(sample["label"]), dtype=torch.long)
        else:
            y = torch.tensor(sample["label"], dtype=torch.float32)
        return x, y, torch.tensor(int(info["ssl_label"]), dtype=torch.long)

    def __len__(self) -> int:
        return self.dataset_size


from rsc_toolkit.datasets import collate as _collate  # single collate implementation

# (x, y(3-label float | 4-class long) stack, ssl_label long -1/0/1 stack). PLAdapter passes batch[2] as meta_labels.
pcmcl_train_collate = _collate.pcmcl_train_collate


# ====================================================================== datamodule
from rsc_toolkit.trainer.data.index_datamodule import IndexDataModule  # noqa: E402  (lightning dependency only here)


class PCMCLDataModule(IndexDataModule):
    meta_kind = "pcmcl_ssl_label"  # train batch[2] = ssl_label long ∈ {-1,0,1} -- ast_pcmcl/beats_pcmcl only
    """
    train = PCMCLConcatDataset (dynamic concatenation, PM), val/test = the parent (IndexDataModule) IndexDataset path
    (desired_length_sec, pad_type=repeat_fade -> test cycles cut_pad individually, waveform/fbank via features.kind).
    The train output format follows features.kind: waveform -> (L,), fbank/mel -> (n_mels, frames).
    """

    def __init__(self, *args, pcmcl: Optional[dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pcmcl_cfg = dict(pcmcl or {})

    def _build_train_dataset(self):
        kind = self.features_cfg.kind
        if kind == "waveform":
            output_kind = "waveform"
        elif kind in {"mel", "fbank"}:
            output_kind = "fbank"
        else:
            raise ValueError(f"pcmcl datamodule supports only features.kind waveform|fbank: {kind!r}")
        if len(self.train_splits) != 1:
            raise ValueError("pcmcl train_splits must contain exactly one split")
        cfg = dict(self.pcmcl_cfg)
        cache_dir = cfg.pop("cache_dir", "outputs/pcmcl/cache")
        return PCMCLConcatDataset(
            index_csv=self.index_csv, data_root=self.raw_data_directory, split_file=self.split_file,
            split=self.train_splits[0], train=True, sampling_rate=self.sampling_rate,
            desired_length_sec=self.desired_length_sec, output_kind=output_kind, n_mels=self.features_cfg.n_mels,
            cache_dir=self._resolve_path(cache_dir) if cache_dir else None, fade=self.fade, **cfg,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if self.index_csv is None:
            self.index_csv = self._default_index_path()
        if not self.features_cfg.enabled:
            raise ValueError("pcmcl datamodule requires features.enabled=true (waveform or fbank)")
        if stage in (None, "fit"):
            self.train_ds = self._build_train_dataset()
            self.val_ds = self._wrap_with_features(self._build_index_dataset(splits=self.val_splits), feature_transform=None)
            if self.feature_precompute and self._is_global_zero():
                self._warmup_feature_cache(self.val_ds, split_name="val")  # type: ignore[arg-type]
        if stage in (None, "test"):
            super().setup("test")

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("train_ds is None. Did you forget to call setup('fit')?")
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=self.shuffle_train, drop_last=self.drop_last_train,
            num_workers=self.num_workers, collate_fn=pcmcl_train_collate, pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )
