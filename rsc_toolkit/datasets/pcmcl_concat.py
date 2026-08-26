"""
PC-MCL(arXiv 2601.17080; 공식 코드 wa976/PC-MCL util/dataset_pcmcl.py ConcatenationMultilabelDataset) 의
학습 데이터셋 — 원 길이 호흡 사이클 보관 + __getitem__ 시점 동적 연결(11 조합) + PM(환자 매칭) 쌍 생성.

공식 코드 실체 (P0, 전부 기본값으로 유지):
- 공식 60/40 스플릿 train 파일을 stem 정렬 순서로 돌며 사이클을 **원 길이 그대로** 추출
  (torchaudio.load → 16 kHz 리샘플 → 녹음 전체에 fade in/out sr/16 → [start,end) 절단; icbhi_util.py get_individual_cycles_torchaudio_beats)
  → 툴킷 datasets/icbhi/processing.py load_audio(fade 1/16) + slice_audio 와 동일.
- 클래스별 리스트(normal/crackle/wheeze/both; both 는 crackle·wheeze 리스트에도 들어감) 와
  '짧은 사이클'(길이 < T/2) 리스트, 환자별 사이클 dict(삽입 순서).
- 에폭당 크기 = N + int(N·mixing_prob) + int(N·ssl_prob). index < N 원본, 다음 구간 연결 샘플, 마지막 구간 PM 샘플.
- 연결: 11 개 additive 조합에서 균등 선택 → 각 클래스의 **짧은** 사이클만 무작위(환자 무관) 로 뽑아 torch.cat →
  T 초 cut_pad(앞 절단 / repeat + fade_out sr/16). 조건(짧은 사이클 부족) 미충족 시 원본 사이클 무작위 반환.
- PM: 50% 양성 = 같은 patient_id 의 두 사이클(70% 비연속 gap≥2 / 30% 연속, 길이 제한 없음, 라벨 OR),
  50% 하드 음성 = 병리 프로필이 같은 두 '환자' — dict 삽입 순서(normal_only 부터) 로 첫 ≥2 그룹에서 즉시 반환하는 quirk.
  PM 샘플의 주 라벨(OR) 은 주 BCE 손실에도 포함된다(마스킹 없음).
- **patient_id quirk**: 공식 스플릿 모드에서 patient_id = 파일명 전체(=녹음). 논문 의도(환자 번호) 는 patient_id_mode="patient".
- 테스트(train=False) 는 원본 사이클만 (cut_pad 후 단독 입력) — 툴킷에서는 기존 IndexDataset(repeat_fade 10 s) 를 재사용한다.

난수: 원본은 python `random` 모듈을 __getitem__ 에서 쓴다(num_workers=0 기본 → 단일 스트림). 여기서는 전용
random.Random 인스턴스를 쓰며 호출 순서를 원본과 같게 유지했으므로 rng_seed 를 주고 num_workers=0 으로 돌리면
같은 seed 에서 같은 샘플 열이 나온다(골든으로 고정). num_workers>0 이면 워커마다 torch.initial_seed()(Lightning
seed_everything(workers=True) 가 파생) 로 시드된 별도 스트림 → 원본과 '동일 분포' 이지 동일 열은 아니다.
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
    """원본 _convert_to_multilabel: 0→[1,0,0], 1→[0,1,0], 2→[0,0,1], 3→[0,1,1]."""
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
    """원본 _get_multilabel_mixing_combinations 와 같은 순서·내용 (random.choice 인덱스가 같아야 한다)."""
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
    """원본 _concatenate_two/three_audios: (1, L) 로 맞춰 시간축 연결 (다채널이면 첫 채널)."""
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
        # ---- 원본 인자 ----
        mode: str = "default",                  # default(concat+multi[+PM]) | concat_single | noconcat_multi
        mixing_prob: float = 1.0,               # 논문 '원본당 증강 1개' = 1.0 (argparse 기본 0.5)
        conditional_mixing: bool = False,       # 원본 --conditional_mixing (store_true; 켜면 random.random() 한 번 더 소비, 확률은 전부 1.0)
        mixing_label_strategy: str = "additive",
        enable_ssl: bool = True,
        ssl_prob: float = 0.3,
        ssl_method: str = "coherence",
        # ---- 논문-코드 불일치 플래그 (기본 = 코드) ----
        patient_id_mode: str = "recording",     # recording(공식 모드 quirk: 파일명 전체) | patient(환자 번호 split('_')[0])
        hard_negative_mode: str = "dict_order", # dict_order(quirk: 첫 ≥2 그룹) | random_profile(≥2 그룹 중 무작위) | basic(임의 다른 환자)
        # ---- 툴킷 ----
        rng_seed: Optional[int] = None,         # None: 워커별 torch.initial_seed() 파생 (학습). 정수: 고정 스트림 (골든/테스트)
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
        if ssl_method != "coherence":
            raise NotImplementedError("ssl_method 는 논문이 쓰는 'coherence'(PM) 만 이식했다 (concat_single SSL 은 범위 밖)")

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
        # 원본 cut_pad_sample_torchaudio(pad_types='repeat') == 툴킷 repeat_fade (PAFA/patch-mix 골든으로 확인)
        self.cut_pad = AudioPreprocessor(self.sample_rate, AudioPreprocessConfig(desired_length_sec=self.desired_length, pad_type="repeat_fade"))

        self.all_cycles: List[Dict[str, Any]] = self._load_cycles()
        self._build_lists()

        if self.mode == "concat_single":
            self.mixing_combinations = single_label_mixing_combinations()
        else:
            self.mixing_combinations = multilabel_mixing_combinations(self.mixing_label_strategy)

        n = len(self.all_cycles)
        if self.train_flag:
            self.n_mixed = int(n * self.mixing_prob)
            self.n_ssl = int(n * self.ssl_prob) if self.enable_ssl else 0
            self.dataset_size = n + self.n_mixed + self.n_ssl
        else:
            self.n_mixed, self.n_ssl = 0, 0
            self.dataset_size = n
        logger.info(
            "PCMCLConcatDataset(%s, mode=%s): cycles=%d short(<T/2)=%d patients(%s)=%d mixed=%d ssl=%d size=%d",
            split, mode, n, sum(1 for c in self.all_cycles if c["length"] < self.half_target_samples),
            patient_id_mode, len(self.patient_cycles), self.n_mixed, self.n_ssl, self.dataset_size,
        )

    # ------------------------------------------------------------------ 적재
    def _cache_path(self) -> Optional[pathlib.Path]:
        if self.cache_dir is None:
            return None
        key = hashlib.sha1(f"{self.index_csv.resolve()}|{self.split_file.resolve() if self.split_file else None}|{self.split}|{self.sample_rate}|{self.fade}".encode()).hexdigest()[:10]
        return self.cache_dir / f"cycles_{self.split}_{self.sample_rate}_{'fade' if self.fade else 'nofade'}_{key}.pt"

    def _load_cycles(self) -> List[Dict[str, Any]]:
        # 스플릿 적용은 IndexDataset 에 맡긴다 (split_file 우선순위 규칙 동일)
        ds = IndexDataset(index_csv=self.index_csv, data_root=self.data_root, sampling_rate=self.sample_rate,
                          sample_level="cycle", split=self.split, split_file=self.split_file, fade=self.fade)
        df = ds.df.copy()
        df["_stem"] = df["recording_path"].map(lambda p: pathlib.Path(str(p)).stem)
        df["_row"] = range(len(df))
        # 원본: filenames = sorted(stem) 순회, 파일 안에서는 주석 행 순서
        df = df.sort_values(["_stem", "_row"], kind="stable").reset_index(drop=True)
        sample_ids = df["sample_id"].astype(str).tolist()

        cache = self._cache_path()
        if cache is not None and cache.exists():
            try:
                blob = torch.load(cache, weights_only=False)
                if blob.get("sample_ids") == sample_ids:
                    cycles = blob["cycles"]
                    logger.info("사이클 캐시 적재 %s (%d)", cache, len(cycles))
                    return self._attach_ids(cycles)
                logger.warning("사이클 캐시 %s 의 sample_id 가 인덱스와 달라 다시 만든다", cache)
            except Exception as exc:  # pragma: no cover
                logger.warning("사이클 캐시 %s 적재 실패(%s) — 다시 만든다", cache, exc)

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
                    "audio": audio,                       # (1, L) 원 길이
                    "original_label": label4,
                    "filename": stem,
                    "sample_id": str(r["sample_id"]),
                    "length": int(audio.shape[-1]),
                })
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")  # 동시 첫 run 들의 캐시 생성 경합 방지 (pid 별 tmp → 원자적 replace)
            torch.save({"sample_ids": sample_ids, "cycles": cycles}, tmp)
            tmp.replace(cache)
            logger.info("사이클 캐시 저장 %s (%d)", cache, len(cycles))
        return self._attach_ids(cycles)

    def _attach_ids(self, cycles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for c in cycles:
            if self.mode == "concat_single":
                c["label"] = c["original_label"]                      # 4-class single label
            else:
                c["label"] = convert_to_multilabel(c["original_label"])
            # 원본 dataset_pcmcl.py:112 — 공식 모드: filename.split('.')[0] == 파일명 전체(녹음 단위)
            c["patient_id"] = c["filename"].split("_")[0] if self.patient_id_mode == "patient" else c["filename"]
        return cycles

    def _build_lists(self) -> None:
        self.classwise_cycle_list: List[List[Dict[str, Any]]] = [[] for _ in range(4)]
        self.short_classwise_cycle_list: List[List[Dict[str, Any]]] = [[] for _ in range(4)]
        self.patient_cycles: Dict[str, List[Dict[str, Any]]] = {}
        for c in self.all_cycles:
            self.patient_cycles.setdefault(c["patient_id"], []).append(c)
            short = c["length"] < self.half_target_samples
            if self.mode == "concat_single":
                # ConcatenationSingleLabelDataset: 4-class 각각 한 리스트에만
                self.classwise_cycle_list[c["original_label"]].append(c)
                if short:
                    self.short_classwise_cycle_list[c["original_label"]].append(c)
                continue
            ml = c["label"]
            for k in range(3):
                if ml[k] > 0:
                    self.classwise_cycle_list[k].append(c)
                    if short:
                        self.short_classwise_cycle_list[k].append(c)
            if c["original_label"] == 3:
                self.classwise_cycle_list[3].append(c)
                if short:
                    self.short_classwise_cycle_list[3].append(c)

    # ------------------------------------------------------------------ 난수
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

    # ---------------------------------------------------------------- 연결
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
        if len(combination) == 6:
            c1, c2, rn, rc, rw, _ = combination
            if c1 == c2:
                if len(short[c1]) >= 2:
                    a, b = rng.sample(short[c1], 2)
                    audio = _cat(a["audio"], b["audio"])
                    return {"audio": audio, "label": [rn, rc, rw], "is_mixed": True, "mix_info": f"concat_self_{c1}",
                            "concat_classes": [c1, c1], "concat_ids": [a["sample_id"], b["sample_id"]], "length": audio.shape[-1]}
                return rng.choice(self.all_cycles)
            if len(short[c1]) > 0 and len(short[c2]) > 0:
                a = rng.choice(short[c1])
                b = rng.choice(short[c2])
                audio = _cat(a["audio"], b["audio"])
                return {"audio": audio, "label": [rn, rc, rw], "is_mixed": True, "mix_info": f"concat_{c1}+{c2}",
                        "concat_classes": [c1, c2], "concat_ids": [a["sample_id"], b["sample_id"]], "length": audio.shape[-1]}
        elif len(combination) == 7:
            c1, c2, c3, rn, rc, rw, _ = combination
            if len(short[c1]) > 0 and len(short[c2]) > 0 and len(short[c3]) > 0:
                a = rng.choice(short[c1])
                b = rng.choice(short[c2])
                c = rng.choice(short[c3])
                audio = _cat(a["audio"], b["audio"], c["audio"])
                return {"audio": audio, "label": [rn, rc, rw], "is_mixed": True, "mix_info": f"concat_{c1}+{c2}+{c3}",
                        "concat_classes": [c1, c2, c3], "concat_ids": [a["sample_id"], b["sample_id"], c["sample_id"]], "length": audio.shape[-1]}
        return rng.choice(self.all_cycles)

    def _generate_mixed_single(self, combination) -> Dict[str, Any]:
        """ablation concat_single: 4-class dominant 라벨 (원본 ConcatenationSingleLabelDataset._generate_mixed_sample)."""
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
    @staticmethod
    def _or_label(a: Dict[str, Any], b: Dict[str, Any]) -> List[int]:
        return [max(a["label"][k], b["label"][k]) for k in range(3)]

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
        # 원본: `random.random() < 0.7 and len(cs) >= 3` — random() 이 먼저 소비된다
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
            order = list(groups.items())                         # quirk: 삽입 순서 → 사실상 normal_only
        else:  # random_profile (논문 의도 추정 — 저자 확인 전)
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

    # -------------------------------------------------------------- 출력
    def _process_audio(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.dim() > 2:
            audio = audio[0].unsqueeze(0)
        audio = self.cut_pad(audio)                       # (1, T)
        if self.output_kind == "waveform":
            return audio.squeeze(0).float()               # (T,)
        from rsc_toolkit.processors.mel import generate_fbank

        # 원본 generate_fbank (kaldi 128 mel, mean -4.2677393 / std 4.5689974, /(2·std)) → (frames, n_mels); 툴킷 배치 규약 (n_mels, frames)
        return generate_fbank(audio, sampling_rate=self.sample_rate, n_mels=self.n_mels).transpose(0, 1).contiguous()

    def get_sample(self, index: int) -> Dict[str, Any]:
        """원본 __getitem__ 의 분기 (가공 전 샘플 dict + ssl 정보). 골든 대조용으로 공개."""
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


def pcmcl_train_collate(batch: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """(x, y(3-label float | 4-class long), ssl_label long -1/0/1). PLAdapter 는 batch[2] 를 meta_labels 로 넘긴다."""
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    ssl = torch.stack([b[2] for b in batch], dim=0)
    return x, y, ssl


# ====================================================================== datamodule
from rsc_toolkit.trainer.data.index_datamodule import IndexDataModule  # noqa: E402  (lightning 의존은 여기서만)


class PCMCLDataModule(IndexDataModule):
    """
    train = PCMCLConcatDataset(동적 연결·PM), val/test = 부모(IndexDataModule) 의 IndexDataset 경로
    (desired_length_sec·pad_type=repeat_fade → 테스트 사이클 단독 cut_pad, features.kind 로 waveform/fbank).
    train 의 출력 형식은 features.kind 를 따른다: waveform → (L,), fbank/mel → (n_mels, frames).
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
            raise ValueError(f"pcmcl 데이터모듈은 features.kind waveform|fbank 만 지원한다: {kind!r}")
        if len(self.train_splits) != 1:
            raise ValueError("pcmcl train_splits 는 하나여야 한다")
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
            raise ValueError("pcmcl 데이터모듈은 features.enabled=true (waveform 또는 fbank) 가 필요하다")
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
