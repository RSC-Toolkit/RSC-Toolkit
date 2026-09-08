"""
SPRSound → ICBHI 4-class OOD 평가 전용 로더 (BTS-CARD; 학습 미사용).

**비표준 사용 주의**: SPRSound 의 원래 목적은 7-class event 분류(2022 inter-patient test)다. BTS-CARD 논문은
2022 inter-patient valid 세트(1,429 event)를 ICBHI 4-class 로 병합해 OOD 테스트로만 쓴다. 여기서 만드는 것은
그 파생 뷰이며 원본 데이터·인덱스는 바꾸지 않는다.

라벨 병합 (공식 노트북 data/Preprocessing_SPRSound.ipynb cell 2/5):
  Normal → 0, Fine Crackle / Coarse Crackle → 1(Crackle), Wheeze → 2, Both(Wheeze+Crackle) → 3,
  Rhonchi / Stridor → 코드는 **제거**(map_annotation_type → None; 기본 rhonchi_stridor="drop")
                       논문 §3.1 은 wheeze 병합 서술 → rhonchi_stridor="wheeze" 플래그.
  2022 inter-test 에는 Rhonchi/Stridor 가 0건이라 OOD 셋은 어느 쪽이든 1040/83/305/1 로 같다.
  (노트북 quirk: 'wheeze + crackle' 분기는 원본 문자열 'Wheeze+Crackle' 과 매치되지 않아 else 로 통과한 뒤
   _get_lungsound_label('Wheeze+Crackle') → 3 으로 동작 — 결과는 같다.)

메타 3필드 (노트북 cell 7): [age(float, 파일명), gender(int; 0=male, 1=female), loc p1..p4 → 0..3]
문장 템플릿 (util/sprsound_util.py): "This sound was recorded from the {loc} of a {sex} pediatric patient."
  loc: 0 left posterior / 1 left lateral / 2 right posterior / 3 right lateral (SPRSound README 정의와 일치;
  논문 Table 1 의 'L/R Anterior' 표기가 데이터셋 문서와 어긋남). age ≥ 19 만 'adult' — valid 는 전원 소아.
  CLAP·tiny-bert 모두 같은 문장, 마스킹 없음 (sprsound_dataset.py:35-41).

두 경로:
  (A) 저자 배포 valid_spr.pt (Google Drive 1zGR…; pretrained_models/upstream/btscard/valid_spr.pt)
      항목 = (log-mel (1,1001,64), label 0-3, meta3, 파일 경로 문자열)
  (B) 툴킷 자체 파이프라인: configs/data/sprsound_btscard.yaml 인자로 IndexDataset( 48 kHz, 녹음 전체 fade in/out
      sr/16 → 구간 절단 → 8 s repeat 패딩 + fade_out(pad_type repeat_fade) → CLAPProcessor log-mel) 의 test 셋을
      4-class 로 변환. 노트북 cell 5 순서와 동일. 기존 sprsound.yaml(pad_type repeat, fade_out 없음)과는 다르다.
  둘의 특징·라벨·문장 일치는 scripts/reproduction/check_btscard_sprsound_parity.py (tests/golden/btscard/sprsound_parity.json) 로 기록한다:
  1,429 event 전수 라벨·meta3·문장 일치, log-mel max|diff| 6.7e-3 (24건 > 1e-3; 비트 일치 아님 — **원인 미확인**(추정: 저자 환경 torch 2.0.1/Windows 의
  Resample·numpy 수치 차이; 실측 미수행). 구간 산술은 _NotebookSliceIndexDataset 로 노트북과 동일하게 맞춤).
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
    """7-class 이름 → ICBHI 4-class id. Rhonchi/Stridor 는 drop→None, wheeze→2."""
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
        if rhonchi_stridor == "wheeze":
            return 2
        raise ValueError(f"rhonchi_stridor must be drop|wheeze, got {rhonchi_stridor!r}")
    raise KeyError(f"알 수 없는 SPRSound 라벨 {label!r}")


def generate_meta_description_sprsound(age: str, sex: str, loc: str) -> str:
    """util/sprsound_util.py:1-24 그대로."""
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
    """util/sprsound_util.py get_meta_text_descriptions_sprsound: [age, sex, loc] → 문장."""
    age, sex, loc = int(meta3[0]), int(meta3[1]), int(meta3[2])
    age_str = "adult" if age >= 19 else "pediatric"
    sex_str = "male" if sex == 0 else "female"
    return generate_meta_description_sprsound(age_str, sex_str, LOC_ID_TO_STR[loc])


def meta_fields_from_meta7_sprsound(meta7):
    """
    SPRSound 방언: BTS 계열의 7필드 벡터에서 age/sex/loc 만 읽어 문장 필드로 만든다.

    ICBHI 판(`meta_fields_from_meta7`)과 두 가지가 다르다:
      · loc 어휘가 SPRSound 의 p1~p4(LOC_ID_TO_STR) 다 — ICBHI 흉부 위치 7종을 쓰면 없는 위치를 지어내게 된다.
      · dev 를 쓰지 않는다(None). SPRSound 는 전 8,085 이벤트가 단일 청진기(Yunting2) 라 구분 정보가 없다.
    나머지 세 자리(BMI·체중·신장)는 -1(결측)이고 여기서도 읽지 않는다.
    """
    age, sex, loc = int(meta7[0]), int(meta7[1]), int(meta7[5])
    return {
        "age": "adult" if age >= 19 else "pediatric",
        "sex": "male" if sex == 0 else "female",
        "loc": LOC_ID_TO_STR[loc],
        "dev": None,
    }


def meta3_from_index_meta(meta: Dict[str, Any]) -> List[float]:
    """툴킷 SPRSound 인덱스 meta dict → 노트북 cell 7 의 [age, gender, loc_id]."""
    loc = meta.get("location_id")
    if loc is None:
        loc = LOC_CODE_TO_ID[str(meta["recording_location"])]
    return [float(meta["age"]), float(int(float(meta["gender"]))), float(int(loc))]


from rsc_toolkit.datasets import collate as _collate  # P2-6: collate 단일 구현

# (x, y, meta7 stack)
sprsound_ood_collate = _collate.stack_meta_collate


# --------------------------------------------------------------------------- (A) 저자 배포
class SPRSoundOODAuthorDataset(Dataset):
    """valid_spr.pt 로더. 항목 → (feat (1,1001,64), label, meta3 float(3), recording_id)."""

    def __init__(self, path: str = AUTHOR_PT_DEFAULT):
        self.path = pathlib.Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(f"저자 valid_spr.pt 가 없다: {self.path} (README Google Drive 1zGR…)")
        self.items = torch.load(self.path, map_location="cpu", weights_only=False)
        logger.info("SPRSound 저자 OOD 텐서 로드: %s (%d events)", self.path, len(self.items))

    def __len__(self):
        return len(self.items)

    @staticmethod
    def recording_id(path_str: str) -> str:
        # 노트북 Windows 경로 ('...\\40512331_8.1_1_p1_3548.wav')
        return path_str.replace("\\", "/").rsplit("/", 1)[-1].replace(".wav", "")

    def __getitem__(self, i):
        feat, label, meta, rid = self.items[i]
        return feat, int(label), meta.float(), self.recording_id(rid)


# --------------------------------------------------------------------------- (B) 툴킷 파이프라인
class SPRSoundOOD4ClassDataset(Dataset):
    """
    IndexFeatureDataset(툴킷 SPRSound test split, CLAP 특징) → 4-class 파생 뷰.
    라벨 이름은 인덱스 label id 가 아니라 번들 metadata.csv(Filename, Start, End → Label)에서 sample_id 로 찾는다
    (LabelEncoder 의 id 순서에 의존하지 않기 위해).
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
                raise KeyError(f"metadata.csv 에서 {sid} 의 라벨을 찾지 못했다")
            label7 = name_by_sid[sid]
            y4 = sprsound_label_to_4class(label7, rhonchi_stridor)
            if y4 is None:
                dropped += 1
                continue
            meta = row.meta
            if isinstance(meta, str):
                meta = json.loads(meta)
            self.rows.append((i, y4, meta3_from_index_meta(meta), sid, label7))
        logger.info("SPRSound OOD 4-class 뷰: %d events (Rhonchi/Stridor %s: %d)", len(self.rows), rhonchi_stridor, dropped)

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
    IndexDataset 위에서 노트북 cell 5 의 구간 산술을 그대로 재현한다:
      start = int(float(ms) * 0.001 * sr) — 인덱스 CSV 의 t0(초, float) 만 쓰면 int(9.274 * 48000) = 445151 처럼 1 샘플이
      어긋나는 event 가 있다(대조 시 73/1429 건 log-mel max|diff| > 1e-3). ms 정수는 round(t0 * 1000) 으로 복원한다.
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
    IndexDataset(48 kHz, fade, repeat_fade) 항목 → HF ClapProcessor log-mel (1, 1001, 64), FeatureCache 캐시.
    processors/clap.py CLAPProcessor 와 같은 프로세서·캐시 키(model_name, sampling_rate)를 쓰되 입력을 numpy 1-D 로 넘긴다
    (transformers 5.15.1 의 ClapFeatureExtractor 는 torch 1-D 텐서를 2-D 로 취급해 MemoryError 를 낸다 — 공용 파일 미수정).
    """

    def __init__(self, index_ds, sampling_rate: int, model_name: str, cache_dir: pathlib.Path, audio_chain=None):
        from transformers import ClapProcessor

        from rsc_toolkit.datasets.feature_cache import FeatureCache

        self.index_ds = index_ds
        self.processor = ClapProcessor.from_pretrained(model_name, sampling_rate=sampling_rate)
        self.cache = FeatureCache(cache_dir)
        # audio_chain: 이 경로의 앞단 전처리(48 kHz·fade·repeat_fade 8 s·노트북식 ms 반올림 절단)를 체인 대장에 남긴다
        # (dataset_name="sprsound_ood" — 툴킷 sprsound_bts 경로와는 캐시 dir 도, 대장 키도 다르다).
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
    """configs/data/sprsound_btscard.yaml 의 전처리 인자로 IndexDataset(test = 2022 inter-patient) 을 세워 4-class 뷰로."""
    from omegaconf import OmegaConf

    from rsc_toolkit.datasets.index_dataset import IndexDataset
    from rsc_toolkit.processors.audio import AudioPreprocessConfig, AudioPreprocessor

    from rsc_toolkit.paths import project_root_from

    root = project_root_from(project_root)  # 인자 → RSCTK_PROJECT_ROOT → cwd(저장소 표지) → 개발 checkout (설치본의 parents[2] 추정 대체)
    cfg = OmegaConf.to_container(OmegaConf.load(root / data_config), resolve=False)["params"]
    data_dir = root / cfg["data_dir"]
    sr = int(cfg["sampling_rate"])
    pre = AudioPreprocessor(sr, AudioPreprocessConfig(float(cfg["desired_length_sec"]), str(cfg["pad_type"])))
    splits = list(cfg["test_splits"])
    assert len(splits) == 1, "SPRSound OOD 는 단일 test split 만 지원"
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
        "fade": bool(cfg.get("fade", True)), "time_unit": "ood_ms_round",  # _NotebookSliceIndexDataset 의 절단 규칙
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
