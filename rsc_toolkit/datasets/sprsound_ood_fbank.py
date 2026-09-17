"""
QLung 의 SPRSound OOD 평가 뷰 — 7-class 이벤트 → ICBHI 4-class 병합(Rhonchi·Stridor 제외), 16 kHz kaldi fbank 128×798 (ICBHI 학습 모델 제로샷 평가 전용).

논문 3.1: SPRSound 를 Normal 6,199 / Crackle 1,044 (Fine+Coarse) / Wheeze 811 (Wheeze+Rhonchi+Stridor) / Both 31 = 8,085 로 병합
  → 툴킷 인덱스(.datasets/sprsound/index/index_cycle.csv) 의 train 6,656 + inter-patient test 1,429 와 클래스별로 정확히 일치.
  평가 범위(8,085 전체 vs inter-test 1,429)는 논문 미명시 → 둘 다 집계 (scripts/reproduction/eval_qlung_ood.py).
  **툴킷 4-class 뷰는 Rhonchi 39·Stridor 15 를 wheeze 에 넣지 않고 제외한다**(2026-09-16 사용자 규칙) → 6,199 / 1,044 / 757 / 31 = 8,031;
  inter-test 1,429 는 두 클래스가 0 건이라 동일(1,040 / 83 / 305 / 1). 옛 all-8,085 집계(ood.json 'all')는 병합 기준이라 비교 불가.
저장소에 SPRSound 코드가 없어 전처리는 **가정**(플래그, configs/data/sprsound_qlung_ood.yaml):
  - 16 kHz 리샘플(녹음 전체, 8 kHz 원음), 녹음 fade in/out sr/16, 구간 절단(초 단위 int(t·sr)), 8 s repeat_fade — ICBHI 와 동일 체인
    (논문 3.1 '모든 실험 8 s 표준화' 는 명시; 리샘플·패딩·정규화 상수는 미명시 → ICBHI 와 같다고 가정)
  - kaldi fbank 128 mel / 10 ms / AudioSet 고정 mean·std 정규화 (ICBHI AST 와 동일)
  - 피처 캐시는 별도 디렉터리(downloaded_datasets/features_cache_qlung_ood) — 캐시 키가 프로세서 설정만 반영해 다른 전처리 체인과 섞이지 않도록.
라벨 병합은 lungmix_data.SPRSOUND_7_TO_4 (인덱스 id 기반; 번들 metadata.csv 개수로 검증됨) 를 재사용한다.
"""
from __future__ import annotations

import pathlib
from typing import Dict, List, Optional, Sequence

from omegaconf import OmegaConf

from rsc_toolkit.datasets.lungmix_data import SPRSOUND_7_TO_4, SPRSOUND_EVENT_ID_TO_NAME, SPRSOUND_NAME_TO_4CLASS  # noqa: F401
from rsc_toolkit.datasets.qlung_data import QLungDataModule

DATA_CONFIG_DEFAULT = "rsc_toolkit/configs/data/sprsound_qlung_ood.yaml"
CLASS_NAMES_4 = ("normal", "crackle", "wheeze", "both")
# 논문 3.1 병합 분포 (train 6,656 + inter-test 1,429; Rhonchi·Stridor → wheeze) — 논문 수치 기록용, 툴킷 뷰와 다르다
PAPER_MERGED_COUNTS = (6199, 1044, 811, 31)
# 툴킷 4-class 뷰 (Rhonchi 39·Stridor 15 제외 → 8,031 이벤트)
TOOLKIT_4CLASS_COUNTS = (6199, 1044, 757, 31)


def sprsound_label7_to_4(label_id: int) -> Optional[int]:
    """툴킷 SPRSound 인덱스 라벨 id(0 Normal/1 Fine/2 Wheeze/3 Coarse/4 Both/5 Rhonchi/6 Stridor) → 4-class. 제외 클래스(5·6)는 None."""
    return SPRSOUND_7_TO_4.get(int(label_id))


def build_sprsound_ood_datamodule(splits: Sequence[str], data_config: str = DATA_CONFIG_DEFAULT,
                                  project_root: Optional[str] = None, overrides: Optional[Dict] = None) -> QLungDataModule:
    """
    configs/data/sprsound_qlung_ood.yaml 의 params 로 QLungDataModule 을 만들되 val/test split 을 `splits` 로 바꾼다
    (예: ["test"] = inter-test 1,429, ["train", "test"] = 8,085). setup("test") 후 test_dataloader() 로 (x, y4, aqs) 배치.
    """
    from rsc_toolkit.paths import project_root_from

    root = project_root_from(project_root)  # 인자 → RSCTK_PROJECT_ROOT → cwd(저장소 표지) → 개발 checkout
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
    """QLungAQSIndexDataset(또는 ConcatDataset) 의 4-class 라벨 개수."""
    from torch.utils.data import ConcatDataset

    if isinstance(ds, ConcatDataset):
        out = [0, 0, 0, 0]
        for sub in ds.datasets:
            for i, c in enumerate(class_counts_4(sub)):
                out[i] += c
        return out
    counts = ds.df["label"].value_counts().to_dict()
    return [int(counts.get(i, 0)) for i in range(4)]
