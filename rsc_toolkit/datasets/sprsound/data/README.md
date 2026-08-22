# SPRSound 번들 데이터

이 디렉토리는 패키지에 함께 배포되는 SPRSound 보조 파일을 둔다.
원본 오디오는 여기 두지 않는다 — `rsctk datasets download sprsound --data_dir <경로>` 로
공식 GitHub 저장소(SJTU-YONGFU-RESEARCH-GRP/SPRSound)에서 받는다.

## 파일

- `metadata.csv` — 인덱스 생성에 쓰는 이벤트 단위 어노테이션 표.
  `generate_annotation.py` 로 원본 json 어노테이션에서 만들었다:
  - train: `Classification/train_classification_json/*.json`
  - valid: `Classification/valid_classification_json/2022/inter_test_json/*.json` (2022 inter-test 만)
  - `start`/`end` 를 ms → 초로 변환, `Wheeze+Crackle` 라벨을 `Both` 로 통합
  - 컬럼: Filename, Patient_number, Age, Gender, Recording_location, Recording_number,
    Record_annotation, Start, End, Label, Split
- `generate_annotation.py` — 위 `metadata.csv` 를 만든 스크립트 (출처 기록용).
  `Classification/` 이 현재 디렉토리에 있다는 전제로 짜여 있으므로 재생성 시 경로를 맞춰 실행한다.

## 참고

- 상류 저장소는 main 브랜치 zip 으로 받으므로 시점에 따라 내용이 늘어날 수 있다.
  번들 `metadata.csv` 는 train + 2022 inter-test 범위만 덮는다.
- 이전에 있던 `get_data.py`(gdown 으로 Google Drive 미러에서 받는 스크립트)는 제거했다.
  공식 다운로더가 같은 일을 하고, gdown 은 프로젝트 의존성이 아니며, 미러의 출처가 불분명했다.
