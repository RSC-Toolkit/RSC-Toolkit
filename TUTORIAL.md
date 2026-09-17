# RSC-Toolkit 튜토리얼 (최소 경로)

노트북 판 [`notebooks/tutorial.ipynb`](notebooks/tutorial.ipynb) 와 같은 내용을 셸 명령으로 적었다.
필요한 것: Python ≥ 3.9(권장 3.11). Part A 는 CPU 에서 수 분. Part B 는 GPU 1장, 디스크 약 8 GB(ICBHI zip 1.98 GB + 추출 3.9 GB + 피처 캐시 2.7 GB), 다운로드 속도에 따라 30분 안팎.

## 0. 설치

```bash
git clone <repo-url> RSC-Toolkit && cd RSC-Toolkit
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[trainer,models,notebook]"          # 논문과 같은 핀: pip install -r requirements-lock.txt 뒤 pip install -e . --no-deps
# CPU 만 쓸 때(Part A): 위 줄 앞에 CPU torch 를 먼저 깔면 6 GB 가 아니라 1 GB 안팎이다
#   pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0
python -m rsc_toolkit.cli --help                     # 또는 rsctk --help
```

작업 디렉터리를 하나 정한다. 아래는 `WORK=$PWD/outputs/tutorial` (git 이 무시하는 위치) 기준이며, 학습 산출물은 `cwd` 아래 `outputs/<group>/<run>/` 에 생기므로 명령은 `WORK` 에서 실행한다.

```bash
export WORK=$PWD/outputs/tutorial ROOT=$PWD; mkdir -p "$WORK"; cd "$WORK"
export RSCTK_PRETRAINED_ROOT=$WORK/pretrained                                                # 자동 다운로드 가중치 위치
SANDBOX=$(mktemp -d /tmp/rsc_tutorial_sandbox.XXXXXX)                                        # 재평가의 쓰기 격리 디렉터리 (outputs/ 밖)
```

## Part A — toy 데이터 (CPU)

### A-1. 인덱스
```bash
rsctk datasets build-index toy --project_root "$ROOT" --data_dir "$WORK/toy" --sample_level cycle
```

### A-2. 학습 (CNN6, 1 epoch)
```bash
rsc-toolkit-train --config-name default data=toy_fbank model=cnn model.params.backbone=cnn6 callbacks=best_checkpoint \
  data.params.project_root="$ROOT" data.params.data_dir="$WORK/toy" data.params.features.cache_dir="$WORK/cache_toy" \
  data.params.num_workers=0 trainer.params.max_epochs=1 trainer.params.accelerator=cpu \
  run.name=toy-cnn run.group=tutorial hydra.run.dir="$WORK/hydra/toy-cnn" +run.overwrite=true
cat "$WORK/outputs/tutorial/toy-cnn/best.json"
```
`callbacks=best_checkpoint` 가 검증 Score 최고 시점의 `best.pth`·`best.json` 을 `outputs/<run.group>/<run.name>/` 에 쓰고, `hydra.run.dir` 아래 `.hydra/config.yaml` 이 설정 스냅샷이다.

> `+run.overwrite=true`: 같은 `run.name` 의 best.pth/best.json 이 이미 있으면 툴킷은 덮어쓰지 않고 멈춘다 — 튜토리얼은 재실행을 전제로 하므로 명시 허용한다.

### A-3. 재평가
```bash
rsc-toolkit-eval "$WORK/outputs/tutorial/toy-cnn" --project-root "$ROOT" --hydra-dir "$WORK/hydra/toy-cnn" --sandbox "$SANDBOX"
```
스냅샷으로 같은 팩토리에서 모델·데이터를 다시 세우고 `best.pth` 를 strict 로드해 평가한 뒤 `best.json` 과 대조한다(PASS = |Δ| ≤ 0.05). run 디렉터리는 절대 경로로 준다(상대 경로는 `--project-root` 기준으로 풀린다). `--sandbox` 는 쓰기 격리용이며 `outputs/` 밖이어야 한다.

### A-4. 자기 모델
파일 하나로 끝난다. `BaseRSCModel` 을 상속하고 `training_forward` 만 구현해 등록한다(계약: `rsc_toolkit/models/base.py`).

```bash
mkdir -p "$WORK/plugins/configs/model"
cat > "$WORK/plugins/my_model.py" <<'PY'
import torch
from torch import nn
from rsc_toolkit.models.base import BaseRSCModel
from rsc_toolkit.models.factory import ModelFactory

@ModelFactory.register_model_class("tutorial_cnn")
class TutorialCNN(BaseRSCModel, nn.Module):
    def __init__(self, n_cls: int = 4, hidden: int = 16, **params):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(), nn.MaxPool2d(4),
                                 nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Linear(hidden, n_cls))
    def forward(self, x):
        return self.net(x.unsqueeze(1) if x.dim() == 3 else x)      # [B, F, T] → [B, 1, F, T]
    def training_forward(self, audio_batch, labels, loss_func, **kwargs):
        logits = self.forward(audio_batch)
        return logits, loss_func(logits, labels)
PY
printf 'name: tutorial_cnn\nparams:\n  n_cls: 4\n  hidden: 16\n' > "$WORK/plugins/configs/model/tutorial_cnn.yaml"

RSC_TOOLKIT_PLUGINS=my_model PYTHONPATH="$WORK/plugins" \
rsc-toolkit-train --config-name default --config-dir "$WORK/plugins/configs" data=toy_fbank model=tutorial_cnn callbacks=best_checkpoint \
  data.params.project_root="$ROOT" data.params.data_dir="$WORK/toy" data.params.features.cache_dir="$WORK/cache_toy" \
  data.params.num_workers=0 trainer.params.max_epochs=1 trainer.params.accelerator=cpu \
  run.name=toy-mymodel run.group=tutorial hydra.run.dir="$WORK/hydra/toy-mymodel" +run.overwrite=true
RSC_TOOLKIT_PLUGINS=my_model PYTHONPATH="$WORK/plugins" \
rsc-toolkit-eval "$WORK/outputs/tutorial/toy-mymodel" --project-root "$ROOT" --hydra-dir "$WORK/hydra/toy-mymodel" --sandbox "$SANDBOX"
```
- 입력 `audio_batch` 는 데이터 config 가 정한 모양(fbank 면 `[B, F, T]`)이고, `loss_func` 는 config 의 손실이 주입된다. 반환은 `(logits [B, n_cls], loss)`.
- 등록: 환경변수 `RSC_TOOLKIT_PLUGINS=<모듈>` 또는 패키지 엔트리포인트 `rsc_toolkit.plugins`. 모델 config 는 Hydra `--config-dir` 로 저장소 밖에서 준다.
- 손실·지표·datamodule·프로세서도 같은 등록소 방식이다 — README "자기 모델 붙이기" 절.

## Part B — ICBHI 2017 (GPU)

### B-1. 다운로드 · 인덱스 · 검증
```bash
rsctk datasets download    icbhi --project_root "$ROOT" --data_dir "$WORK/icbhi"       # 공식 zip 1.98 GB
rsctk datasets build-index icbhi --project_root "$ROOT" --data_dir "$WORK/icbhi" --sample_level cycle
rsctk datasets validate    icbhi --project_root "$ROOT" --data_dir "$WORK/icbhi"
```

### B-2. 1 epoch 학습 (표 1 "CNN6 fine-tuning" 레시피, 원래 400 epoch)
GPU 학습에서는 dataloader 워커당 BLAS/OpenCV 스레드를 1 로 캡한다(공유 서버에서 워커 × 스레드 풀 폭주 방지; CPU 학습인 Part A 에는 걸지 않는다).
PANNs Cnn6 사전학습 가중치는 zenodo(record 3960586)에서 `$RSCTK_PRETRAINED_ROOT/upstream/Cnn6_mAP=0.343.pth` 로 자동 다운로드된다.
zenodo 가 막힌 망이면 그 파일을 같은 경로에 직접 두면 된다.
```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1
rsc-toolkit-train --config-name sgscl_nonast_icbhi model.name=cnn6_sft \
  data.params.project_root="$ROOT" data.params.data_dir="$WORK/icbhi" data.params.features.cache_dir="$WORK/cache_icbhi" \
  trainer.params.max_epochs=1 run.name=icbhi-cnn6-1epoch run.group=tutorial hydra.run.dir="$WORK/hydra/icbhi-cnn6-1epoch" +run.overwrite=true
rsc-toolkit-eval "$WORK/outputs/sgscl/icbhi-cnn6-1epoch" --project-root "$ROOT" --hydra-dir "$WORK/hydra/icbhi-cnn6-1epoch" --sandbox "$SANDBOX"
```
(이 레시피의 콜백은 `$WORK/outputs/sgscl/<run.name>` 에 저장한다.)

### B-3. 공개된 논문 weight 재평가
논문 표의 weight 는 run 묶음(`best.pth` · `best.json` · `.hydra/config.yaml`)으로 공개된다. 목록·링크는
[`weights_manifest.csv`](weights_manifest.csv), 내려받기는 `scripts/download_paper_weights.py`(gdown 사용, `notebook` extra 에 포함; sha256 대조).
url 은 Google Drive 폴더 링크(`…/folders/<id>`; run 폴더 자체 또는 run 폴더들을 담은 상위 폴더)이며, 아직 `[Google Drive]` 자리표시자면 안내만 나온다 — 그때는 묶음을 `$WORK/paper_weights/<run>/` 에 직접 둔다.

```bash
python "$ROOT/scripts/download_paper_weights.py" --list
python "$ROOT/scripts/download_paper_weights.py" --run icbhi-cnn6-ce-s1 --out "$WORK/paper_weights"
rsc-toolkit-eval "$WORK/paper_weights/icbhi-cnn6-ce-s1" --project-root "$ROOT" --sandbox "$SANDBOX" \
  --override data.params.data_dir="$WORK/icbhi" --override data.params.features.cache_dir="$WORK/cache_icbhi"
```
묶음이 설정 스냅샷을 품고 있으므로 `outputs/` 없이 재평가된다. 경로 override 가 붙은 평가는 `status: OK` 와 함께 `best.json`(논문 표의 값)과의 |Δ| 를 보여주며, 같은 공식 test 이므로 |Δ| 는 0 이다(override 없이 돌리면 PASS/MISMATCH 판정).
논문 weight 전체를 다시 검증하려면 `scripts/verify_paper_weights.sh`(README "평가" 절).

## 패키지 설치만으로 쓰기
자기 모델만 붙이는 사용자는 clone 없이도 된다: `pip install "rsc-toolkit[trainer,models] @ git+<repo-url>"`.
같은 명령이 동작하되 toy 데이터(`sample_data/`)는 휠에 동봉되지 않으므로 Part A 대신 자기 데이터·ICBHI(Part B)로 시작한다.

## 재현·검증 도구 한눈에
| 무엇 | 명령 |
|---|---|
| 논문 레시피 학습 | `scripts/train.sh <config> [override…]` (README 표) |
| run 하나 재평가 | `rsc-toolkit-eval outputs/<group>/<run>` |
| 논문 weight 전수 재검증 | `scripts/verify_paper_weights.sh` / `scripts/verify_paper_ood.py` |
| 튜토리얼 자동 실행 | `scripts/tutorial/run_tutorial.sh [--full]` (CI 는 Part A) |
