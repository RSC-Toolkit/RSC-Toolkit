# -*- coding: utf-8 -*-
"""
notebooks/tutorial.ipynb 생성기. 노트북은 이 파일에서 만들어지므로 셀을 고칠 때는 여기를 고치고 다시 생성한다:

  .venv/bin/python scripts/tutorial/make_notebook.py            # → notebooks/tutorial.ipynb (출력 없는 깨끗한 노트북)
  scripts/tutorial/run_tutorial.sh [--full]                     # 실행 (기본 Part A 만; --full 은 ICBHI 까지)

Part A(toy, CPU, 수 분)는 CI 가 매번 실행하고, Part B(ICBHI 다운로드·GPU 1 epoch·공개 weight 재평가)는 개발 머신에서 돌린다.
"""
from __future__ import annotations

import pathlib
import sys

import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "notebooks" / "tutorial.ipynb"   # 인자로 다른 출력 경로 (테스트용)

md = lambda s: nbf.v4.new_markdown_cell(s.strip("\n"))
code = lambda s: nbf.v4.new_code_cell(s.strip("\n"))

cells = []

cells.append(md(r"""
# RSC-Toolkit 튜토리얼 — 설치부터 자기 모델 학습까지

이 노트북은 툴킷을 **처음 쓰는 사람**이 한 번에 따라 하는 최소 경로다.

| Part | 무엇을 하나 | 필요한 것 | 시간 |
|---|---|---|---|
| A | 번들 toy 데이터로 학습 → 재평가 → **자기 모델 등록·학습** | CPU 만 | 수 분 |
| B | ICBHI 2017 공식 데이터 다운로드 → 1 epoch 학습 → **공개된 논문 weight 재평가** | GPU 1장, 디스크 ≈ 8 GB | 다운로드 속도에 따라 30분 안팎 |

**준비**: Python ≥ 3.9 (권장 3.11), 저장소 clone 뒤 `pip install -e ".[trainer,models,notebook]"`
(논문과 같은 핀으로 맞추려면 `pip install -r requirements-lock.txt`). 패키지만 설치해 자기 모델을 붙이는 경로는
`docs/TUTORIAL.md` 의 "패키지 설치" 절을 보라.

환경변수로 동작을 바꿀 수 있다: `RSC_TUTORIAL_QUICK=1` → Part B 를 건너뛴다(CI 가 이렇게 돈다). `RSC_TUTORIAL_WORK` → 작업 디렉터리(기본 `<clone>/outputs/tutorial`). `RSC_TUTORIAL_ROOT` → 저장소 루트를 직접 지정.
"""))

cells.append(code(r"""
import json, os, pathlib, subprocess, sys

# 저장소 루트(pyproject.toml 과 rsc_toolkit/ 이 있는 곳) — 노트북은 notebooks/ 에서 실행되므로 부모를 거슬러 찾는다
def find_root(start: pathlib.Path) -> pathlib.Path:
    env = os.environ.get("RSC_TUTORIAL_ROOT")
    if env:
        return pathlib.Path(env).resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() and (p / "rsc_toolkit").is_dir():
            return p
    raise SystemExit("저장소 루트를 찾지 못했다 — RSC-Toolkit clone 안에서 실행하거나 RSC_TUTORIAL_ROOT 를 설정하라")

ROOT = find_root(pathlib.Path.cwd().resolve())
WORK = pathlib.Path(os.environ.get("RSC_TUTORIAL_WORK") or ROOT / "outputs" / "tutorial").resolve()
WORK.mkdir(parents=True, exist_ok=True)
QUICK = os.environ.get("RSC_TUTORIAL_QUICK", "0") == "1"
import tempfile
SANDBOX = pathlib.Path(tempfile.mkdtemp(prefix="rsc_tutorial_sandbox_"))   # 재평가의 쓰기 격리 디렉터리 — 실행마다 새로, outputs/ 밖(도구가 강제)

BIN = pathlib.Path(sys.executable).parent            # 이 커널의 venv 에 설치된 콘솔 스크립트를 쓴다
RSCTK, TRAIN, EVAL = BIN / "rsctk", BIN / "rsc-toolkit-train", BIN / "rsc-toolkit-eval"
for exe in (RSCTK, TRAIN, EVAL):
    assert exe.exists(), f"{exe} 가 없다 — pip install -e '.[trainer,models,notebook]' 를 이 커널의 환경에 했는지 확인"

ENV = {**os.environ,
       "RSCTK_PRETRAINED_ROOT": str(WORK / "pretrained"),          # 자동 다운로드되는 사전학습 가중치가 놓일 곳
       "HYDRA_FULL_ERROR": "1"}
# GPU 학습(Part B)에서만 dataloader 워커당 BLAS/OpenCV 스레드를 1 로 캡한다 — 공유 서버에서 워커 × 스레드 풀 폭주 방지.
# CPU 학습(Part A)은 코어를 다 써야 하므로 캡하지 않는다.
ENV_GPU = {**ENV, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENCV_FOR_THREADS_NUM": "1"}

def run(*cmd, cwd=WORK, check=True, env=None):
    # 명령을 그대로 보여주고 실행한다 (cwd 기본 = WORK: 학습 산출물 outputs/... 이 WORK 아래에 생긴다)
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd), f"\n  (cwd={cwd})")
    proc = subprocess.run(cmd, cwd=str(cwd), env=env or ENV, text=True, capture_output=True)
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-25:])
    print(tail)
    if check and proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}")
    return proc

import rsc_toolkit, torch
print("rsc_toolkit:", pathlib.Path(rsc_toolkit.__file__).parent, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("ROOT:", ROOT, "\nWORK:", WORK, "\nQUICK:", QUICK)
"""))

cells.append(md(r"""
## Part A — toy 데이터 (CPU)

`sample_data/toy_dataset/` 은 ICBHI 환자 107 의 녹음만 담은 번들 데이터다. 실제 ICBHI 와 같은 파일 형식이라
인덱스 생성 → 학습 → 평가 경로가 실데이터와 동일하게 돈다.

### A-1. 인덱스 만들기
툴킷은 원본 오디오를 건드리지 않고 **파생 인덱스**(`index/index_cycle.csv`)만 만든다. train/test 구분은 인덱스가 아니라 데이터 config 의 `split_file`(여기서는 `sample_data/toy_dataset/official_split.txt`)이 정한다.
"""))

cells.append(code(r"""
TOY = WORK / "toy"
run(RSCTK, "datasets", "build-index", "toy", "--project_root", ROOT, "--data_dir", TOY, "--sample_level", "cycle")
import pandas as pd
idx = pd.read_csv(TOY / "index" / "index_cycle.csv")
split = pd.read_csv(ROOT / "sample_data" / "toy_dataset" / "official_split.txt", sep=r"\s+", header=None, names=["recording", "split"])
print(idx.shape, "cycles;", idx["label"].value_counts().to_dict(), "| split file:", split["split"].value_counts().to_dict())
"""))

cells.append(md(r"""
### A-2. 학습 (CNN6, CPU, 1 epoch)

모든 학습은 `rsc-toolkit-train --config-name <config> key=value …` 한 줄이다. 여기서 쓰는 것:
- `data=toy_fbank` — toy 데이터 + log-mel fbank 피처(캐시는 `data.params.features.cache_dir`)
- `model=cnn model.params.backbone=cnn6` — 툴킷 내장 PANNs 계열 CNN(6층)
- `callbacks=best_checkpoint` — 검증 Score 가 최고일 때 `outputs/<run.group>/<run.name>/best.pth·best.json` 저장
- `hydra.run.dir` — 설정 스냅샷(`.hydra/config.yaml`) 위치. 재평가가 이 스냅샷으로 모델·데이터를 다시 세운다
"""))

cells.append(code(r"""
def train_toy(run_name: str, *extra, config_dir=None):
    # Hydra 는 옵션(--config-dir 등)을 override 들보다 앞에 두어야 한다 (argparse 가 뒤섞인 위치 인자를 받지 않는다)
    opts = ["--config-dir", str(config_dir)] if config_dir else []
    model = ["model=cnn", "model.params.backbone=cnn6"] if not any(str(e).startswith("model=") for e in extra) else []
    run(TRAIN, "--config-name", "default", *opts, "data=toy_fbank", *model, "callbacks=best_checkpoint",
        f"data.params.project_root={ROOT}", f"data.params.data_dir={TOY}", f"data.params.features.cache_dir={WORK / 'cache_toy'}",
        "data.params.num_workers=0", "trainer.params.max_epochs=1", "trainer.params.accelerator=cpu",
        f"run.name={run_name}", "run.group=tutorial", f"hydra.run.dir={WORK / 'hydra' / run_name}", *extra)
    best = json.loads((WORK / "outputs" / "tutorial" / run_name / "best.json").read_text())
    print("best.json:", best)
    return best

best_toy = train_toy("toy-cnn")
"""))

cells.append(md(r"""
### A-3. 재평가

`rsc-toolkit-eval <run 디렉터리>` 는 스냅샷으로 같은 팩토리에서 모델·데이터를 다시 세우고 `best.pth` 를 strict 로드해
검증 스플릿을 평가한 뒤 `best.json` 과 대조한다(PASS = |Δ| ≤ 0.05). 학습 산출물은 읽기만 하고 쓰기는 `--sandbox`(outputs/ 밖의 임시 디렉터리) 로 격리된다.
"""))

cells.append(code(r"""
def evaluate(run_dir, hydra_dir=None, *overrides):
    out = WORK / f"{pathlib.Path(run_dir).name}.eval.json"
    cmd = [EVAL, run_dir, "--project-root", ROOT, "--sandbox", SANDBOX, "--out", out, "--quiet"]
    if hydra_dir is not None:
        cmd += ["--hydra-dir", hydra_dir]
    for ov in overrides:
        cmd += ["--override", ov]
    run(*cmd)
    res = json.loads(out.read_text())
    print(res["status"], {k: round(v, 3) for k, v in res["metrics"].items() if k in ("S_p", "S_e", "icbhi_score")}, "| |Δ| =", res.get("max_abs_diff"))
    return res

res_toy = evaluate(WORK / "outputs" / "tutorial" / "toy-cnn", WORK / "hydra" / "toy-cnn")
assert res_toy["status"] == "PASS", res_toy
"""))

cells.append(md(r"""
### A-4. 자기 모델 붙이기

툴킷을 포크하지 않는다. 파일 하나에 `BaseRSCModel` 을 상속한 클래스를 쓰고 `training_forward` 를 구현해 등록하면
학습·검증·재평가 루프가 그대로 돈다(계약 전문: `rsc_toolkit/models/base.py`).

- 입력 `audio_batch` 는 데이터 config 가 정한 모양으로 온다(fbank 면 `[B, F, T]`, 채널 축 없음).
- `loss_func` 는 config 의 손실(가중 CE 등)이 주입되므로 모델 안에서 새로 만들지 않는다.
- 반환은 `(logits [B, n_cls], loss)`.

등록은 환경변수 `RSC_TOOLKIT_PLUGINS=<모듈>` (또는 패키지 엔트리포인트 `rsc_toolkit.plugins`)로, 모델 config 는
Hydra 의 `--config-dir` 로 **저장소 밖 디렉터리**에서 준다.
"""))

cells.append(code(r"""
PLUG = WORK / "plugins"; (PLUG / "configs" / "model").mkdir(parents=True, exist_ok=True)
(PLUG / "my_model.py").write_text('''
import torch
from torch import nn
from rsc_toolkit.models.base import BaseRSCModel
from rsc_toolkit.models.factory import ModelFactory


@ModelFactory.register_model_class("tutorial_cnn")
class TutorialCNN(BaseRSCModel, nn.Module):
    # 작은 2-layer CNN. n_cls 는 model.params.n_cls 로 주입된다.

    def __init__(self, n_cls: int = 4, hidden: int = 16, **params):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(hidden, n_cls),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:                 # [B, F, T] → [B, 1, F, T]
            x = x.unsqueeze(1)
        return self.net(x)

    def training_forward(self, audio_batch, labels, loss_func, **kwargs):
        logits = self.forward(audio_batch)
        return logits, loss_func(logits, labels)
''')
(PLUG / "configs" / "model" / "tutorial_cnn.yaml").write_text('''
name: tutorial_cnn
params:
  n_cls: 4
  hidden: 16
''')
ENV["RSC_TOOLKIT_PLUGINS"] = "my_model"                       # 툴킷이 import 할 모듈 이름
ENV["PYTHONPATH"] = str(PLUG) + os.pathsep + ENV.get("PYTHONPATH", "")

best_my = train_toy("toy-mymodel", "model=tutorial_cnn", config_dir=PLUG / "configs")
res_my = evaluate(WORK / "outputs" / "tutorial" / "toy-mymodel", WORK / "hydra" / "toy-mymodel")
assert res_my["status"] == "PASS", res_my
"""))

cells.append(md(r"""
Part A 끝. 같은 명령의 `data=`·`model=`·`--config-name` 만 바꾸면 논문 재현 레시피(README 의 표)가 그대로 돈다.

## Part B — ICBHI 2017 실데이터 (GPU)

`RSC_TUTORIAL_QUICK=1` 이면 아래 셀들은 건너뛴다.

### B-1. 다운로드 · 인덱스 · 검증
공식 배포 zip(1.98 GB, 920 녹음)을 받아 `WORK/icbhi/` 에 풀고, 공식 60/40 스플릿으로 사이클 인덱스를 만든다.
"""))

cells.append(code(r"""
ICBHI = WORK / "icbhi"
if QUICK:
    print("skipped (RSC_TUTORIAL_QUICK=1)")
else:
    run(RSCTK, "datasets", "download", "icbhi", "--project_root", ROOT, "--data_dir", ICBHI)        # 이미 있으면 건너뛴다
    run(RSCTK, "datasets", "build-index", "icbhi", "--project_root", ROOT, "--data_dir", ICBHI, "--sample_level", "cycle")
    run(RSCTK, "datasets", "validate", "icbhi", "--project_root", ROOT, "--data_dir", ICBHI)
    idx = pd.read_csv(ICBHI / "index" / "index_cycle.csv")
    print(idx.shape, "cycles;", idx["label"].value_counts().sort_index().to_dict())
"""))

cells.append(md(r"""
### B-2. 1 epoch 학습 (CNN6 fine-tuning 레시피)

논문 재현 config `sgscl_nonast_icbhi` + `model.name=cnn6_sft` 가 표 1 의 "CNN6 fine-tuning" 행이다(원래 400 epoch).
여기서는 1 epoch 만 돌려 경로를 확인한다. PANNs Cnn6 사전학습 가중치는 zenodo(record 3960586)에서 `$RSCTK_PRETRAINED_ROOT/upstream/Cnn6_mAP=0.343.pth` 로 자동 다운로드된다 — zenodo 가 막힌 망이면 그 파일을 같은 경로에 직접 둔다. GPU 학습이라 워커당 스레드 캡(`ENV_GPU`)을 건다.
"""))

cells.append(code(r"""
if QUICK:
    print("skipped (RSC_TUTORIAL_QUICK=1)")
else:
    name = "icbhi-cnn6-1epoch"
    run(TRAIN, "--config-name", "sgscl_nonast_icbhi", "model.name=cnn6_sft",
        f"data.params.project_root={ROOT}", f"data.params.data_dir={ICBHI}", f"data.params.features.cache_dir={WORK / 'cache_icbhi'}",
        "trainer.params.max_epochs=1", f"run.name={name}", "run.group=tutorial", f"hydra.run.dir={WORK / 'hydra' / name}", env=ENV_GPU)
    # sgscl_nonast_paper 콜백은 outputs/sgscl/<run.name> 에 저장한다
    print(json.loads((WORK / "outputs" / "sgscl" / name / "best.json").read_text()))
    res_1ep = evaluate(WORK / "outputs" / "sgscl" / name, WORK / "hydra" / name)
    assert res_1ep["status"] == "PASS", res_1ep
"""))

cells.append(md(r"""
### B-3. 공개된 논문 weight 재평가

논문 표의 weight 는 run 묶음(`best.pth` · `best.json` · `.hydra/config.yaml`)으로 공개된다. 목록과 링크는
`docs/paper/weights_manifest.csv` 에 있고 `scripts/download_paper_weights.py`(gdown, `notebook` extra 에 포함)가 받아서 sha256 을 대조한다.
링크(run 폴더 링크)가 아직 `[Google Drive]` 자리표시자면 안내만 나온다 — 그때는 묶음을 직접 `WORK/paper_weights/<run>/` 에 두면 된다.

받은 묶음은 설정 스냅샷을 품고 있으므로 `rsc-toolkit-eval <묶음> --override data.params.data_dir=<내 ICBHI 경로>` 로
바로 재평가된다. 경로 override 가 붙은 평가는 `status: OK` 와 함께 `best.json`(논문 표의 값)과의 |Δ| 를 돌려주며, 같은 공식 test 이므로 |Δ| 는 0 이다.
"""))

cells.append(code(r"""
if QUICK:
    print("skipped (RSC_TUTORIAL_QUICK=1)")
else:
    RUN = "icbhi-cnn6-ce-s1"                                   # 표 1 "CNN6 fine-tuning" seed 1 (17 MB)
    bundle = WORK / "paper_weights" / RUN
    if not (bundle / "best.pth").exists():
        proc = run(sys.executable, ROOT / "scripts" / "download_paper_weights.py", "--run", RUN, "--out", WORK / "paper_weights", check=False)
        if proc.returncode == 2:
            print(f"\n공개 링크가 아직 없다. 묶음을 {bundle} 에 두고 이 셀을 다시 실행하라.")
    if (bundle / "best.pth").exists():
        res_paper = evaluate(bundle, None, f"data.params.data_dir={ICBHI}", f"data.params.features.cache_dir={WORK / 'cache_icbhi'}")
        print("논문 표 값(best.json):", json.loads((bundle / "best.json").read_text()))
        # 경로 override 가 있으면 도구는 판정 대신 status=OK 와 대조값(|Δ|)을 돌려준다 — 같은 공식 test 이므로 |Δ| 는 0 이어야 한다
        assert res_paper["status"] in ("PASS", "OK") and (res_paper.get("max_abs_diff") or 1.0) <= 0.05, res_paper
"""))

cells.append(md(r"""
## 다음
- 논문 29편의 재현 레시피(모델 × 데이터셋별 `--config-name` 과 override)는 README 의 "논문 재현 레시피" 표에 있다.
- 자기 모델을 패키지로 배포하려면 `pyproject.toml` 에 `[project.entry-points."rsc_toolkit.plugins"]` 를 선언한다(README "자기 모델 붙이기").
- 손실·지표·datamodule·프로세서도 같은 등록소 방식이다(`rsc_toolkit/registries.py`).
"""))

for i, c in enumerate(cells):                       # 결정적 cell id — 재생성해도 커밋본과 바이트 동일(랜덤 id 로 git 이 더러워지지 않게)
    c["id"] = f"cell-{i:02d}"
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print("wrote", OUT, f"({len(cells)} cells)")
