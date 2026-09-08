#!/bin/bash
# 배포 휠 스모크 — "저장소 밖에서, 설치된 패키지만으로" 툴킷이 뜨는지 확인한다 (리팩토링 P2-1).
#
#   1. 휠 빌드 (pip 격리 빌드 — pyproject 의 build-requires 대로 setuptools 를 받는다)
#   2. 빈 venv 에 휠만 설치 (--no-deps; 무거운 의존성은 현재 인터프리터의 site-packages 를 PYTHONPATH 로 빌려 쓴다.
#      CI 처럼 의존성까지 새로 깔고 싶으면 FULL_DEPS=1)
#   3. 저장소 밖 임시 cwd 에서: import · registry 내장 목록 · 번들 데이터 파일 · hydra config 합성 ·
#      콘솔 스크립트(rsctk, rsc-toolkit-train) 를 검사한다. 하나라도 실패하면 exit 1.
#
# 사용: scripts/packaging/smoke_installed_wheel.sh            (PY=… 로 인터프리터 지정 가능, 기본 .venv/bin/python)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
WORK=${WORK:-$(mktemp -d "${TMPDIR:-/tmp}/rsc_wheel_smoke.XXXXXX")}
echo "smoke: root=$ROOT py=$PY work=$WORK"

# 1. 휠 빌드
"$PY" -m pip wheel "$ROOT" --no-deps -w "$WORK/wheel" -q
WHL=$(ls "$WORK"/wheel/rsc_toolkit-*.whl)
echo "wheel: $WHL"

# 2. 빈 venv 에 설치
"$PY" -m venv "$WORK/venv"
VPY="$WORK/venv/bin/python"
if [ "${FULL_DEPS:-0}" = "1" ]; then
  "$VPY" -m pip install -q "$WHL[trainer,models]"
  PP=""
else
  "$VPY" -m pip install -q --no-deps "$WHL"
  PP=$("$PY" -c "import site; print(':'.join(site.getsitepackages()))")
fi
SP=$("$VPY" -c "import site; print(site.getsitepackages()[0])")
export PYTHONPATH="$SP${PP:+:$PP}"

# 3. 저장소 밖에서 검사
cd "$WORK"
"$VPY" - <<'EOF'
import importlib, pathlib, sys
import rsc_toolkit
pkg = pathlib.Path(rsc_toolkit.__file__).resolve().parent
assert "site-packages" in str(pkg), f"설치본이 아니라 저장소 트리를 import 했다: {pkg}"
print("rsc_toolkit:", pkg)

# 번들 데이터 (pyproject package-data 글롭이 실제로 휠에 실렸는가)
for rel in ["configs/default.yaml", "configs/data/icbhi_patchmix.yaml", "datasets/icbhi/data/official_split.txt",
            "datasets/icbhi/data/metadata.txt", "datasets/sprsound/data/metadata.csv",
            "datasets/hflungv1/data/metadata.csv", "processors/data/magma_lut_bgr.npy"]:
    assert (pkg / rel).exists(), f"번들 데이터 누락: {rel}"
print("bundled data: ok")

# registry 내장 목록 (lazy 선언이므로 무거운 import 없이 이름이 보여야 한다)
import rsc_toolkit.registries as r
assert len(r.DATAMODULES) >= 11 and len(r.LOSSES) >= 4 and len(r.METRICS) >= 3 and len(r.ADAPTERS) == 2
assert {"icbhi", "sprsound", "toy"} <= set(r.DATASETS)
assert len(r.MODELS) >= 30, f"모델 스캔 실패: {len(r.MODELS)}"
print("registries: models", len(r.MODELS), "datamodules", len(r.DATAMODULES), "datasets", sorted(r.DATASETS))

# hydra config 합성 (설치된 configs 패키지에서). ${hydra:runtime.cwd} 는 @hydra.main 밖에서 못 풀므로 그 키는 안 본다.
from hydra import compose, initialize_config_module
with initialize_config_module(config_module="rsc_toolkit.configs", version_base=None):
    cfg = compose(config_name="patchmix_icbhi", overrides=["seed=1"])
assert cfg.model.name == "ast_patchmix" and cfg.data.params.dataset_name == "icbhi", (cfg.model.name, cfg.data.name)
print("compose: ok")

# 등록 이름 해석 (모델 클래스 실제 import)
from rsc_toolkit.trainer.factories.build_loss import build_loss
build_loss({"name": "cross_entropy"})
print("build_loss: ok")
EOF

"$WORK/venv/bin/rsctk" datasets --help | grep -q 'download' && echo "rsctk: ok"
"$WORK/venv/bin/rsc-toolkit-train" --help | grep -q 'Hydra' && echo "rsc-toolkit-train: ok"
echo "SMOKE-OK $WHL"
