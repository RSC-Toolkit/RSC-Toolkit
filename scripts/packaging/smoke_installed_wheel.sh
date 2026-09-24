#!/bin/bash
# Release wheel smoke test — checks that the toolkit works "outside the repository, from the installed package only".
#
#   1. Build the wheel (isolated pip build — setuptools is fetched per pyproject build-requires)
#   2. Install only the wheel into an empty venv (--no-deps; heavy dependencies are borrowed from the current interpreter's site-packages via PYTHONPATH.
#      Set FULL_DEPS=1 to install the dependencies too, as in CI)
#   3. From a temporary cwd outside the repository, check: import · built-in registry entries · bundled data files · hydra config composition ·
#      console scripts (rsctk, rsc-toolkit-train). Any failure exits 1.
#
# Usage: scripts/packaging/smoke_installed_wheel.sh          (set the interpreter with PY=…, default .venv/bin/python)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
WORK=${WORK:-$(mktemp -d "${TMPDIR:-/tmp}/rsc_wheel_smoke.XXXXXX")}
echo "smoke: root=$ROOT py=$PY work=$WORK"

# 1. Build the wheel
"$PY" -m pip wheel "$ROOT" --no-deps -w "$WORK/wheel" -q
WHL=$(ls "$WORK"/wheel/rsc_toolkit-*.whl)
echo "wheel: $WHL"

# 2. Install into an empty venv
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

# 3. Check outside the repository
cd "$WORK"
"$VPY" - <<'EOF'
import importlib, pathlib, sys
import rsc_toolkit
pkg = pathlib.Path(rsc_toolkit.__file__).resolve().parent
assert "site-packages" in str(pkg), f"imported the repository tree instead of the installed package: {pkg}"
print("rsc_toolkit:", pkg)

# Bundled data (did the pyproject package-data globs actually land in the wheel?)
for rel in ["configs/default.yaml", "configs/data/icbhi_patchmix.yaml", "datasets/icbhi/data/official_split.txt",
            "datasets/icbhi/data/metadata.txt", "datasets/sprsound/data/metadata.csv",
            "datasets/hflungv1/data/metadata.csv", "processors/data/magma_lut_bgr.npy"]:
    assert (pkg / rel).exists(), f"bundled data missing: {rel}"
print("bundled data: ok")

# Built-in registry entries (declared lazily, so names must be visible without heavy imports)
import rsc_toolkit.registries as r
assert len(r.DATAMODULES) >= 11 and len(r.LOSSES) >= 4 and len(r.METRICS) >= 3 and len(r.ADAPTERS) == 2
assert {"icbhi", "sprsound", "toy"} <= set(r.DATASETS)
assert len(r.MODELS) >= 30, f"model scan failed: {len(r.MODELS)}"
print("registries: models", len(r.MODELS), "datamodules", len(r.DATAMODULES), "datasets", sorted(r.DATASETS))

# hydra config composition (from the installed configs package). ${hydra:runtime.cwd} cannot be resolved outside @hydra.main, so that key is not checked.
from hydra import compose, initialize_config_module
with initialize_config_module(config_module="rsc_toolkit.configs", version_base=None):
    cfg = compose(config_name="patchmix_icbhi", overrides=["seed=1"])
assert cfg.model.name == "ast_patchmix" and cfg.data.params.dataset_name == "icbhi", (cfg.model.name, cfg.data.name)
print("compose: ok")

# Registered-name resolution (actually imports the model class)
from rsc_toolkit.trainer.factories.build_loss import build_loss
build_loss({"name": "cross_entropy"})
print("build_loss: ok")
EOF

"$WORK/venv/bin/rsctk" datasets --help | grep -q 'download' && echo "rsctk: ok"
"$WORK/venv/bin/rsc-toolkit-train" --help | grep -q 'Hydra' && echo "rsc-toolkit-train: ok"
echo "SMOKE-OK $WHL"
