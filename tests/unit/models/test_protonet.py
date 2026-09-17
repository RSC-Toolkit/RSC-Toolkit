"""
Port golden tests for Prototype (ICASSP 2022) — compared with the official L3S/PrototypeSound.

The official code cannot be imported as is because of (1) `np.float` (removed in numpy>=1.24), (2) the matplotlib dependency and (3) forced cuda
in `move_data_to_gpu`. Here only those three are minimally worked around to load the original module and compare values.
Skipped if the clone is missing.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import numpy as np
import pytest
import torch

REF = pathlib.Path("outputs/reference/prototypesound")
pytestmark = pytest.mark.skipif(not REF.exists(), reason="PrototypeSound clone missing")


def _load_official():
    """Loads the official models_pytorch.py by file path without polluting sys.path.

    What the official module needs: `config` (utils/config.py constants), `matplotlib.pyplot`, `np.float`.
    `move_data_to_gpu` is defined inside models_pytorch.py, so the utils package is not needed.
    sys.path is never touched, since that would hijack imports (lightning etc.) in subsequent tests.
    """
    np.float = float  # the official PrototypeLayer uses np.zeros(dtype=np.float) (removed in numpy>=1.24)

    if "official_models_pytorch" in sys.modules:
        return sys.modules["official_models_pytorch"]

    # fill matplotlib.pyplot / config imported by the official module with stubs **only while loading**, then restore immediately.
    # (leftover stubs make torchmetrics fail looking for matplotlib.axes, and touching sys.path hijacks the lightning import.)
    added = []
    for name in ("matplotlib", "matplotlib.pyplot"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__getattr__ = lambda n: (lambda *a, **k: None)  # noqa: ARG005
            sys.modules[name] = m
            added.append(name)
    if "matplotlib" in added:
        sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]
    if "config" not in sys.modules:  # load utils/config.py as is (do not re-type the constants)
        cspec = importlib.util.spec_from_file_location("config", REF / "baseline_cnn" / "utils" / "config.py")
        cmod = importlib.util.module_from_spec(cspec)
        cspec.loader.exec_module(cmod)
        sys.modules["config"] = cmod
        added.append("config")

    try:
        spec = importlib.util.spec_from_file_location(
            "official_models_pytorch", REF / "baseline_cnn" / "pytorch" / "models_pytorch.py"
        )
        M = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(M)
        sys.modules["official_models_pytorch"] = M
    finally:
        for name in added:
            sys.modules.pop(name, None)
    M.move_data_to_gpu = lambda x, cuda: (torch.Tensor(x) if isinstance(x, np.ndarray) else x).float()
    return M


@pytest.mark.parametrize(
    "form", ["vector1d", "vector2d", "vector2d_att", "vector2d_avgp", "vector2d_maxp", "vector2d_maxp_att"]
)
def test_prototype_layer_matches_official(form):
    """All 6 proto_forms are bit-identical to the official loop implementation (vectorization must not change values)."""
    from rsc_toolkit.models.protonet import PrototypeLayer

    M = _load_official()
    torch.manual_seed(0)
    x = torch.randn(3, 512, 7, 8)
    torch.manual_seed(7)
    ref = M.PrototypeLayer(n_proto=4, distance="cosine", proto_form=form).eval()
    torch.manual_seed(7)
    our = PrototypeLayer(n_proto=4, proto_form=form).eval()
    our.load_state_dict(ref.state_dict(), strict=False)
    with torch.no_grad():
        rs, _ = ref(x)
        os_, _ = our(x)
    assert torch.allclose(rs, os_, atol=1e-6), f"{form}: max|diff|={(rs - os_).abs().max().item():.3e}"


def test_diverse_loss_matches_official():
    from rsc_toolkit.models.protonet import diverse_loss

    M = _load_official()
    torch.manual_seed(7)
    ref_layer = M.PrototypeLayer(n_proto=4, distance="cosine", proto_form="vector2d")
    t = M.ProtoNet.__new__(M.ProtoNet)
    torch.nn.Module.__init__(t)
    t.classes_num = 4
    with torch.no_grad():
        assert abs(t.diverse_loss(ref_layer.prototype).item() - diverse_loss(ref_layer.prototype, 4).item()) < 1e-6


@pytest.mark.parametrize("method,official", [("cnn8", "DecisionLevelMaxPooling"), ("proto", "ProtoNet")])
def test_full_model_matches_official(method, official):
    """Bit-identical to the official code from feature extraction (torchlibrosa) through the encoder to the output."""
    from rsc_toolkit.models.protonet import ProtoNetModel

    M = _load_official()
    torch.manual_seed(0)
    wav = torch.randn(2, 16000)
    torch.manual_seed(11)
    ref = (M.DecisionLevelMaxPooling(classes_num=4) if method == "cnn8"
           else M.ProtoNet(classes_num=4, n_proto=4, proto_form="vector2d")).eval()
    torch.manual_seed(11)
    our = ProtoNetModel(n_cls=4, method=method, proto_form="vector2d").eval()
    mapping = {
        k.replace("spectrogram_extractor", "logmel.spectrogram_extractor")
         .replace("logmel_extractor", "logmel.logmel_extractor"): v
        for k, v in ref.state_dict().items()
    }
    missing, unexpected = our.load_state_dict(mapping, strict=False)
    assert not unexpected and [m for m in missing] == ["_class_weight"]
    with torch.no_grad():
        assert torch.equal(ref(wav), our(wav))


def test_fix_length_matches_numpy_wrap_and_center_crop():
    """Short input: np.pad(wrap); long input: center crop (eval) — official data_generator.read_batch_audio."""
    from rsc_toolkit.datasets.protonet_data import fix_length

    short = torch.arange(10, dtype=torch.float).view(1, 10)
    out = fix_length(short, 25, train=False)
    expected = torch.as_tensor(np.pad(short.numpy()[0], (0, 15), mode="wrap"), dtype=torch.float)
    assert torch.equal(out[0], expected)

    long = torch.arange(100, dtype=torch.float).view(1, 100)
    out = fix_length(long, 40, train=False)
    start = int((100 - 40) / 2)
    assert torch.equal(out[0], long[0, start:start + 40])

    torch.manual_seed(0)
    tr = fix_length(long, 40, train=True)
    assert tr.shape[-1] == 40


def test_bandpass_matches_official_filter():
    """scipy butter/lfilter coefficients and application are identical to the official preprocessing."""
    from scipy.signal import butter, lfilter

    from rsc_toolkit.datasets.protonet_data import ButterBandpass

    fs, low, high, order = 4000, 100.0, 1800.0, 5
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    x = np.random.RandomState(0).randn(1, 5000)
    ref = lfilter(b, a, x, axis=-1)
    got = ButterBandpass(low, high, fs, order)(torch.as_tensor(x, dtype=torch.float32))
    assert np.allclose(ref, got.numpy(), atol=1e-4)
