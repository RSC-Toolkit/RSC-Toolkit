"""**Cache-key parity** after moving features.kind to the PROCESSORS registry.

The if-elif in IndexDataModule._build_feature_processor (former constructor arguments) moved to each processor's from_options.
The cache key is get_processor_config(), so if get_processor_config() of (old) direct construction and (new) from_options
construction match, existing caches (tens of GB) still hit. This pins that per built-in kind.
The truth table of the SpecAugment gate (is_spectrogram) must also match the former {"mel","fbank"} whitelist.
"""
from __future__ import annotations

from dataclasses import asdict

import pytest

from rsc_toolkit.registries import AUGMENTATIONS, PROCESSORS
from rsc_toolkit.trainer.data.index_datamodule import FeatureConfig, IndexDataModule, SpecAugmentOptions


def _opts(**overrides):
    cfg = FeatureConfig(**overrides)
    opts = asdict(cfg)
    params = opts.pop("params", None) or {}
    opts.update(params)
    return opts


def test_builtin_processor_kinds_snapshot():
    assert set(PROCESSORS) == {"clap", "mel", "fbank", "sclmeta_mel", "waveform"}
    assert set(AUGMENTATIONS) == {"specaugment"}
    assert PROCESSORS.get("mel") is PROCESSORS.get("fbank")  # alias


def test_is_spectrogram_truth_table_matches_former_whitelist():
    expected = {"clap": False, "mel": True, "fbank": True, "sclmeta_mel": False, "waveform": False}
    assert {k: bool(getattr(PROCESSORS.get(k), "is_spectrogram", False)) for k in expected} == expected


def test_mel_from_options_cache_key_parity():
    from rsc_toolkit.processors.mel import MelBankAudioProcessor

    opts = _opts(kind="fbank", n_mels=128, target_frames=798, frame_shift=10, window_type="hanning",
                 dither=0.0, normalize=True, per_sample_stats=False)
    old = MelBankAudioProcessor(sampling_rate=16000, n_mels=128, target_frames=798, frame_shift=10,
                                window_type="hanning", dither=0.0, normalize=True, per_sample_stats=False,
                                feature_cache=None)
    new = PROCESSORS.get("fbank").from_options(opts, sampling_rate=16000, feature_cache=None)
    assert type(new) is MelBankAudioProcessor
    assert new.get_processor_config() == old.get_processor_config() == {
        "sampling_rate": 16000, "n_mels": 128, "target_frames": 798, "frame_shift": 10,
        "window_type": "hanning", "dither": 0.0, "normalize": True, "per_sample_stats": False,
    }


def test_sclmeta_mel_from_options_cache_key_parity():
    from rsc_toolkit.processors.sclmeta_mel import SCLMetaMelProcessor

    # keep the former rule that fills win_length=None with n_fft (fc.win_length or fc.n_fft)
    opts = _opts(kind="sclmeta_mel", n_mels=64, n_fft=1024, win_length=None, hop_length=512, f_min=50.0, f_max=2000.0)
    old = SCLMetaMelProcessor(sampling_rate=16000, n_mels=64, n_fft=1024, win_length=1024, hop_length=512,
                              f_min=50.0, f_max=2000.0, feature_cache=None)
    new = PROCESSORS.get("sclmeta_mel").from_options(opts, sampling_rate=16000, feature_cache=None)
    assert type(new) is SCLMetaMelProcessor
    assert new.get_processor_config() == old.get_processor_config()
    assert new.get_processor_config()["win_length"] == 1024


def test_waveform_from_options_ignores_cache():
    from rsc_toolkit.processors.waveform import WaveformProcessor

    sentinel = object()
    new = PROCESSORS.get("waveform").from_options(_opts(kind="waveform"), sampling_rate=16000, feature_cache=sentinel)
    assert type(new) is WaveformProcessor and new.feature_cache is None
    assert new.get_processor_config() == WaveformProcessor(sampling_rate=16000).get_processor_config()


def test_clap_from_options_cache_key_parity(monkeypatch):
    import rsc_toolkit.processors.clap as clap_mod

    # avoid downloading HF weights — only constructor arguments and cache key are checked
    monkeypatch.setattr(clap_mod.ClapProcessor, "from_pretrained", staticmethod(lambda *a, **k: None))
    opts = _opts(kind="clap", clap_model_name="laion/clap-htsat-unfused")
    old = clap_mod.CLAPProcessor(sampling_rate=48000, model_name="laion/clap-htsat-unfused", feature_cache=None)
    new = PROCESSORS.get("clap").from_options(opts, sampling_rate=48000, feature_cache=None)
    assert type(new) is clap_mod.CLAPProcessor
    assert new.get_processor_config() == old.get_processor_config() == {
        "model_name": "laion/clap-htsat-unfused", "sampling_rate": 48000,
    }


def test_feature_config_params_slot_overrides_flat_fields():
    cfg = FeatureConfig(kind="fbank", n_mels=64, params={"n_mels": 128, "extra": 1})
    opts = asdict(cfg)
    params = opts.pop("params")
    opts.update(params)
    assert opts["n_mels"] == 128 and opts["extra"] == 1


class _DM:
    """Minimal stand-in that borrows only the gate methods of IndexDataModule (bypasses the constructor)."""

    def __init__(self, kind: str, specaug_enabled: bool):
        self.features_cfg = FeatureConfig(kind=kind)
        self.specaugment_cfg = SpecAugmentOptions(enabled=specaug_enabled)

    _feature_processor_cls = IndexDataModule._feature_processor_cls
    _build_feature_transform = IndexDataModule._build_feature_transform


@pytest.mark.parametrize("kind,expect", [("fbank", True), ("mel", True), ("clap", False), ("sclmeta_mel", False), ("waveform", False)])
def test_specaugment_gate_parity(kind, expect):
    from rsc_toolkit.augmentations.specaugment import SpecAugment

    dm = _DM(kind, specaug_enabled=True)
    out = dm._build_feature_transform(training=True)
    assert (out is not None) is expect
    if expect:
        assert isinstance(out, SpecAugment) and out.policy == "icbhi_ast_sup" and out.mask == "mean"
    assert dm._build_feature_transform(training=False) is None
    assert _DM(kind, specaug_enabled=False)._build_feature_transform(training=True) is None


def test_unknown_kind_lists_available():
    dm = _DM("no_such_kind", specaug_enabled=True)
    with pytest.raises(KeyError, match="Unknown features.kind 'no_such_kind'.*Available"):
        dm._feature_processor_cls()


def test_plugin_kind_with_is_spectrogram_gets_specaugment():
    """A new spectrogram kind gets SpecAugment just by declaring it — previously the whitelist had to be edited."""
    from rsc_toolkit.processors.base import BaseAudioProcessor

    @PROCESSORS.register("_test_plugin_spec")
    class PluginSpec(BaseAudioProcessor):
        is_spectrogram = True

        def make_features(self, audio, sample_id=None):
            return audio

    try:
        assert _DM("_test_plugin_spec", specaug_enabled=True)._build_feature_transform(training=True) is not None
    finally:
        PROCESSORS.pop("_test_plugin_spec", None)
