"""META_LABELS registry + declaration and checking of the batch[2] (meta) side channel.

(a) if-chain of meta_label_mode -> registry. The 4 built-in modes keep the same functions and names (parity is trivial); an unknown mode
    fails with the list of available ones (also satisfying the NotImplementedError the existing tests expect); plugin modes use a decorator.
(b) datamodule `meta_kind` and model `accepts_meta` are checked once in adapter setup — undeclared combinations pass (opt-in);
    if both sides are declared and disagree, TypeError before start (blocks silent mis-training such as reading an AQS float as a stethoscope id).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from torch import nn

from rsc_toolkit.datasets.meta_labels import build_meta_label_fn
from rsc_toolkit.registries import META_LABELS, MODELS
from rsc_toolkit.trainer.adapters.pl_adapter import PLAdapter, check_meta_contract


def test_builtin_meta_label_modes_snapshot():
    assert set(META_LABELS) == {"dev", "patient", "sa", "sa_sprs"}


def test_builtin_modes_resolve_to_the_same_functions():
    assert build_meta_label_fn("dev")({"meta": {"device": "Litt3200"}}) == 2
    assert build_meta_label_fn("patient")({"meta": {"patient_id": "226"}}) == 226
    assert build_meta_label_fn("sa")({"meta": {"sex": "F", "adult_BMI": 22.0}}) == 1
    assert build_meta_label_fn("sa_sprs")({"meta": {"gender": 1, "age": 7.0}}) == 3


def test_unknown_mode_fails_with_available_list_and_keeps_legacy_exception_type():
    with pytest.raises(NotImplementedError, match="Available"):
        build_meta_label_fn("loc")
    with pytest.raises(KeyError):
        build_meta_label_fn("loc")


def test_plugin_mode_via_decorator():
    @META_LABELS.register("_test_hospital")
    def build_hospital():
        return lambda item: int(item["meta"]["hospital"])

    try:
        assert build_meta_label_fn("_test_hospital")({"meta": {"hospital": 7}}) == 7
    finally:
        META_LABELS.pop("_test_hospital", None)


# ── (b) declarations ───────────────────────────────────────────────────────────────────
def test_in_tree_datamodules_declare_meta_kind():
    from rsc_toolkit.datasets.amsd_data import AMSDDataModule
    from rsc_toolkit.datasets.bts_precomputed import BTSPrecomputedDataModule
    from rsc_toolkit.datasets.btscard_sprs import BTSCARDSprsDataModule
    from rsc_toolkit.datasets.lungmix_data import LungmixDataModule
    from rsc_toolkit.datasets.pcmcl_concat import PCMCLDataModule
    from rsc_toolkit.datasets.qlung_data import QLungDataModule
    from rsc_toolkit.datasets.respirenet_chunks import RespireNetDataModule
    from rsc_toolkit.trainer.data.index_datamodule import IndexDataModule

    assert QLungDataModule.meta_kind == "qlung_aqs"
    assert PCMCLDataModule.meta_kind == "pcmcl_ssl_label"
    assert BTSPrecomputedDataModule.meta_kind == "bts_meta7"
    assert BTSCARDSprsDataModule.meta_kind == "bts_meta7"
    assert AMSDDataModule.meta_kind == "amsd_sample_index"
    assert RespireNetDataModule.meta_kind == "respirenet_chunk_index"
    assert LungmixDataModule.meta_kind == "lungmix_valid_len"
    # generic IndexDataModule uses meta_label_mode directly as kind (None when disabled)
    assert IndexDataModule.meta_kind.fget(SimpleNamespace(meta_label_mode="dev")) == "dev"
    assert IndexDataModule.meta_kind.fget(SimpleNamespace(meta_label_mode=None)) is None


@pytest.mark.parametrize(
    "model_name,kind",
    [
        ("ast_sgscl", "dev"), ("resnet18_sft", "dev"), ("cnn6_sft", "dev"),
        ("beats", "patient"), ("cnn6_sclmeta", "sa"), ("cnn6_sclmeta", "sa_sprs"),
        ("ast_qlung", "qlung_aqs"), ("clap_qlung", "qlung_aqs"),
        ("ast_pcmcl", "pcmcl_ssl_label"), ("beats_pcmcl", "pcmcl_ssl_label"),
        ("bts", "bts_meta7"), ("bts_card", "bts_meta7"), ("audioclap", "bts_meta7"),
        ("ast_amsd", "amsd_sample_index"), ("respirenet", "respirenet_chunk_index"),
        ("ast_lungmix", "lungmix_valid_len"),
    ],
)
def test_in_tree_models_declare_accepts_meta(model_name, kind):
    assert kind in MODELS.get(model_name).accepts_meta


# ── (b) checks ─────────────────────────────────────────────────────────────────────────
class _Model(nn.Module):
    def __init__(self, accepts=None):
        super().__init__()
        if accepts is not None:
            self.accepts_meta = accepts
        self.head = nn.Linear(2, 2)

    def training_forward(self, audio_batch, labels, loss_func, **kwargs):
        logits = self.head(audio_batch)
        return logits, loss_func(logits, labels)


def _dm(kind):
    return SimpleNamespace(meta_kind=kind)


def test_check_meta_contract_truth_table():
    check_meta_contract(_Model(), _dm(None))                   # neither declared
    check_meta_contract(_Model(), _dm("dev"))                  # model undeclared -> pass (opt-in)
    check_meta_contract(_Model(("dev",)), _dm(None))           # datamodule sends no meta -> pass
    check_meta_contract(_Model(("dev",)), _dm("dev"))          # match
    check_meta_contract(_Model(("sa", "sa_sprs")), _dm("sa_sprs"))
    check_meta_contract(_Model(("dev",)), None)                # no datamodule (script path) -> pass
    with pytest.raises(TypeError, match="qlung_aqs"):
        check_meta_contract(_Model(("dev",)), _dm("qlung_aqs"))


def test_pl_adapter_setup_runs_the_check_against_trainer_datamodule():
    adapter = PLAdapter(rsc_model=_Model(("dev",)), loss_func=nn.CrossEntropyLoss(),
                        optimizer_cfg={"name": "sgd", "params": {"lr": 0.1}})
    adapter._trainer = SimpleNamespace(datamodule=_dm("qlung_aqs"))
    with pytest.raises(TypeError, match="accepts_meta"):
        adapter.setup("fit")
    adapter._trainer = SimpleNamespace(datamodule=_dm("dev"))
    adapter.setup("fit")
    adapter._trainer = None  # no trainer attached (unit tests, scripts) does not crash either
    adapter.setup("fit")
