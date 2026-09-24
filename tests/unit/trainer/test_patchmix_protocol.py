"""
Tests for the patch-mix training protocol components.

Model and data are pinned by goldens (other tests); here the protocol pieces are checked
to produce the same numbers as the original formulas:
- LiveWeightAveraging == util/misc.py update_moving_average (old*beta + (1-beta)*new)
- PaperCosineLR == util/misc.py adjust_learning_rate
- ICBHIBestScoreCheckpoint == the "best Score && Se>5" saving rule in main.py
- WeightedCrossEntropyFromCounts == the main.py --weighted_loss formula (compared against golden values)
"""
import math
import pathlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from rsc_toolkit.trainer.callbacks.icbhi_best_checkpoint import ICBHIBestScoreCheckpoint
from rsc_toolkit.trainer.callbacks.ma_update import LiveWeightAveraging
from rsc_toolkit.trainer.callbacks.paper_cosine_lr import PaperCosineLR
from rsc_toolkit.trainer.losses import WeightedCrossEntropyFromCounts


def _repo_root() -> pathlib.Path:
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("repository root not found")


GOLDEN = _repo_root() / "tests" / "golden" / "patchmix"


def test_live_weight_averaging_blends_with_previous_iteration():
    model = nn.Linear(2, 2, bias=False)
    pl_module = SimpleNamespace(rsc_model=model)
    cb = LiveWeightAveraging(beta=0.5)

    with torch.no_grad():
        model.weight.copy_(torch.ones(2, 2))
    cb.on_train_batch_start(None, pl_module, None, 0)

    with torch.no_grad():
        model.weight.copy_(torch.zeros(2, 2))  # mimic optimizer.step()
    cb.on_train_batch_end(None, pl_module, None, None, 0)

    # original: old*beta + (1-beta)*new = 1*0.5 + 0.5*0 = 0.5
    assert torch.allclose(model.weight, torch.full((2, 2), 0.5))


def test_live_weight_averaging_includes_buffers():
    """The original update_moving_average averages the whole state_dict (including buffers)."""
    model = nn.BatchNorm1d(3)
    pl_module = SimpleNamespace(rsc_model=model)
    cb = LiveWeightAveraging(beta=0.5)

    with torch.no_grad():
        model.running_mean.copy_(torch.ones(3))
    cb.on_train_batch_start(None, pl_module, None, 0)
    with torch.no_grad():
        model.running_mean.copy_(torch.zeros(3))
    cb.on_train_batch_end(None, pl_module, None, None, 0)

    assert torch.allclose(model.running_mean, torch.full((3,), 0.5))


def test_paper_cosine_lr_matches_reference_formula():
    lr, epochs, rate = 5e-5, 50, 0.1
    cb = PaperCosineLR(lr, epochs, rate)
    eta_min = lr * rate ** 3
    for epoch in (1, 10, 25, 50):
        expected = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / epochs)) / 2
        assert cb.lr_at(epoch) == pytest.approx(expected, rel=1e-12)
    assert cb.lr_at(epochs) == pytest.approx(eta_min, rel=1e-9)


def test_paper_cosine_lr_sets_optimizer_lr():
    opt = torch.optim.Adam([nn.Parameter(torch.zeros(1))], lr=123.0)
    trainer = SimpleNamespace(current_epoch=0, optimizers=[opt])
    cb = PaperCosineLR(5e-5, 50)
    cb.on_train_epoch_start(trainer, None)
    assert opt.param_groups[0]["lr"] == pytest.approx(cb.lr_at(1))


def _fake_trainer(score, se, sp=80.0, epoch=4):
    return SimpleNamespace(
        sanity_checking=False,
        current_epoch=epoch,
        callback_metrics={
            "icbhi_score": torch.tensor(score),
            "S_e": torch.tensor(se),
            "S_p": torch.tensor(sp),
        },
    )


def test_best_checkpoint_requires_se_above_threshold(tmp_path):
    """Original: `if sc > best_acc[-1] and se > 5` — nothing is saved unless Se passes the threshold."""
    cb = ICBHIBestScoreCheckpoint(dirpath=str(tmp_path))
    pl_module = SimpleNamespace(rsc_model=nn.Linear(2, 2))

    cb.on_validation_end(_fake_trainer(score=59.0, se=4.9), pl_module)
    assert not (tmp_path / "best.pth").exists()

    cb.on_validation_end(_fake_trainer(score=58.0, se=40.0), pl_module)
    assert (tmp_path / "best.pth").exists()
    saved = torch.load(tmp_path / "best.pth")
    assert saved["icbhi_score"] == pytest.approx(58.0)
    assert "model" in saved

    # a lower score does not update
    cb.on_validation_end(_fake_trainer(score=57.0, se=45.0), pl_module)
    assert torch.load(tmp_path / "best.pth")["icbhi_score"] == pytest.approx(58.0)


def test_best_checkpoint_skips_sanity_check(tmp_path):
    cb = ICBHIBestScoreCheckpoint(dirpath=str(tmp_path))
    trainer = _fake_trainer(score=99.0, se=50.0)
    trainer.sanity_checking = True
    cb.on_validation_end(trainer, SimpleNamespace(rsc_model=nn.Linear(2, 2)))
    assert not (tmp_path / "best.pth").exists()


@pytest.mark.skipif(not (GOLDEN / "loss_goldens.pt").exists(), reason="golden missing")
def test_weighted_ce_from_counts_matches_reference_golden():
    """The golden weighted values were computed with the reference formula (main.py:264-269)."""
    from rsc_toolkit.methods.patchmix import PatchMixLoss

    g = torch.load(GOLDEN / "loss_goldens.pt")
    ce = WeightedCrossEntropyFromCounts(g["class_nums"])
    got = float(PatchMixLoss(ce)(g["pred"], g["y_a"], g["y_b"], g["lam"]))
    assert got == pytest.approx(g["patchmix_loss_weighted"], abs=1e-6)


def test_patchmix_run_config_composes_with_paper_protocol():
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    with initialize_config_module(config_module="rsc_toolkit.configs", version_base=None):
        cfg = compose(config_name="patchmix_icbhi")

    assert cfg.model.name == "ast_patchmix"
    assert cfg.model.params.method == "ce"
    assert cfg.model.params.input_fdim == 798 and cfg.model.params.input_tdim == 128  # quirk
    assert cfg.loss.name == "cross_entropy_from_counts"
    assert cfg.seed == 1
    assert cfg.data.params.pad_type == "repeat_fade"
    assert cfg.data.params.features.target_frames == 798
    assert cfg.data.params.drop_last_train is True
    assert list(cfg.data.params.val_splits) == ["test"]
    assert cfg.trainer.params.precision == "16-mixed"

    callbacks = OmegaConf.to_container(cfg.trainer.params.callbacks, resolve=False)
    targets = [c["_target_"] for c in callbacks]
    assert any("ma_update.LiveWeightAveraging" in t for t in targets)
    assert any("paper_cosine_lr.PaperCosineLR" in t for t in targets)
    assert any("icbhi_best_checkpoint.ICBHIBestScoreCheckpoint" in t for t in targets)
    ma = callbacks[targets.index("rsc_toolkit.trainer.callbacks.ma_update.LiveWeightAveraging")]
    assert ma["beta"] == 0.5


ICBHI_INDEX = _repo_root() / ".datasets" / "icbhi" / "index" / "index_cycle.csv"


@pytest.mark.skipif(not ICBHI_INDEX.exists(), reason="real ICBHI index missing")
def test_weighted_ce_class_nums_match_real_official_train_split():
    """class_nums hard-coded in the config match the real official train distribution (drift guard)."""
    from rsc_toolkit.datasets.index_dataset import IndexDataset

    ds = IndexDataset(
        index_csv=ICBHI_INDEX, split="train",
        split_file=_repo_root() / "rsc_toolkit" / "datasets" / "icbhi" / "data" / "official_split.txt",
    )
    counts = ds.df["label"].value_counts().sort_index().tolist()
    assert counts == [2063, 1215, 501, 363]
