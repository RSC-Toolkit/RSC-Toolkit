"""
Port-equivalence test for RespireNet CBA (concatenation-based augmentation).

new_augment(scale) from the original microsoft/RespireNet image_dataloader.py is extracted via AST and bound to a dummy object;
after seeding the global random, its result must match the toolkit concat_augment(classwise, scale, random.Random(seed))
in everything (audio, labels, file names, idx). The test passes only if even the RNG call order is the same.
"""
import ast
import pathlib
import random

import numpy as np
import pytest

from rsc_toolkit.datasets.respirenet_chunks import concat_augment

REF = pathlib.Path("/raid/doyoung/projects/RSC-Toolkit/outputs/reference/respirenet/image_dataloader.py")
needs_ref = pytest.mark.skipif(not REF.exists(), reason="original RespireNet repository missing")


def _load_original_new_augment():
    tree = ast.parse(REF.read_text())
    fn = next(n for c in tree.body if isinstance(c, ast.ClassDef) for n in c.body
              if isinstance(n, ast.FunctionDef) and n.name == "new_augment")
    mod = ast.Module(body=[fn], type_ignores=[])
    ns = {"np": np, "random": random}
    exec(compile(mod, str(REF), "exec"), ns)
    return ns["new_augment"]


class _Dummy:
    def __init__(self, classwise):
        self.classwise_cycle_list = [list(c) for c in classwise]
        self.cycle_list = []
        self.filenames_with_labels = []


def _synthetic_classwise(rng, counts=(40, 17, 9, 5)):
    out = [[], [], [], []]
    for c, n in enumerate(counts):
        for k in range(n):
            audio = rng.standard_normal(rng.integers(200, 900)).astype(np.float32)
            out[c].append((audio, c, f"f{c}_{k}", k, 0))
    return out


@needs_ref
@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("scale", [1, 2])
def test_concat_augment_matches_original(seed, scale):
    classwise = _synthetic_classwise(np.random.default_rng(seed))
    orig = _load_original_new_augment()
    d = _Dummy(classwise)
    random.seed(seed)
    orig(d, scale=scale)
    ours = concat_augment(classwise, scale, random.Random(seed))
    assert len(ours) == len(d.cycle_list) == len(d.filenames_with_labels)
    for (a, lab, name, idx), o, fl in zip(ours, d.cycle_list, d.filenames_with_labels):
        assert lab == o[1] and name == o[2] and idx == o[3]
        assert fl == f"{name}_{idx}_{lab}"
        np.testing.assert_array_equal(a, o[0])
    # scale=1: the 3 abnormal classes are filled up to the normal count, normal gets 0 extra
    n0 = len(classwise[0])
    got = np.bincount([x[1] for x in ours], minlength=4)
    assert got[0] == (scale - 1) * n0
    for c in (1, 2, 3):
        assert len(classwise[c]) + got[c] == scale * n0


def test_concat_augment_off_when_scale_zero():
    classwise = _synthetic_classwise(np.random.default_rng(0))
    assert concat_augment(classwise, 0, random.Random(0)) == []
