"""Bundle selection of scripts/download_paper_weights.py on Drive folder listings."""
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("download_paper_weights", ROOT / "scripts" / "download_paper_weights.py")
dpw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dpw)


def _bundle(*prefix):
    return [(f"{'/'.join(prefix)}/{name}", (*prefix, *name.split("/"))) for name in ("best.pth", "best.json", ".hydra/config.yaml", "sha256.txt")]


def test_seed1_run_named_like_its_family_selects_only_its_own_bundle():
    entries = []
    for run in ("icbhi-patchmix-ce", "icbhi-patchmix-ce-s2", "icbhi-patchmix-ce-s3"):
        entries += _bundle("icbhi", "icbhi-patchmix-ce", run)
    entries += _bundle("icbhi", "icbhi-cnn6-ce", "icbhi-cnn6-ce-s1")
    wanted, problem = dpw.select_bundle_files(entries, "icbhi-patchmix-ce")
    assert problem is None
    assert sorted("/".join(p) for _, p in wanted) == [".hydra/config.yaml", "best.json", "best.pth", "sha256.txt"]
    assert all(fid.startswith("icbhi/icbhi-patchmix-ce/icbhi-patchmix-ce/") for fid, _ in wanted)


def test_suffixed_run_and_single_bundle_folder():
    entries = _bundle("icbhi", "icbhi-cnn6-ce", "icbhi-cnn6-ce-s1") + _bundle("icbhi", "icbhi-cnn6-ce", "icbhi-cnn6-ce-s2")
    wanted, problem = dpw.select_bundle_files(entries, "icbhi-cnn6-ce-s2")
    assert problem is None and {fid.split("/")[2] for fid, _ in wanted} == {"icbhi-cnn6-ce-s2"}
    single = _bundle("somefolder")
    wanted, problem = dpw.select_bundle_files(single, "icbhi-cnn6-ce-s1")
    assert problem is None and len(wanted) == 4


def test_missing_run_reports_a_problem():
    entries = _bundle("icbhi", "icbhi-cnn6-ce", "icbhi-cnn6-ce-s1") + _bundle("icbhi", "icbhi-cnn6-ce", "icbhi-cnn6-ce-s2")
    wanted, problem = dpw.select_bundle_files(entries, "icbhi-bts-s1")
    assert wanted == [] and "icbhi-bts-s1" in problem
