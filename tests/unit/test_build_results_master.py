"""results_master 스캐너가 학습 run 이 아닌 best.json(평가 sandbox·배포 번들·튜토리얼)을 거르는지.

2026-09-15: RespireNet ICBHI→SPRSound 제로샷 sandbox 가 남긴 callback_out/<run>/best.json 이 원 run 과
같은 run.name 으로 results_master 에 들어가 표 1 family 집계 n 이 5→10 이 됐다(56.37→51.88).
"""
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("build_results_master", ROOT / "scripts" / "reproduction" / "build_results_master.py")
brm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(brm)

P = pathlib.Path


def test_regular_runs_are_kept():
    assert brm.is_experiment_run(P("respirenet/icbhi-respirenet-cba-s1/best.json"))
    assert brm.is_experiment_run(P("sgscl/icbhi-cnn6-ce-s1/best.json"))
    # 한 겹 더 들어간 변형 런(outputs/<group>/<parent>/<sub>/best.json)도 학습 run 이다
    assert brm.is_experiment_run(P("mvst/mvst-fusion-s1/view-a/best.json"))


def test_eval_sandbox_outputs_are_skipped():
    assert not brm.is_experiment_run(P("ood_sprsound_2026-09-14/sandbox/callback_out/icbhi-respirenet-cba-s1/best.json"))
    assert not brm.is_experiment_run(P("verify_2026-09-14/sandbox/icbhi-cnn6-ce-s1/best.json"))


def test_bundles_tutorial_dates_smoke_are_skipped():
    assert not brm.is_experiment_run(P("weights_release/icbhi-cnn6-ce-s1/best.json"))
    assert not brm.is_experiment_run(P("tutorial/paper_weights/icbhi-cnn6-ce-s1/best.json"))
    assert not brm.is_experiment_run(P("tutorial/outputs/sgscl/icbhi-cnn6-1epoch/best.json"))
    assert not brm.is_experiment_run(P("2026-09-14/21-18-00/best.json"))
    assert not brm.is_experiment_run(P("sgscl/smoke-cnn6/best.json"))
