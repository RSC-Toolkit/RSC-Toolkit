"""표 2 #Param (M) 열 — params.csv 로더와 셀 서식 (2026-09-15 사용자 요청: M 단위, 소수점 첫째 자리)."""
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("build_paper_tables", ROOT / "scripts" / "paper" / "build_paper_tables.py")
bpt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bpt)


def test_param_cell_prefers_sprsound_side_and_formats_one_decimal():
    params = {"sprs-cnn6-ce": 4_312_345, "icbhi-cnn6-ce": 4_310_000}
    b, c = {"family": "sprs-cnn6-ce"}, {"family": "icbhi-cnn6-ce"}
    assert bpt._param_cell(b, c, params) == "4.3"
    assert bpt._param_cell(None, c, params) == "4.3"          # SPR→SPR 이 없는 행(RespireNet)은 ICBHI 쪽
    assert bpt._param_cell(b, None, {}) == "--"               # params.csv 에 없으면 --
    assert bpt._param_cell({"family": "sprs-mvst-fusion"}, None, {"sprs-mvst-fusion": 440_930_000}) == "440.9"


def test_load_params_reads_repo_csv_when_present():
    params = bpt.load_params()
    if not bpt.PARAMS.exists():
        assert params == {}
        return
    assert all(isinstance(v, int) and v > 0 for v in params.values())
    assert "sprs-cnn6-ce" in params
