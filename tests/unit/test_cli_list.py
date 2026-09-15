"""`rsc-toolkit list` — 등록소 목록 서브커맨드 (README·scripts/train.sh 가 안내하던 명령; 예전엔 없었다)."""
from __future__ import annotations

import io
import subprocess
import sys

from rsc_toolkit import cli


def test_list_registries_writes_every_kind_and_known_names():
    buf = io.StringIO()
    assert cli.list_registries([], buf) == 0
    text = buf.getvalue()
    for kind in cli.REGISTRY_KINDS:
        assert f"{kind} (" in text
    assert "  ast_patchmix\n" in text and "  icbhi_score\n" in text and "  split_iter\n" in text


def test_list_registries_selects_kinds_and_rejects_unknown(capsys):
    buf = io.StringIO()
    assert cli.list_registries(["losses"], buf) == 0
    assert buf.getvalue().startswith("losses (") and "models (" not in buf.getvalue()
    assert cli.list_registries(["nope"], io.StringIO()) == 2
    assert "nope" in capsys.readouterr().err


def test_console_entry_list_models():
    r = subprocess.run([sys.executable, "-m", "rsc_toolkit.cli", "list", "models"], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-800:]
    assert r.stdout.startswith("models (") and "cnn6_sft" in r.stdout
