# PortForge · 世界级投资组合优化系统 — 作者：晨星 (CJX0712)
"""CLI 冒烟单测。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portforge.cli import main


def test_cli_determinism_returns_zero():
    assert main(["determinism"]) == 0


def test_cli_run_writes_json(tmp_path):
    out = tmp_path / "bench.json"
    rc = main(["run", "--out", str(out)])
    assert rc == 0
    assert out.exists()
    import json

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["system"] == "PortForge"
    assert any(r["is_flagship"] for r in data["rows"])
