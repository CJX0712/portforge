# PortForge · 世界级投资组合优化系统 — 作者：晨星 (CJX0712)
"""pipeline 单测：跨 seed 聚合 + 确定性逐位一致。"""

from __future__ import annotations

from portforge.core.config import PortfolioConfig
from portforge.pipeline.portfolio_pipeline import PortfolioPipeline


def _cfg():
    return PortfolioConfig(n_assets=8, n_periods=1600, seed=101, n_seeds=2)


def test_pipeline_rows_complete():
    rep = PortfolioPipeline(_cfg()).run()
    names = {r["name"] for r in rep.rows}
    assert "portfuse" in names
    assert len(names) == 7
    assert rep.passed in (True, False)  # 必为布尔（如实判定）


def test_pipeline_determinism_bit_identical():
    cfg = _cfg()
    r1 = PortfolioPipeline(cfg).run()
    r2 = PortfolioPipeline(cfg).run()
    for a, b in zip(r1.rows, r2.rows):
        assert a["name"] == b["name"]
        assert a["sharpe_mean"] == b["sharpe_mean"]
        assert a["ann_return_mean"] == b["ann_return_mean"]
