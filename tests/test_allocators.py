# PortForge · 世界级投资组合优化系统 — 作者：晨星 (CJX0712)
"""分配器族单测：多头长仓契约 + 各方法数学性质。"""

from __future__ import annotations

import numpy as np
from portforge.core.config import PortfolioConfig
from portforge.core.seed import set_all
from portforge.core.types import ReturnPanel
from portforge.data.synthetic import generate_panel
from portforge.portfolio import (
    BASE_ALLOCATORS,
    BlackLittermanAllocator,
    EqualWeightAllocator,
    HRPAllocator,
    MaxSharpeAllocator,
    MinVarianceAllocator,
    RiskParityAllocator,
)


def _panel(n=12, T=400, seed=1) -> ReturnPanel:
    cfg = PortfolioConfig(n_assets=n, n_periods=T, seed=seed)
    return generate_panel(cfg, seed=seed)


def test_all_base_allocators_long_only_unit_sum():
    set_all(3)
    panel = _panel()
    for A in BASE_ALLOCATORS:
        w = A().fit(panel).allocate()
        assert w.shape == (panel.returns.shape[1],)
        assert np.all(w >= -1e-9)
        assert abs(w.sum() - 1.0) < 1e-4


def test_min_variance_lowers_variance_vs_equal():
    set_all(5)
    panel = _panel(n=15, T=600)
    w_eq = EqualWeightAllocator().fit(panel).allocate()
    w_mv = MinVarianceAllocator().fit(panel).allocate()
    cov = np.cov(panel.returns, rowvar=False)
    var_eq = w_eq @ cov @ w_eq
    var_mv = w_mv @ cov @ w_mv
    assert var_mv <= var_eq + 1e-6


def test_risk_parity_equal_risk_contribution():
    # ERC 在 LedoitWolf 收缩协方差上等风险（与 allocator 同口径核对）
    set_all(9)
    panel = _panel(n=10, T=800)
    w = RiskParityAllocator().fit(panel).allocate()
    from sklearn.covariance import LedoitWolf

    cov = LedoitWolf().fit(panel.returns).covariance_  # 与 base._estimate_moments 一致
    rc = w * (cov @ w)  # 边际风险贡献
    rc_norm = rc / rc.sum()
    assert rc_norm.std() < 0.02  # 顶级 ERC 性质：各资产风险贡献近似相等


def test_hrp_weights_positive_and_sum_one():
    set_all(11)
    panel = _panel(n=14, T=700)
    w = HRPAllocator().fit(panel).allocate()
    assert np.all(w > 0)
    assert abs(w.sum() - 1.0) < 1e-4


def test_black_litterman_long_only():
    set_all(13)
    panel = _panel(n=12, T=600)
    w = BlackLittermanAllocator().fit(panel).allocate()
    assert np.all(w >= -1e-9)
    assert abs(w.sum() - 1.0) < 1e-4


def test_max_sharpe_runs():
    set_all(17)
    panel = _panel(n=10, T=500)
    w = MaxSharpeAllocator().fit(panel).allocate()
    assert abs(w.sum() - 1.0) < 1e-3
