# PortForge · 世界级投资组合优化系统 — 作者：晨星 (CJX0712)
"""指标单测：已知序列口径可验证。"""

from __future__ import annotations

import numpy as np
from portforge.eval.metrics import compute_metrics, portfolio_returns


def test_portfolio_returns_matmul():
    w = np.array([0.5, 0.5])
    R = np.array([[0.1, 0.2], [0.0, 0.0]])
    assert np.allclose(portfolio_returns(w, R), [0.15, 0.0])


def test_max_drawdown_known():
    r = np.array([0.10, -0.10, 0.05])
    m = compute_metrics(r, annualization=1, risk_free=0.0)
    # 累乘 1.1 * 0.9 * 1.05 = 1.0395；峰 1.1，谷 0.99 → 回撤 -0.10
    assert abs(m.max_drawdown - (-0.10)) < 1e-9


def test_sharpe_sign_and_scale():
    r = np.array([0.01] * 50)  # 零波动 → 守卫返回 0
    m = compute_metrics(r, annualization=252, risk_free=0.0)
    assert m.sharpe == 0.0
    r2 = np.array([0.02, -0.01, 0.03, -0.005, 0.015])
    m2 = compute_metrics(r2, annualization=252, risk_free=0.0)
    assert np.isfinite(m2.sharpe)
    assert m2.sharpe > 0  # 正超额收益 / 正波动


def test_cvar_positive_tail_loss():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.02, 2000)
    m = compute_metrics(r, annualization=252, risk_free=0.0, cvar_alpha=0.95)
    # 5% 尾部期望损失应为正（损失幅度）
    assert m.cvar95 >= 0.0
