# PortForge · 世界级投资组合优化系统 — 作者：晨星 (CJX0712)
"""回测单测：防泄漏切分 + 输出完整性。"""

from __future__ import annotations

import numpy as np
from portforge.core.config import PortfolioConfig
from portforge.core.seed import set_all
from portforge.data.synthetic import generate_panel
from portforge.eval.backtest import walk_forward


def _small_cfg():
    return PortfolioConfig(n_assets=10, n_periods=1500, seed=21, n_seeds=2)


def test_walk_forward_all_methods_present():
    set_all(21)
    cfg = _small_cfg()
    panel = generate_panel(cfg, seed=21)
    oos, turn = walk_forward(panel, cfg)
    assert set(oos) == {
        "equal_weight",
        "min_variance",
        "max_sharpe",
        "risk_parity",
        "hrp",
        "black_litterman",
        "portfuse",
    }
    for arr in oos.values():
        assert arr.shape[0] > 0
        assert np.all(np.isfinite(arr))
    assert all(0.0 <= v <= 1.0 for v in turn.values())


def test_walk_forward_temporal_split_no_leak():
    # 单再平衡内 train/val/test 三段严格两两不相交（无前视偏差）
    cfg = _small_cfg()
    panel = generate_panel(cfg, seed=21)
    R = panel.returns
    T = R.shape[0]
    rebal, lookback = 63, 600
    t = 2 * lookback
    while t + rebal <= T:
        train_start = max(0, t - 2 * lookback)
        train = set(range(train_start, t - lookback))
        val = set(range(t - lookback, t))
        test = set(range(t, t + rebal))
        # 三段两两不相交
        assert train.isdisjoint(val)
        assert val.isdisjoint(test)
        assert train.isdisjoint(test)
        # 时序：train 全在 val 前，val 全在 test 前
        assert max(train) < min(val) < min(test)
        t += rebal
