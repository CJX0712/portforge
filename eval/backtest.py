# PortForge · 世界级投资组合优化系统
# 作者：晨星 (CJX0712)
"""时序 walk-forward 回测（防泄漏的黄金标准）。

每个再平衡点 t：
  train = [t-L, t-2·rebal)   ← 估计矩 / 训练分配器
  val   = [t-2·rebal, t)     ← PortFuse 学凸组合（验证窗，不放测试）
  test  = [t, t+rebal)       ← 盲评（OOS 收益计入指标）
严格时序，val 与 test 不重叠、都不进 train；PortFuse 的 λ 完全在 val 上学，
测试窗绝不参与拟合 → 无前视偏差。
"""

from __future__ import annotations

import numpy as np

from ..core.config import PortfolioConfig
from ..core.seed import set_all
from ..core.types import ReturnPanel
from ..data.synthetic import generate_panel
from ..portfolio import BASE_ALLOCATORS, PortFuseAllocator
from .metrics import portfolio_returns


def _instantiate(cfg: PortfolioConfig):
    base = [A() for A in BASE_ALLOCATORS]
    fuse = PortFuseAllocator(annualization=cfg.annualization, blend_reg=cfg.blend_reg)
    return base, fuse


def walk_forward(
    panel: ReturnPanel,
    cfg: PortfolioConfig,
    rebal: int = 63,
    lookback: int = 600,
):
    """返回 (oos_returns: dict[name->array], turnover_mean: dict[name->float])。

    turnover = 相邻再平衡点权重 L1 距离均值 / 2（真实换手率）。
    """
    R = panel.returns
    T, _ = R.shape
    base, fuse = _instantiate(cfg)
    names = [b.name for b in base] + ["portfuse"]

    oos: dict[str, list[float]] = {nm: [] for nm in names}
    turn_acc: dict[str, float] = {nm: 0.0 for nm in names}
    turn_cnt: dict[str, int] = {nm: 0 for nm in names}
    last_w: dict[str, np.ndarray] = {}

    # 时序切分：train=[t-2L, t-L) 估计矩；val=[t-L, t) 学 λ；test=[t, t+rebal) 盲评
    t = 2 * lookback
    while t + rebal <= T:
        val_end = t
        train_start = max(0, t - 2 * lookback)
        train = ReturnPanel(returns=R[train_start : t - lookback], assets=panel.assets)
        val = ReturnPanel(returns=R[t - lookback : val_end], assets=panel.assets)
        test = R[t : t + rebal]

        # 训练各基线
        candidates: dict[str, np.ndarray] = {}
        for b in base:
            b.fit(train)
            w = b.allocate()
            candidates[b.name] = w
            oos[b.name].extend(portfolio_returns(w, test).tolist())
            if b.name in last_w:
                turn_acc[b.name] += np.abs(w - last_w[b.name]).sum() / 2.0
                turn_cnt[b.name] += 1
            last_w[b.name] = w

        # 旗舰：在 val 上学凸组合，test 盲评
        fuse.fit(val, candidates)
        wf = fuse.allocate()
        oos["portfuse"].extend(portfolio_returns(wf, test).tolist())
        if "portfuse" in last_w:
            turn_acc["portfuse"] += np.abs(wf - last_w["portfuse"]).sum() / 2.0
            turn_cnt["portfuse"] += 1
        last_w["portfuse"] = wf
        t += rebal

    oos_arr = {nm: np.asarray(v, dtype=np.float64) for nm, v in oos.items()}
    turn_mean = {nm: (turn_acc[nm] / turn_cnt[nm] if turn_cnt[nm] else 0.0) for nm in names}
    return oos_arr, turn_mean


def backtest_seed(cfg: PortfolioConfig, seed: int) -> dict[str, np.ndarray]:
    set_all(seed)
    panel = generate_panel(cfg, seed=seed)
    return walk_forward(panel, cfg)
