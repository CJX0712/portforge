# PortForge · 世界级投资组合优化系统
# 作者：晨星 (CJX0712)
"""样本外绩效指标（统一口径，跨分配器公平比较）。

所有指标基于组合收益序列 r_t = R @ w。语义统一：Sharpe 越大越好。
"""

from __future__ import annotations

import numpy as np

from ..core.types import Metrics


def portfolio_returns(weights: np.ndarray, returns: np.ndarray) -> np.ndarray:
    return np.asarray(returns, dtype=np.float64) @ np.asarray(weights, dtype=np.float64)


def compute_metrics(
    r: np.ndarray,
    annualization: int = 252,
    risk_free: float = 0.0,
    cvar_alpha: float = 0.95,
    turnover: float = 0.0,
) -> Metrics:
    r = np.asarray(r, dtype=np.float64).ravel()
    if r.size == 0:  # 防御：空收益序列（如回测窗为 0）返回全 0，不崩
        return Metrics(turnover=turnover)
    ann = annualization
    rf_per = risk_free / ann
    mean = r.mean()
    std = r.std()
    excess = mean - rf_per
    ann_return = mean * ann
    ann_vol = std * np.sqrt(ann)
    sharpe = (excess / std) * np.sqrt(ann) if std > 0 else 0.0
    downside = r[r < rf_per] - rf_per
    dsd = np.sqrt((downside**2).mean()) if downside.size > 0 else 0.0
    sortino = (excess / dsd) * np.sqrt(ann) if dsd > 0 else 0.0
    # 最大回撤
    cum = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(dd.min()) if dd.size else 0.0
    # CVaR（期望尾部损失，正值表示损失幅度）
    q = np.quantile(r, cvar_alpha)
    tail = r[r <= q]
    cvar = float(-tail.mean()) if tail.size else 0.0
    return Metrics(
        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        cvar95=cvar,
        turnover=turnover,
    )
