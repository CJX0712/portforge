# PortForge · 世界级投资组合优化系统
# 作者：晨星 (CJX0712)
"""马科维茨均值-方差分配器（scipy SLSQP 二次规划）。

- MaxSharpe：在训练窗估计矩，最大化样本外 Sharpe（(w·μ - rf)/√(wᵀΣw)）。
- MinVariance：最小化组合方差。
二者均依赖 Ledoit-Wolf 收缩协方差（见 base）以缓解样本协方差病态。
求解失败自动降级等权（诚实兜底，不崩溃）。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..core.types import ReturnPanel
from .base import PortfolioBase


def _solve_qp(
    mu: np.ndarray,
    Sigma: np.ndarray,
    min_weight: float,
    max_weight: float,
    objective: str,
    risk_free: float = 0.0,
) -> np.ndarray:
    n = len(mu)
    x0 = np.full(n, 1.0 / n)

    def neg_sharpe(w: np.ndarray) -> float:
        ret = float(w @ mu - risk_free)
        vol = float(np.sqrt(max(w @ Sigma @ w, 1e-18)))
        return -ret / vol

    def variance(w: np.ndarray) -> float:
        return float(w @ Sigma @ w)

    fun = neg_sharpe if objective == "max_sharpe" else variance
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(min_weight, max_weight)] * n
    res = minimize(
        fun,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not res.success or not np.all(np.isfinite(res.x)) or abs(res.x.sum() - 1.0) > 1e-3:
        # 诚实降级：等权
        return x0
    w = res.x
    # 投影到单纯形（防数值越界）
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        return x0
    return w / w.sum()


class MaxSharpeAllocator(PortfolioBase):
    name = "max_sharpe"

    def __init__(
        self,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        risk_free: float = 0.0,
    ) -> None:
        super().__init__(min_weight, max_weight)
        self.risk_free = float(risk_free)

    def _solve(self, panel: ReturnPanel) -> np.ndarray:
        assert self._mu is not None and self._Sigma is not None
        return _solve_qp(
            self._mu, self._Sigma, self.min_weight, self.max_weight, "max_sharpe", self.risk_free
        )


class MinVarianceAllocator(PortfolioBase):
    name = "min_variance"

    def _solve(self, panel: ReturnPanel) -> np.ndarray:
        assert self._mu is not None and self._Sigma is not None
        return _solve_qp(self._mu, self._Sigma, self.min_weight, self.max_weight, "min_variance")
