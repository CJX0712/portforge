# PortForge · 世界级投资组合优化系统
# 作者：晨星 (CJX0712)
"""分配器基类：统一 fit（估计矩）/ allocate 骨架 + Ledoit-Wolf 收缩协方差。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.interfaces import assert_long_only
from ..core.types import ReturnPanel


class PortfolioBase(ABC):
    """所有单分配器的公共基类。

    硬契约：``fit`` 只在 ``panel``（训练窗）上估计矩，绝不窥视 val/test。
    """

    name: str = "base"

    def __init__(self, min_weight: float = 0.0, max_weight: float = 1.0) -> None:
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self._mu: np.ndarray | None = None
        self._Sigma: np.ndarray | None = None
        self._weights: np.ndarray | None = None

    # ---- 矩估计（Ledoit-Wolf 收缩，顶级协方差估计） ----
    def _estimate_moments(self, panel: ReturnPanel) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.covariance import LedoitWolf

        R = panel.returns
        mu = R.mean(axis=0)
        lw = LedoitWolf().fit(R)
        Sigma = lw.covariance_
        # 数值保安：对称 + 投影到最近 PSD。
        # 注意：必须用特征值投影而非逐元素裁剪——逐元素裁剪会把负相关协方差
        # 抬到 1e-10，破坏资产间负相关结构，导致下游 ERC/HRP 等权重失真。
        Sigma = (Sigma + Sigma.T) / 2.0
        w, V = np.linalg.eigh(Sigma)
        w = np.clip(w, 1e-10, None)
        Sigma = (V * w) @ V.T
        return mu, Sigma

    def fit(self, panel: ReturnPanel) -> PortfolioBase:
        self._mu, self._Sigma = self._estimate_moments(panel)
        self._weights = self._solve(panel)
        assert_long_only(self._weights)
        return self

    @abstractmethod
    def _solve(self, panel: ReturnPanel) -> np.ndarray:
        """子类实现：返回多头长仓权重（sum=1, >=0）。"""

    def allocate(self) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("请先调用 fit()")
        return self._weights.copy()
