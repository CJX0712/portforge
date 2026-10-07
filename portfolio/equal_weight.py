# PortForge · 世界级投资组合优化系统
# 作者：晨星 (CJX0712)
"""等权 1/N 分配器（最经典基线，零估计误差，稳健 OOS 基准）。"""

from __future__ import annotations

import numpy as np

from ..core.types import ReturnPanel
from .base import PortfolioBase


class EqualWeightAllocator(PortfolioBase):
    name = "equal_weight"

    def _solve(self, panel: ReturnPanel) -> np.ndarray:
        n = panel.returns.shape[1]
        return np.full(n, 1.0 / n)
