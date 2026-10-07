# PortForge · 世界级投资组合优化系统
# 作者：晨星 (CJX0712)
"""黑利特曼 BL（Black & Litterman 1992, Goldman Sachs）纯 numpy 贝叶斯后验。

顶级数学内核：以市场均衡收益 π=δ·Σ·w_mkt 为先验，注入少量观点（这里由
训练窗样本 Sharpe 选出 top 资产「跑赢均值」观点），经贝叶斯更新得到后验预期收益：

  M = (τΣ)⁻¹ + PᵀΩ⁻¹P ;  μ_BL = M⁻¹[(τΣ)⁻¹π + PᵀΩ⁻¹q]

后验收益正部归一化即为长仓权重。零外部依赖、确定性可复现。
"""

from __future__ import annotations

import numpy as np

from ..core.types import ReturnPanel
from .base import PortfolioBase


class BlackLittermanAllocator(PortfolioBase):
    name = "black_litterman"
    _TAU = 0.05
    _DELTA = 2.5  # 风险厌恶

    def _solve(self, panel: ReturnPanel) -> np.ndarray:
        assert self._mu is not None and self._Sigma is not None
        n = len(self._mu)
        Sigma = self._Sigma + 1e-10 * np.eye(n)
        w_mkt = np.full(n, 1.0 / n)
        pi = self._DELTA * (Sigma @ w_mkt)
        mu = self._mu

        # 观点：训练窗 top-K 资产「跑赢等权均值」
        k = max(1, n // 5)
        sort_idx = np.argsort(mu)[::-1]
        pick = sort_idx[:k]
        P = np.zeros((k, n))
        q = np.zeros(k)
        for i, a in enumerate(pick):
            P[i, a] = 1.0
            q[i] = max(0.0, mu[a] - mu.mean())
        # 观点置信：收益离差越大越确信
        Omega = np.diag(np.maximum((P @ (Sigma / np.sqrt(n)) @ P.T).diagonal(), 1e-8))

        tS = self._TAU * Sigma
        try:
            inv_tS = np.linalg.inv(tS)
            M = inv_tS + P.T @ np.linalg.inv(Omega) @ P
            rhs = inv_tS @ pi + P.T @ np.linalg.inv(Omega) @ q
            post_mu = np.linalg.solve(M, rhs)
        except Exception:
            post_mu = pi.copy()

        w = np.clip(post_mu, 0.0, None)
        if w.sum() <= 0:
            return w_mkt
        return w / w.sum()
