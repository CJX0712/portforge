# PortForge · 世界级投资组合优化系统
# 作者：晨星 (CJX0712)
"""风险平价 ERC（Equal Risk Contribution，Spinu 2013 牛顿不动点）。

顶级数学内核：风险平价要求每个资产贡献相同边际风险
``w_i (Σw)_i = 1/n · wᵀΣw``。等价于解 ``Σx = 1/x``（x = w / (wᵀΣw)），
该式是凸问题 ``min ½xᵀΣx − Σln xᵢ`` 的一阶条件，用阻尼牛顿不动点迭代求解，
纯 numpy 零优化器依赖、确定性可复现。SLSQP 直接最小化两两风险贡献差会陷入局部极小，
本实现改用解析梯度/海森的牛顿法，全局唯一收敛到等风险解。
"""

from __future__ import annotations

import numpy as np

from ..core.types import ReturnPanel
from .base import PortfolioBase


def _nearest_pd(M: np.ndarray) -> np.ndarray:
    """投影到最近正定矩阵（对称 + 特征值裁剪），保证海森正定、迭代稳定。"""
    M = (M + M.T) / 2.0
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 1e-10, None)
    return (V * w) @ V.T


def _erc_weights(Sigma: np.ndarray) -> np.ndarray:
    """风险平价 ERC：Spinu(2013) 阻尼牛顿不动点，纯 numpy 确定性求解。

    解 x>0 满足 ``Σx = 1/x``（逐元素），则 ``w = x / Σxᵢ`` 即为等风险贡献组合。
    牛顿迭代 ``x ← x − H⁻¹(Σx − 1/x)``，H = Σ + diag(1/x²) 正定；
    回溯步长保证 x 严格为正，梯度范数 < 1e-12 判敛。
    """
    Sigma = np.asarray(Sigma, dtype=float)
    Sigma = _nearest_pd(Sigma)
    n = Sigma.shape[0]

    # 初值：逆波动加权（稳定、正）
    x = 1.0 / np.sqrt(np.diag(Sigma) + 1e-12)
    x = x / x.sum()

    for _ in range(2000):
        g = Sigma @ x - 1.0 / x  # 梯度 = Σx − 1/x
        H = Sigma + np.diag(1.0 / (x * x))  # 海森正定
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.solve(H + 1e-8 * np.eye(n), g)

        # 回溯保证 x>0（凸问题下牛顿方向在足够小步长内下降且保正）
        alpha = 1.0
        x_new = x - alpha * step
        while np.any(x_new <= 0) and alpha > 1e-4:
            alpha *= 0.5
            x_new = x - alpha * step
        x = np.clip(x_new, 1e-10, None)

        if np.linalg.norm(g, np.inf) < 1e-12:
            break

    w = x / x.sum()
    return w / w.sum()


class RiskParityAllocator(PortfolioBase):
    name = "risk_parity"

    def _solve(self, panel: ReturnPanel) -> np.ndarray:
        assert self._Sigma is not None
        try:
            w = _erc_weights(self._Sigma)
            if not np.all(np.isfinite(w)) or abs(w.sum() - 1.0) > 1e-4:
                raise _err_bad()
            return w
        except Exception:
            # 兜底：逆波动加权（风险平价的一阶近似，确定性可复现）
            inv = 1.0 / np.sqrt(np.diag(self._Sigma) + 1e-12)
            return inv / inv.sum()


def _err_bad() -> Exception:
    from ..core import errors as _err

    return _err.AllocError("ERC 牛顿不动点未收敛")
