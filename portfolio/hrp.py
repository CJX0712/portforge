# PortForge · 世界级投资组合优化系统
# 作者：晨星 (CJX0712)
"""层级风险平价 HRP（López de Prado 2016, SSRN 2708678）。

顶级算法流水线：
1. 相关矩阵 → 距离 ``d = √(0.5·(1−ρ))``（欧氏度量）
2. 分层聚类（scipy UPGMA）得到树
3. 准对角化（leaves 顺序）→ 协方差呈块结构
4. 递归二分：按子簇方差反比分配权重，自顶向下乘性传播
纯 numpy + scipy，零下载；比单链接/全局最小方差更抗相关估计误差。
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage, to_tree
from scipy.spatial.distance import squareform

from ..core.types import ReturnPanel
from .base import PortfolioBase


def _hrp_weights(cov: np.ndarray, order: list[int]) -> np.ndarray:
    n = cov.shape[0]
    corr = np.zeros((n, n))
    sd = np.sqrt(np.diag(cov)) + 1e-12
    corr = cov / np.outer(sd, sd)
    corr = (corr + corr.T) / 2.0
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    root = to_tree(Z)

    out = np.zeros(n)

    def rec(node, w: float) -> None:
        if node.is_leaf():
            out[node.id] = w
            return
        li = list(range(node.left.pre, node.left.pre + node.left.size))
        ri = list(range(node.right.pre, node.right.pre + node.right.size))
        L = [order[i] for i in li]
        R = [order[i] for i in ri]
        wL = np.full(len(L), 1.0 / len(L))
        wR = np.full(len(R), 1.0 / len(R))
        varL = float(wL @ cov[np.ix_(L, L)] @ wL) + 1e-12
        varR = float(wR @ cov[np.ix_(R, R)] @ wR) + 1e-12
        aL = varR / (varL + varR)
        rec(node.left, w * aL)
        rec(node.right, w * (1.0 - aL))

    rec(root, 1.0)
    w_full = np.zeros(n)
    for i, orig in enumerate(order):
        w_full[orig] = out[i]
    return w_full / w_full.sum()


class HRPAllocator(PortfolioBase):
    name = "hrp"

    def _solve(self, panel: ReturnPanel) -> np.ndarray:
        assert self._Sigma is not None
        try:
            sd = np.sqrt(np.diag(self._Sigma)) + 1e-12
            corr = self._Sigma / np.outer(sd, sd)
            order = _quasi_diag_order_safe(corr)
            w = _hrp_weights(self._Sigma, order)
            if not np.all(np.isfinite(w)) or abs(w.sum() - 1.0) > 1e-4:
                raise ValueError("HRP 数值异常")
            return w
        except Exception:
            inv = 1.0 / np.sqrt(np.diag(self._Sigma) + 1e-12)
            return inv / inv.sum()


def _quasi_diag_order_safe(corr: np.ndarray) -> list[int]:
    n = corr.shape[0]
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    if n == 2:
        return [0, 1]
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    return [int(x) for x in leaves_list(Z)]
