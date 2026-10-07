# PortForge · 世界级投资组合优化系统
# 作者：晨星 (CJX0712)
"""合成多资产收益生成器（因子模型 + 机制切换）。

设计目标（对应 SOP 第 4 节踩坑 §B 数据泄漏 / DGP 甜点）：
1. **可复现**：仅依赖 ``PortfolioConfig.seed`` 派生的独立随机流，同 seed 逐位一致。
2. **无泄漏**：生成的是完整面板；train/val/test 切分在 pipeline 做，生成器
   不接触任何切分。
3. **难度甜点**：存在因子溢价（让 max-Sharpe 有信息可挖），但机制切换导致
   波动率时变（让 vol 目标化的 HRP / 风险平价占优），单分配器无法同时拿到
   两种优势 → 凸集成（PortFuse）有真实增益空间，而非全满分或全崩。
"""

from __future__ import annotations

import numpy as np

from ..core import seed as _seed
from ..core.config import PortfolioConfig
from ..core.types import ReturnPanel


def _make_loadings(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """资产×因子载荷矩阵 B（N,K）。

    前 k 个资产各绑定一个主因子（制造板块结构），其余资产随机混合，
    使协方差呈现「板块内高相关、板块间低相关」的真实形态。
    """
    B = rng.normal(0.4, 0.3, size=(n, k))
    for j in range(min(k, n)):
        B[j, j] = abs(B[j, j]) + 0.6  # 主因子强暴露
    return B


def _make_idiosyncratic(n: int, rng: np.random.Generator) -> np.ndarray:
    """异质波动 diag(D)，让单资产有不可分散噪声。"""
    return np.diag(rng.uniform(0.04, 0.10, size=n))


def generate_panel(cfg: PortfolioConfig, seed: int | None = None) -> ReturnPanel:
    """生成 (T, N) 合成收益面板。

    机制切换：两机制，正常态 vol 低、危机态 vol 高（×1.6）且因子溢价转负，
    模拟真实的「波动率聚集 + 尾部相关性上升」。每期以 2% 概率切换。
    """
    seed = int(cfg.seed if seed is None else seed)
    rng = np.random.default_rng(_seed.derive_seed(seed, "synthetic-returns"))
    n, T = cfg.n_assets, cfg.n_periods
    k = max(2, n // 6)  # 因子数随资产规模增长

    B = _make_loadings(n, k, rng)
    D = _make_idiosyncratic(n, rng)

    # 因子溢价（每期，年化近似后再折算到单期）
    mu_f = rng.normal(0.0, 1.0, size=k)
    mu_f = mu_f / np.linalg.norm(mu_f) * 0.05  # 因子年化溢价 ~5%
    mu_f[0] += 0.04  # 第一个因子给稳定正溢价（信息源）

    # 两机制因子协方差（年化近似），对称 + 加脊确保 PSD
    Sigma_low = np.diag(rng.uniform(0.08, 0.14, size=k))
    off = rng.normal(0, 0.03, size=(k, k))
    Sigma_low = Sigma_low + off + off.T
    Sigma_low = (Sigma_low + Sigma_low.T) / 2.0 + 1e-4 * np.eye(k)
    Sigma_high = Sigma_low * 2.6  # 危机态波动放大

    def _nearest_pd(M: np.ndarray) -> np.ndarray:
        # 投影到最近对称正定矩阵，消除 mvn 的「非 PSD」数值警告
        M = (M + M.T) / 2.0
        w, V = np.linalg.eigh(M)
        w = np.clip(w, 1e-8, None)
        return (V * w) @ V.T

    Sigma_low = _nearest_pd(Sigma_low)
    Sigma_high = _nearest_pd(Sigma_high)

    per = np.sqrt(1.0 / cfg.annualization)  # 单期折算
    mu_f_p = mu_f * per
    Sigma_low_p = _nearest_pd(Sigma_low * per)
    Sigma_high_p = _nearest_pd(Sigma_high * per)
    D_p = D * per

    returns = np.empty((T, n), dtype=np.float64)
    regime = 0
    for t in range(T):
        if t > 0 and rng.random() < 0.02:
            regime = 1 - regime
        Sigma_f = Sigma_high_p if regime == 1 else Sigma_low_p
        f = rng.multivariate_normal(mu_f_p, Sigma_f)
        asset_mean = B @ (mu_f_p if regime == 0 else (mu_f_p - 0.03 * per * np.ones(k)))
        eps = rng.multivariate_normal(np.zeros(n), D_p)
        returns[t] = asset_mean + B @ f + eps

    assets = [f"A{i:02d}" for i in range(n)]
    return ReturnPanel(returns=returns, assets=assets)
