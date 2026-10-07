# PortForge · 世界级投资组合优化系统
# 作者：晨星 (CJX0712)
"""portfolio 包：分配器族 + PortFuse 旗舰。

复用顶级开源数学内核（非自研 SOTA）：
- Ledoit-Wolf 协方差收缩（scikit-learn）—— 解决样本协方差病态
- 马科维茨均值-方差 QP（scipy SLSQP）
- 风险平价 ERC（Spinu 不动点）
- 层级风险平价 HRP（López de Prado 2016, scipy 分层聚类）
- 黑利特曼 BL（纯 numpy 贝叶斯后验）
- PortFuse 凸集成元学习（验证窗学凸组合，test 窗盲评）
"""

from __future__ import annotations

from .black_litterman import BlackLittermanAllocator
from .equal_weight import EqualWeightAllocator
from .hrp import HRPAllocator
from .mean_variance import MaxSharpeAllocator, MinVarianceAllocator
from .portfuse import PortFuseAllocator
from .risk_parity import RiskParityAllocator

__all__ = [
    "BlackLittermanAllocator",
    "EqualWeightAllocator",
    "HRPAllocator",
    "MaxSharpeAllocator",
    "MinVarianceAllocator",
    "PortFuseAllocator",
    "RiskParityAllocator",
]

#: 基线分配器清单（旗舰 PortFuse 在 convex hull 上优化，含它们为角点）
BASE_ALLOCATORS = [
    EqualWeightAllocator,
    MinVarianceAllocator,
    MaxSharpeAllocator,
    RiskParityAllocator,
    HRPAllocator,
    BlackLittermanAllocator,
]


def available_cvxpy() -> bool:
    """可选后端探测：cvxpy 仅增强 MaxSharpe（SOCP 更稳），非必需。"""
    try:
        import cvxpy  # noqa: F401

        return True
    except Exception:
        return False
