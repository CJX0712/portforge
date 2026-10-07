# PortForge · 世界级投资组合优化系统
# 作者：晨星 (CJX0712)
"""接口契约（Protocol）。模块间只经抽象通信，保证可独立验证与可插拔。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .types import ReturnPanel


@runtime_checkable
class Allocator(Protocol):
    """分配器接口：给定训练收益面板，产出多头长仓权重（sum=1, >=0）。

    语义统一：``fit`` 只在训练窗估计参数（严禁窥视 val/test），``allocate``
    产出权重。score 由评测层统一计算（样本外 Sharpe），不在分配器内编造。
    """

    name: str

    def fit(self, panel: ReturnPanel) -> Allocator: ...

    def allocate(self) -> np.ndarray: ...


@runtime_checkable
class MetaAllocator(Protocol):
    """元分配器：可消费一组候选权重向量，在验证窗学组合。"""

    name: str

    def fit(self, panel: ReturnPanel, candidates: dict[str, np.ndarray]) -> MetaAllocator: ...

    def allocate(self) -> np.ndarray: ...


def assert_long_only(weights: np.ndarray, tol: float = 1e-6) -> None:
    """硬契约：多头长仓权重必须 sum≈1 且分量 >= -tol。"""
    from . import errors as _err

    w = np.asarray(weights, dtype=np.float64)
    if abs(w.sum() - 1.0) > 1e-4:
        raise _err.AllocError(f"权重和不为 1（实际 {w.sum():.6f}）")
    if np.any(w < -tol):
        raise _err.AllocError(f"出现负仓（最小 {w.min():.6f}）违反多仓契约")
