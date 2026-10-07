# PortForge · 世界级投资组合优化系统
# 作者：晨星 (CJX0712)
"""全局确定性种子管理。

唯一入口 ``set_all(seed)`` 一次性设齐 numpy / random / 标准库，保证
同 seed 两次运行 benchmark 核心指标逐位一致（除计时字段）。
"""

from __future__ import annotations

import contextlib
import hashlib as _hashlib
import random as _random

import numpy as _np

_DEFAULT_SEED = 42


def set_all(seed: int = _DEFAULT_SEED) -> int:
    """一次性设齐所有随机源，返回实际生效的 seed。

    确定性是 PortForge 的硬契约：任何模块在跑前必须调用本函数（或经
    ``PortfolioConfig`` 注入 seed），否则视为违反可复现门禁。
    """
    seed = int(seed)
    _random.seed(seed)
    _np.random.seed(seed)
    # numpy >= 1.25 推荐 Generator，但全局 legacy 也设齐以兼容旧代码；
    # default_rng 在新版本不会抛异常，这里仅作兼容性兜底
    with contextlib.suppress(Exception):  # pragma: no cover - 兼容性兜底
        _np.random.default_rng(seed)
    return seed


def derive_seed(base: int, tag: str) -> int:
    """由基 seed + 字符串标签派生子 seed，避免多模块互相污染随机流。

    使用稳定哈希，跨进程 / 跨平台一致（不依赖哈希随机化）。
    """
    h = _hashlib.sha256(f"{base}::{tag}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


def reset_default() -> int:
    """恢复默认 seed（测试夹具用）。"""
    return set_all(_DEFAULT_SEED)
