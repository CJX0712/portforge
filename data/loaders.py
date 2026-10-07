# PortForge · 世界级投资组合优化系统
# 作者：晨星 (CJX0712)
"""文件载入：从 CSV / NPY 读取收益面板（真实数据可用）。

合成数据零下载；真实市场数据可由用户放置 CSV（首列日期，其余资产收益）
后走本载入器，pipeline 口径完全一致。
"""

from __future__ import annotations

import csv as _csv

import numpy as np

from ..core.types import ReturnPanel


def load_csv(path: str, has_header: bool = True) -> ReturnPanel:
    """载入 CSV 收益面板，显式 utf-8 编码（Windows 兼容）。"""
    rows: list[list[float]] = []
    assets: list[str] = []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = _csv.reader(fh)
        it = iter(reader)
        if has_header:
            header = next(it)
            assets = [h.strip() for h in header[1:]]
        for r in it:
            if not r:
                continue
            rows.append([float(x) for x in r[1:]])
    arr = np.asarray(rows, dtype=np.float64)
    return ReturnPanel(returns=arr, assets=assets or [f"A{i:02d}" for i in range(arr.shape[1])])


def load_npy(path: str) -> ReturnPanel:
    """载入 .npy 收益矩阵 (T, N)。"""
    arr = np.load(path, allow_pickle=False).astype(np.float64)
    return ReturnPanel(returns=arr)
