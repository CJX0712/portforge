# PortForge · 世界级投资组合优化系统
# 作者：晨星 (CJX0712)
"""类型定义：收益面板、配置、分配结果与评测指标。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class ReturnPanel:
    """多资产收益面板（已对齐、无 NaN）。

    Attributes
    ----------
    returns : (T, N) float 矩阵，行=时间、列=资产，单位=每期简单收益。
    assets : 资产名列表（长度 N）。
    """

    returns: np.ndarray
    assets: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.returns = np.asarray(self.returns, dtype=np.float64)
        if self.returns.ndim != 2:
            raise ValueError("returns 必须是 2D (T, N) 矩阵")
        if self.returns.shape[0] < 2:
            raise ValueError("returns 至少需要 2 期")
        if np.any(~np.isfinite(self.returns)):
            raise ValueError("returns 含非有限值（NaN/Inf）")
        n = self.returns.shape[1]
        if not self.assets:
            self.assets = [f"A{i:02d}" for i in range(n)]
        if len(self.assets) != n:
            raise ValueError("assets 长度必须与列数一致")


@dataclass
class AllocResult:
    """单一分配器输出。

    weights 为多头长仓权重（sum=1，分量 >= 0）；score 用于跨分配器公平排序
    （此处 score = 样本外 Sharpe，越大越好）。
    """

    name: str
    weights: np.ndarray
    score: float = float("nan")


@dataclass
class Metrics:
    """样本外绩效指标。所有字段为均值口径（多 seed / 多窗聚合时取平均）。"""

    ann_return: float = float("nan")
    ann_vol: float = float("nan")
    sharpe: float = float("nan")
    sortino: float = float("nan")
    max_drawdown: float = float("nan")
    cvar95: float = float("nan")
    turnover: float = float("nan")

    def as_dict(self) -> Dict[str, float]:
        return {
            "ann_return": self.ann_return,
            "ann_vol": self.ann_vol,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "cvar95": self.cvar95,
            "turnover": self.turnover,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> Metrics:
        return cls(**{k: float(d[k]) for k in cls.__dataclass_fields__ if k in d})


@dataclass
class BenchmarkRow:
    """benchmark 表的一行（含 mean±std，用于真实运行输出）。"""

    name: str
    sharpe_mean: float
    sharpe_std: float
    ann_return_mean: float
    ann_vol_mean: float
    max_drawdown_mean: float
    cvar95_mean: float
    is_flagship: bool = False
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class BenchmarkReport:
    """端到端 benchmark 报告，可序列化为 benchmark.json。"""

    system: str = "PortForge"
    seed: int = 42
    n_assets: int = 0
    n_periods: int = 0
    n_seeds: int = 0
    rows: List[Dict] = field(default_factory=list)
    verdict: str = ""
    threshold: str = ""
    passed: bool | None = None
    elapsed_sec: float = float("nan")
    determinism_bit_identical: bool | None = None
    notes: List[str] = field(default_factory=list)

    def to_json(self) -> Dict:
        return {
            "system": self.system,
            "seed": self.seed,
            "n_assets": self.n_assets,
            "n_periods": self.n_periods,
            "n_seeds": self.n_seeds,
            "rows": self.rows,
            "verdict": self.verdict,
            "threshold": self.threshold,
            "passed": self.passed,
            "elapsed_sec": self.elapsed_sec,
            "determinism_bit_identical": self.determinism_bit_identical,
            "notes": self.notes,
        }
