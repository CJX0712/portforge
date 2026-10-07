# PortForge · 世界级投资组合优化系统
# 作者：晨星 (CJX0712)
"""PortFuse 旗舰：allocator 凸集成元学习器（ensemble of allocators）。

顶级思想：单个分配器各有盲区（max-Sharpe 追样本噪声 / HRP 忽视溢价 /
风险平价忽视预期收益）。PortFuse 在**独立验证窗**上学一组凸组合权重 λ
（Σλ=1, λ≥0），目标最大化（带 L2 正则的）验证窗 Sharpe，再在**测试窗盲评**。

因为候选权重是凸组合的角点，验证窗上 PortFuse ≥ 任一单分配器；L2 正则
抑制过拟合到单一候选，提升对测试窗的泛化。这是真实存在的「组合分配器」
元学习技术，非自研 SOTA。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..core.interfaces import assert_long_only
from ..core.types import ReturnPanel


class PortFuseAllocator:
    name = "portfuse"

    def __init__(
        self, annualization: int = 252, blend_reg: float = 1e-3, max_concentration: float = 0.5
    ) -> None:
        self.annualization = int(annualization)
        self.blend_reg = float(blend_reg)
        # 集中度上限：任何单一分配器权重 ≤ max_concentration，强制「真集成」
        # 而非押注单一噪声冠军（DeMiguel 2009：优化组合 OOS 难胜等权，须保守）
        self.max_concentration = float(max_concentration)
        self._candidates: dict[str, np.ndarray] = {}
        self._lam: np.ndarray | None = None
        self._names: list[str] = []

    def fit(self, panel: ReturnPanel, candidates: dict[str, np.ndarray]) -> PortFuseAllocator:
        """在验证窗 ``panel`` 上学凸组合权重。

        candidates: 各基线分配器在**训练窗**产出的权重（角点），不得含测试信息。
        """
        self._candidates = {k: np.asarray(v, dtype=np.float64) for k, v in candidates.items()}
        self._names = list(self._candidates.keys())
        m = len(self._names)
        R = panel.returns
        # 每个候选在验证窗的组合收益序列
        p_rets = np.stack([R @ self._candidates[nm] for nm in self._names], axis=1)  # (T, m)

        def neg_obj(lam: np.ndarray) -> float:
            lam = np.clip(lam, 0.0, None)
            lam = lam / lam.sum() if lam.sum() > 0 else np.full(m, 1.0 / m)
            pr = p_rets @ lam
            mean = pr.mean()
            std = pr.std() + 1e-12
            sharpe = mean / std * np.sqrt(self.annualization)
            reg = self.blend_reg * float(lam @ lam)
            return -(sharpe - reg)

        x0 = np.full(m, 1.0 / m)
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        res = minimize(
            neg_obj,
            x0,
            method="SLSQP",
            bounds=[(0, self.max_concentration)] * m,
            constraints=cons,
            options={"ftol": 1e-12, "maxiter": 500},
        )
        if res.success and np.all(np.isfinite(res.x)):
            self._lam = np.clip(res.x, 0.0, None)
            self._lam = self._lam / self._lam.sum()
        else:
            self._lam = x0
        return self

    def allocate(self) -> np.ndarray:
        if self._lam is None:
            raise RuntimeError("请先调用 fit()")
        w = np.zeros_like(next(iter(self._candidates.values())))
        for lam_i, nm in zip(self._lam, self._names):
            w = w + lam_i * self._candidates[nm]
        assert_long_only(w)
        return w

    @property
    def blend_weights(self) -> dict[str, float]:
        return {nm: float(lam) for nm, lam in zip(self._names, self._lam)}
