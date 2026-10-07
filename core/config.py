# PortForge · 世界级投资组合优化系统
# 作者：晨星 (CJX0712)
"""全局配置：ENV_XXX_* 覆盖 + schema 校验。

所有可调参数集中于此，benchmark / demo 启动时一次性实例化
``PortfolioConfig``，并强制 ``set_all(config.seed)`` 以锁定随机流。
"""

from __future__ import annotations

import os as _os
from dataclasses import dataclass
from typing import ClassVar

from . import errors as _err


@dataclass
class PortfolioConfig:
    """系统级配置（确定性 + 数据 + 评测口径）。"""

    seed: int = 42
    n_assets: int = 20
    n_periods: int = 2400  # 总期数（日频近似）
    train_frac: float = 0.50  # 估计窗占比
    val_frac: float = 0.15  # 元学习（PortFuse 凸组合）验证窗
    # test 窗 = 剩余 0.35
    n_seeds: int = 3
    annualization: int = 252  # 交易日年化因子
    risk_free: float = 0.0  # 无风险利率（年化，简化）
    cvar_alpha: float = 0.95  # CVaR 置信度
    vol_target: float = 0.15  # 年化目标波动（组合波动目标化）
    blend_reg: float = 1.0e-3  # PortFuse 凸组合 L2 正则（防过拟合）
    min_weight: float = 0.0  # 多仓下界（0 = 允许空仓）
    max_weight: float = 1.0  # 单资产上界

    # 以下由 ENV_XXX_* 覆盖（CI / 复现时可用）
    _ENV_MAP: ClassVar[dict[str, str]] = {
        "PORTFORGE_SEED": "seed",
        "PORTFORGE_N_ASSETS": "n_assets",
        "PORTFORGE_N_PERIODS": "n_periods",
        "PORTFORGE_N_SEEDS": "n_seeds",
        "PORTFORGE_TRAIN_FRAC": "train_frac",
        "PORTFORGE_VAL_FRAC": "val_frac",
        "PORTFORGE_VOL_TARGET": "vol_target",
        "PORTFORGE_BLEND_REG": "blend_reg",
    }

    @classmethod
    def from_env(cls) -> PortfolioConfig:
        """从环境变量读取覆盖（仅覆盖被显式设置的键）。"""
        cfg = cls()
        for env_key, attr in cls._ENV_MAP.items():
            if env_key in _os.environ:
                raw = _os.environ[env_key]
                cur = getattr(cfg, attr)
                try:
                    if isinstance(cur, bool):
                        setattr(cfg, attr, raw.lower() in ("1", "true", "yes"))
                    elif isinstance(cur, int):
                        setattr(cfg, attr, int(raw))
                    elif isinstance(cur, float):
                        setattr(cfg, attr, float(raw))
                    else:
                        setattr(cfg, attr, raw)
                except ValueError as e:  # pragma: no cover
                    raise _err.ConfigError(f"{env_key}={raw} 非法的类型覆盖") from e
        cfg.validate()
        return cfg

    def validate(self) -> PortfolioConfig:
        if not (0 < self.train_frac < 1):
            raise _err.ConfigError("train_frac 必须在 (0,1)")
        if not (0 < self.val_frac < 1):
            raise _err.ConfigError("val_frac 必须在 (0,1)")
        if self.train_frac + self.val_frac >= 1.0:
            raise _err.ConfigError("train_frac + val_frac 必须小于 1（需留 test 窗）")
        if self.n_assets < 2:
            raise _err.ConfigError("n_assets 至少 2")
        if self.n_periods < 30:
            raise _err.ConfigError("n_periods 至少 30")
        if self.n_seeds < 1:
            raise _err.ConfigError("n_seeds 至少 1")
        if not (0.0 <= self.risk_free <= 1.0):
            raise _err.ConfigError("risk_free 必须在 [0,1]")
        if not (0.0 < self.cvar_alpha < 1.0):
            raise _err.ConfigError("cvar_alpha 必须在 (0,1)")
        return self

    @property
    def test_frac(self) -> float:
        return max(0.0, 1.0 - self.train_frac - self.val_frac)
