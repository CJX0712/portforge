# PortForge · 世界级投资组合优化系统 — 作者：晨星 (CJX0712)
"""core 层单测：确定性 / 配置 / 接口契约。"""

from __future__ import annotations

import numpy as np
import pytest
from portforge.core import errors as E
from portforge.core.config import PortfolioConfig
from portforge.core.interfaces import assert_long_only
from portforge.core.seed import derive_seed, set_all


def test_set_all_deterministic():
    set_all(7)
    a = np.random.randn(5)
    set_all(7)
    b = np.random.randn(5)
    assert np.array_equal(a, b)


def test_derive_seed_stable_and_distinct():
    s1 = derive_seed(42, "x")
    s2 = derive_seed(42, "x")
    s3 = derive_seed(42, "y")
    assert s1 == s2
    assert s1 != s3


def test_config_validates_splits():
    cfg = PortfolioConfig(n_assets=10, n_periods=300, train_frac=0.5, val_frac=0.2)
    cfg.validate()
    assert abs(cfg.test_frac - 0.3) < 1e-9


def test_config_rejects_no_test_window():
    with pytest.raises(E.ConfigError):
        PortfolioConfig(train_frac=0.6, val_frac=0.5).validate()


def test_assert_long_only_accepts():
    w = np.array([0.2, 0.3, 0.5])
    assert_long_only(w)  # 不抛


def test_assert_long_only_rejects_negative():
    with pytest.raises(E.AllocError):
        assert_long_only(np.array([0.5, -0.5, 1.0]))


def test_assert_long_only_rejects_nonsum():
    with pytest.raises(E.AllocError):
        assert_long_only(np.array([0.3, 0.3, 0.3]))
