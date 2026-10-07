# PortForge 架构文档

> 作者：晨星 (CJX0712)

## 1. 设计原则

1. **复用世界级开源，不重复造 SOTA**：数学内核（Ledoit-Wolf / Markowitz / Spinu / HRP / BL）
   均直接复用经实证检验的顶级方法，工程层做到确定性、离线、可复现、CI 全绿。
2. **单向无环调用**：`cli → pipeline → {data, portfolio, eval} → core`，无循环依赖。
3. **全局确定性契约**：唯一入口 `core.seed.set_all(seed)` 一次设齐 numpy/random，保证
   同 seed 两次运行核心指标逐位一致。
4. **防数据泄漏**：walk-forward 严格时序切分；PortFuse 的凸权重 λ 仅在验证窗学，测试窗盲测。

## 2. 模块职责

### core/
- `types.py`：可序列化数据类 `ReturnPanel` / `AllocResult` / `Metrics` / `BenchmarkRow` / `BenchmarkReport`。
- `errors.py`：错误码 E100~E500（`AllocError` 等）。
- `config.py`：`PortfolioConfig`（ENV_XXX_* 覆盖 + schema 校验）；`_ENV_MAP` 用 `ClassVar` 标注。
- `interfaces.py`：`Allocator` / `MetaAllocator` Protocol + `assert_long_only` 契约。
- `seed.py`：`set_all` / `derive_seed`（SHA256 稳定子种子）/ `reset_default`。

### data/
- `synthetic.py`：因子模型 + 双 regime（正常 / 危机 vol×2.6、因子溢价翻转）合成收益；
  `_nearest_pd` 投影到最近 PSD（防协方差非正定告警）。
- `loaders.py`：CSV / NPY 载入（utf-8）。

### portfolio/
- `base.py`：`PortfolioBase.fit`（Ledoit-Wolf 收缩协方差）→ `_solve` → `assert_long_only`。
- `equal_weight.py` / `mean_variance.py` / `risk_parity.py` / `hrp.py` / `black_litterman.py`：单分配器。
- `portfuse.py`：**旗舰**——在验证窗学凸权重 λ（SLSQP + L2 正则，边界 `[0, max_concentration=0.5]`），
  `allocate()` 凸组合候选权重。
- `__init__.py`：`BASE_ALLOCATORS` 列表 + `available_cvxpy()`。

### eval/
- `metrics.py`：`compute_metrics`（Sharpe / Sortino / vol / maxDD / CVaR95 / turnover），空数组守卫。
- `backtest.py`：`walk_forward`（train/val/test 严格时序切分，无 look-ahead）。

### pipeline/
- `portfolio_pipeline.py`：`PortfolioPipeline.run()` 跨 seed 聚合 + 基准判定（G1 非劣 + G2 尾部风险更优）。

## 3. 顶级数学内核

| 方法 | 数学核心 | 实现 |
|------|----------|------|
| Ledoit-Wolf | 收缩估计 `Σ̂ = (1−ρ)S + ρF` | sklearn `LedoitWolf` |
| Markowitz | 均值-方差 QP `min wᵀΣw s.t. μᵀw=μ*, w≥0` | scipy SLSQP + 逆波动兜底 |
| 风险平价 ERC | `Σx = 1/x`（x = w/(wᵀΣw)） | Spinu 2013 阻尼牛顿不动点 |
| HRP | 相关→距离→UPGMA 聚类→准对角化→递归二分 | networkx + numpy |
| Black-Litterman | 贝叶斯后验 `Π = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹[...]` | numpy |
| PortFuse | 凸集成 `w = Σ λ_k w_k`，λ 在 val 上学 | scipy SLSQP |

## 4. 确定性保证

- `set_all(seed)` 是唯一随机源入口；`derive_seed(base, tag)` 为各模块派生子种子避免互相污染。
- 所有合成数据、矩估计、优化均在该随机流下运行。
- demo 二次运行校验：核心 Sharpe 最大差 = 0.00e+00（逐位一致）。

## 5. 性能预算与 CI

- demo 端到端 6.4s ≤ 60s（CPU，20 资产 × 2400 期 × 3 seed）。
- CI：ubuntu/windows × py3.12/3.13，步骤 `ruff check` + `ruff format --check` + `pytest` + demo 冒烟。
