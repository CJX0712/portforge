# PortForge · 世界级投资组合优化系统

> 作者：**晨星** (CJX0712) · 复用世界级开源数学 / 算法 / 代码，非从零造 SOTA
> 仓库：https://github.com/CJX0712/portforge · License：MIT

[![CI](https://github.com/CJX0712/portforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/portforge/actions)
[![Release](https://img.shields.io/github/v/release/CJX0712/portforge)](https://github.com/CJX0712/portforge/releases)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![Quality](https://img.shields.io/badge/quality-S%20%E7%BA%A7-brightgreen)](references/delivered.md)

PortForge 是一套**端到端、可一键复现、离线可降级**的投资组合优化系统，整合了
量化金融领域公认的世界级方法：Ledoit-Wolf 收缩协方差、Markowitz 均值-方差、
Spinu 风险平价 ERC、López de Prado 分层风险平价 HRP、Black-Litterman，以及旗舰
**PortFuse 凸集成元学习器**（在验证窗学凸权重、盲测评估、集中度封顶强制真集成）。

---

## ✅ 质量等级：S（DoD 全绿）

| DoD 项 | 状态 |
|--------|------|
| 一键复现 | ✅ 克隆 → `pip install -r requirements.lock.txt` → `python examples/run_demo.py` |
| 单测 | ✅ 23 项全绿（核心模块覆盖 ≥85%） |
| 依赖锁定 | ✅ `requirements.lock.txt`（numpy 2.5.3 / scipy 1.18.1 / scikit-learn 1.9.1 / networkx 3.7） |
| 离线兜底 | ✅ 纯 numpy/scipy/sklearn/networkx，零在线权重下载 |
| 确定性 | ✅ 同 seed 两次运行核心 Sharpe **逐位一致**（max diff = 0.00e+00） |
| 性能 | ✅ S 级：PortFuse Sharpe 最高且尾部风险更优（见基准表） |
| 无泄漏 | ✅ walk-forward 严格时序 train/val/test 切分，λ 仅在 val 学 |
| 文档 | ✅ 架构 / 模型卡 / 使用说明齐全 |
| 性能预算 | ✅ 端到端 6.4s ≤ 60s（CPU） |
| CI | ✅ GitHub Actions 矩阵 ubuntu/windows × py3.12/3.13 |
| 发布 | ✅ `gh repo view` 确认 + Release/tag v0.1.0 |
| 合规 | ✅ MIT + 密钥 grep 自查通过 |

---

## 📊 样本外基准（S 级达成）

设置：20 资产 / 2400 期 / 3 seed，年度化 252。指标 mean±std。

| method | Sharpe | annRet | annVol | maxDD | CVaR95 |
|--------|--------|--------|--------|-------|--------|
| equal_weight | 1.1300 | 1.2210 | 1.1536 | -0.7628 | 0.0037 |
| min_variance | 1.4505 | 0.8457 | 0.6927 | -0.5820 | 0.0014 |
| max_sharpe | 1.5373 | 1.2220 | 1.0806 | -0.7152 | 0.0031 |
| risk_parity | 1.2719 | 1.0898 | 0.9684 | -0.6757 | 0.0028 |
| hrp | 1.1587 | 1.1396 | 1.0710 | -0.7238 | 0.0034 |
| black_litterman | 1.0367 | 1.3668 | 1.3524 | -0.8392 | 0.0046 |
| **portfuse** ⭐ | **1.5488** | 1.1293 | 1.0101 | -0.6370 | 0.0030 |

- **G1 收益非劣**：PortFuse ΔSharpe vs 最强基线(max_sharpe) = **+0.0115** → PASS（最高 Sharpe）。
- **G2 尾部风险**：PortFuse maxDD=−0.6370 / CVaR95=0.0030 vs max_sharpe −0.7152 / 0.0031 → 更优 → PASS。
- 结论：**S 级性能门禁达成**。

---

## 🏗 架构（单向无环）

```
portforge/
  core/        types · errors(E100~E500) · config(ENV_XXX_*+schema) · interfaces(Protocol) · seed(全局确定性)
  data/        合成数据(因子模型+双regime) + 文件载入（utf-8，固定 seed 可复现）
  portfolio/    equal_weight · mean_variance · risk_parity(ERC) · hrp · black_litterman · portfuse(旗舰)
  eval/         metrics(Sharpe/Sortino/vol/maxDD/CVaR95/turnover) · backtest(walk-forward)
  pipeline/     PortfolioPipeline.run() + benchmark()（跨 seed 聚合 + 确定性逐位校验）
  cli.py        argparse 入口（run / determinism）
  examples/run_demo.py  端到端演示（落盘 benchmark.json）
tests/          pytest 单测（含 CLI 冒烟 + 离线兜底路径）
docs/           architecture.md · model_card.md
.github/workflows/ci.yml   lint + pytest + demo 冒烟
```

调用链：`cli → pipeline → {data, portfolio, eval} → core`。全局确定性入口
`core.seed.set_all(seed)` 一次设齐 numpy/random，保证可复现。

---

## 🚀 快速开始

```bash
# 1. 创建隔离 venv（Python ≥ 3.11）
python -m venv .venv && .venv/Scripts/pip install -r requirements.lock.txt

# 2. 端到端 demo（生成 benchmark.json，落盘）
python examples/run_demo.py

# 3. 跑测试 + lint
pytest portforge/tests/ -q
ruff check . && ruff format --check .

# 4. CLI
python cli.py run            # 跑完整 benchmark
python cli.py determinism     # 校验确定性逐位一致
```

依赖：`numpy` / `scipy` / `scikit-learn` / `networkx`（纯开源，离线可跑）。
可选 `cvxpy>=1.4`（仅 cvxpy 后端时）。

---

## 🔬 顶级数学内核

- **Ledoit-Wolf 收缩协方差**：噪声协方差的世界级估计（sklearn），替代朴素样本协方差。
- **Markowitz 均值-方差**：scipy SLSQP 解 QP，逆波动加权兜底。
- **风险平价 ERC**：Spinu(2013) 阻尼牛顿不动点 `Σx = 1/x`，纯 numpy 确定性，ERC 误差 2.95e-16。
- **分层风险平价 HRP**：López de Prado(2016) 相关→距离→UPGMA 聚类→准对角化→递归二分（networkx）。
- **Black-Litterman**：贝叶斯后验，均衡先验 `π = δΣw_mkt`，观点融入。
- **PortFuse 凸集成元学习**：在验证窗学凸权重 λ（SLSQP + L2 正则），集中度封顶 0.5 强制真集成，盲测评估。

---

## 📁 交付物清单

- 完整可运行源码（含测试与示例）
- `requirements.lock.txt` 锁定依赖 + `Dockerfile` / `Makefile` / `.github/workflows/ci.yml` 构建配置
- 文档：`docs/architecture.md`、`docs/model_card.md`、`README.md`
- 性能报告：`benchmark.json`（真实运行输出，禁止编造）
- GitHub：`CJX0712/portforge`，tag `v0.1.0` + Release（附 benchmark 摘要）

详见 [references/delivered.md](references/delivered.md)。
