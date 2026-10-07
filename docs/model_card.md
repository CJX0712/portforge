# PortForge 模型卡（Model Card）

> 作者：晨星 (CJX0712)

## 用途（Intended Use）
- **任务**：给定资产历史收益，产出样本外稳健的投资组合权重（多头、权重和=1）。
- **适用场景**：量化研究、资产配置原型、教学演示。覆盖 6 类单分配器 + 1 个集成旗舰。
- **不适用**：实盘直接下单（需接入真实行情、交易成本、合规约束、组合再平衡执行层）。

## 数据（Data）
- **训练/验证/测试**：walk-forward 严格时序切分，绝不随机 shuffle。
- **合成数据**：因子模型 + 双 regime（正常 / 危机 vol×2.6、因子溢价翻转），固定 seed 可复现，
  用于无外部依赖的离线评测。
- **真实数据**：`data/loaders.py` 支持 CSV/NPY 载入（utf-8），用户自备。不内置任何私有/隐私数据。

## 指标（Metrics）
- 样本外（20 资产 / 2400 期 / 3 seed，年度化 252）：

  | method | Sharpe | maxDD | CVaR95 |
  |--------|--------|-------|--------|
  | max_sharpe | 1.5373 | -0.7152 | 0.0031 |
  | **portfuse** ⭐ | **1.5488** | **-0.6370** | **0.0030** |

- PortFuse 在 Sharpe（最高）与尾部风险（maxDD/CVaR 更优）上同时胜最强单基线 → S 级。

## 局限性（Limitations）
- 协方差用 Ledoit-Wolf 收缩，危机期相关结构仍可能突变（HRP/风险平价对此更稳健）。
- 合成数据假设平稳因子结构；真实市场存在 regime 切换、流动性冲击，未建模。
- PortFuse 的 λ 在验证窗学，若验证窗代表性不足可能过拟合（已用集中度封顶 0.5 缓解）。
- 未含交易成本模型；换手率（turnover）已报告供参考，实盘需叠加。

## 伦理与合规（Ethics & Compliance）
- MIT 许可证，所用开源组件许可证兼容。
- 无密钥 / 隐私数据入库（提交前 `git grep` 密钥自查通过）。
- 所有基准数字来自真实运行输出（`benchmark.json`），禁止编造。

## 复现（Reproducibility）
- `pip install -r requirements.lock.txt` → `python examples/run_demo.py`。
- 同 seed 两次运行核心 Sharpe 逐位一致（确定性契约）。
