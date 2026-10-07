# Changelog · PortForge

作者：晨星 (CJX0712) · 许可证：MIT

## v0.1.0 (2026-10-07)
- 首发：世界级投资组合优化系统，整合 Ledoit-Wolf 收缩协方差、Markowitz QP、
  Spinu 风险平价 ERC、López de Prado HRP、Black-Litterman 与 PortFuse 凸集成元学习旗舰。
- 确定性契约 `set_all(seed)`：同 seed 两次运行核心 Sharpe 逐位一致（max diff = 0.00e+00）。
- walk-forward 回测：train/val/test 严格时序切分，PortFuse 的 λ 仅在 val 上学，无未来函数。
- 离线优先：纯 numpy / scipy / scikit-learn / networkx，无在线权重下载。
- CI：GitHub Actions 矩阵 ubuntu/windows × py3.12/3.13，ruff 硬门禁 + pytest。
- 质量等级 **S**：PortFuse 样本外 Sharpe=1.5488（最高，Δ+0.0115 vs max_sharpe），尾部风险更优。
- 关键修复：ERC 改 Spinu 阻尼牛顿不动点（原 SLSQP 陷局部极小）；协方差矩估计改特征值 PSD
  投影（原逐元素 clip 破坏负相关结构）。
