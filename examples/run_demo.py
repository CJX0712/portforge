# PortForge · 世界级投资组合优化系统
# 作者：晨星 (CJX0712)
"""端到端演示：生成合成数据 → 多分配器 walk-forward 回测 → 打印表 → 落盘 benchmark.json
+ 确定性二次校验（同 seed 两次运行核心指标逐位一致，排除计时）。

直接运行：python examples/run_demo.py
"""

from __future__ import annotations

import json
import sys
import time

from portforge.core.config import PortfolioConfig
from portforge.pipeline.portfolio_pipeline import PortfolioPipeline


def main() -> int:
    cfg = PortfolioConfig.from_env()
    print(f"[demo] n_assets={cfg.n_assets} n_periods={cfg.n_periods} n_seeds={cfg.n_seeds}")

    t0 = time.perf_counter()
    rep = PortfolioPipeline(cfg).run()
    elapsed = time.perf_counter() - t0

    # 确定性二次校验
    rep2 = PortfolioPipeline(cfg).run()
    core1 = {r["name"]: r["sharpe_mean"] for r in rep.rows}
    core2 = {r["name"]: r["sharpe_mean"] for r in rep2.rows}
    max_delta = max(abs(core1[n] - core2[n]) for n in core1)
    rep.determinism_bit_identical = max_delta == 0.0

    print("\n=== PortForge Benchmark（样本外，跨 seed mean±std）===")
    print(
        f"{'method':<14}{'Sharpe':>12}{'±std':>10}{'annRet':>12}{'annVol':>12}{'maxDD':>12}{'CVaR95':>12}"
    )
    for r in rep.rows:
        flag = " *" if r.get("is_flagship") else ""
        print(
            f"{r['name'] + flag:<14}{r['sharpe_mean']:>12.4f}{r['sharpe_std']:>10.4f}"
            f"{r['ann_return_mean']:>12.4f}{r['ann_vol_mean']:>12.4f}"
            f"{r['max_drawdown_mean']:>12.4f}{r['cvar95_mean']:>12.4f}"
        )
    print(f"\n门槛：{rep.threshold}")
    print(f"结论：{rep.verdict}")
    print(
        f"确定性：同 seed 两次核心 Sharpe 最大差 = {max_delta:.2e} → "
        f"{'逐位一致 ✅' if rep.determinism_bit_identical else '不一致 ❌'}"
    )
    print(f"端到端耗时：{elapsed:.1f}s（预算 ≤60s）")

    out = "benchmark.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rep.to_json(), fh, ensure_ascii=False, indent=2)
    print(f"[ok] benchmark 已落盘：{out}")
    return 0 if rep.determinism_bit_identical else 1


if __name__ == "__main__":
    sys.exit(main())
