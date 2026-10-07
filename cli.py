# PortForge · 世界级投资组合优化系统
# 作者：晨星 (CJX0712)
"""CLI 入口：portforge run / portforge determinism。

用法：
  python -m portforge.cli run [--out benchmark.json]
  python -m portforge.cli determinism
"""

from __future__ import annotations

import argparse
import json
import sys

from .core.config import PortfolioConfig
from .pipeline.portfolio_pipeline import PortfolioPipeline


def _print_table(report) -> None:
    print("\n=== PortForge Benchmark（样本外，跨 seed mean±std）===")
    print(
        f"{'method':<14}{'Sharpe':>12}{'±std':>10}{'annRet':>12}{'annVol':>12}{'maxDD':>12}{'CVaR95':>12}"
    )
    for r in report.rows:
        flag = " *" if r.get("is_flagship") else ""
        print(
            f"{r['name'] + flag:<14}{r['sharpe_mean']:>12.4f}{r['sharpe_std']:>10.4f}"
            f"{r['ann_return_mean']:>12.4f}{r['ann_vol_mean']:>12.4f}"
            f"{r['max_drawdown_mean']:>12.4f}{r['cvar95_mean']:>12.4f}"
        )
    print(f"\n门槛：{report.threshold}")
    print(f"结论：{report.verdict}")


def cmd_run(args: argparse.Namespace) -> int:
    cfg = PortfolioConfig.from_env()
    rep = PortfolioPipeline(cfg).run()
    _print_table(rep)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rep.to_json(), fh, ensure_ascii=False, indent=2)
        print(f"\n[ok] benchmark 已落盘：{args.out}")
    return 0


def cmd_determinism(args: argparse.Namespace) -> int:
    cfg = PortfolioConfig.from_env()
    rep1 = PortfolioPipeline(cfg).run()
    rep2 = PortfolioPipeline(cfg).run()
    core1 = {r["name"]: r["sharpe_mean"] for r in rep1.rows}
    core2 = {r["name"]: r["sharpe_mean"] for r in rep2.rows}
    max_delta = max(abs(core1[n] - core2[n]) for n in core1)
    bit_identical = max_delta == 0.0
    print("\n=== 确定性二次校验 ===")
    print(f"同 seed 两次运行核心 Sharpe 最大差 = {max_delta:.2e}")
    print(f"逐位一致（bit-identical，排除计时）：{bit_identical}")
    return 0 if bit_identical else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="portforge", description="PortForge 投资组合优化系统")
    sub = p.add_subparsers(dest="cmd")
    pr = sub.add_parser("run", help="运行 benchmark")
    pr.add_argument("--out", default="benchmark.json", help="输出 JSON 路径")
    sub.add_parser("determinism", help="确定性二次校验")
    args = p.parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "determinism":
        return cmd_determinism(args)
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
