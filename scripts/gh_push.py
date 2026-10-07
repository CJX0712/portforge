#!/usr/bin/env python3
# PortForge · 世界级投资组合优化系统
# 作者：晨星 (CJX0712)
"""发布脚本：3 级回退推送至 GitHub 并打 tag + Release。

Level 1: gh repo create（若不存在）+ push + tag + release
Level 2: 仓库已存在 → 直接 push + tag + release
Level 3: gh 不可用时回退纯 git CLI（Release 需手动补）

用法：
    python scripts/gh_push.py --repo portforge --tag v0.1.0 --release "<摘要>"
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    print(f"[run] {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        out = (r.stderr or r.stdout or "").strip()
        print(f"[warn] rc={r.returncode}: {out}")
        if check:
            raise RuntimeError(f"command failed: {cmd}")
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="仓库名（owner 固定 CJX0712）")
    ap.add_argument("--tag", default="v0.1.0")
    ap.add_argument("--release", default="", help="Release 说明")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--public", action="store_true", default=True)
    args = ap.parse_args()

    owner = "CJX0712"
    full = f"{owner}/{args.repo}"

    # ---- git 初始化 + 提交（确定性作者署名）----
    run("git init -q")
    run(f"git checkout -q -B {args.branch}")
    run("git add -A")
    run(
        'git -c user.name="晨星" -c user.email="dev@chenxing.local" '
        f'commit -q -m "PortForge {args.tag}: 世界级投资组合优化系统（作者：晨星）"',
        check=False,
    )

    # 确保 remote
    if run("git remote get-url origin", check=False).returncode != 0:
        run(f"git remote add origin https://github.com/{full}.git")

    # Release 说明写入临时文件（避免 shell 转义问题）
    notes = args.release or f"PortForge {args.tag}"
    fd, notes_path = tempfile.mkstemp(suffix=".md", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(notes)
    release_notes = notes_path

    # ---- Level 1 / 2：gh ----
    try:
        if run(f"gh repo view {full}", check=False).returncode != 0:
            vis = "--public" if args.public else "--private"
            run(
                f'gh repo create {full} {vis} -d "PortForge · 世界级投资组合优化系统（作者：晨星）"'
            )
        run(f"git push -u origin {args.branch}")
        run(f"git tag -f {args.tag}")
        run(f"git push -u origin {args.tag}")
        run(
            f"gh release create {args.tag} --title {args.tag} --notes-file {release_notes}",
            check=False,
        )
        print("[ok] Level1/2: gh push + tag + release 完成")
        return 0
    except RuntimeError as e:
        print(f"[level2] gh 流程失败，回退纯 git CLI：{e}")

    # ---- Level 3：纯 git CLI ----
    run(f"git push -u origin {args.branch}")
    run(f"git tag -f {args.tag}")
    run(f"git push -u origin {args.tag}")
    print("[ok] Level3: git CLI push 完成（Release 请手动创建）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
