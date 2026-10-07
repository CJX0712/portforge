#!/usr/bin/env python3
# PortForge · 世界级投资组合优化系统
# 作者：晨星 (CJX0712)
"""经 GitHub Git Data API 推送首个提交（代理/隧道不可达 git 智能 HTTP 时的可靠通道）。

用 `gh api`（走 WorkBuddy 代理，已验证可达）逐文件建 blob → tree → commit → ref，
绕过 git 协议的 CONNECT 隧道 502。等价于 `git push`，但走 REST API。

用法：
    python scripts/gh_api_push.py --repo portforge --tag v0.1.0 --release "<摘要>"
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys


def gh_api(method: str, path: str, body: dict | None = None) -> dict:
    cmd = ["gh", "api", f"repos/CJX0712/{OWNER_REPO}" + path, "-X", method]
    if body is not None:
        cmd += ["--input", "-"]
    inp = json.dumps(body) if body is not None else None
    r = subprocess.run(cmd, input=inp, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gh api {method} {path} 失败: {r.stderr.strip()}")
    return json.loads(r.stdout or "{}")


def put_ref(ref: str, sha: str) -> None:
    """创建 ref；若已存在（422）则更新为强制指向新 commit。"""
    try:
        gh_api("POST", "/git/refs", {"ref": ref, "sha": sha})
    except RuntimeError as e:
        if "already exists" in str(e) or "422" in str(e):
            # PATCH 端点为 /git/<ref>（如 /git/refs/heads/main）
            gh_api("PATCH", f"/git/{ref}", {"sha": sha, "force": True})
        else:
            raise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tag", default="v0.1.0")
    ap.add_argument("--release", default="")
    ap.add_argument("--branch", default="main")
    args = ap.parse_args()

    global OWNER_REPO
    OWNER_REPO = f"{args.repo}"

    # 1) 列出已跟踪文件（先 add -A 纳入新增/改动文件）
    subprocess.run(["git", "add", "-A"], check=True)
    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.splitlines()
    files = [f for f in files if f]
    print(f"[info] 待推送文件数：{len(files)}")

    # 2) 逐文件建 blob
    blobs = {}
    for f in files:
        with open(f, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        resp = gh_api("POST", "/git/blobs", {"content": b64, "encoding": "base64"})
        blobs[f] = resp["sha"]
        print(f"  blob {f} -> {resp['sha'][:10]}")

    # 3) 建 tree
    tree = [{"path": f, "mode": "100644", "type": "blob", "sha": blobs[f]} for f in files]
    tree_resp = gh_api("POST", "/git/trees", {"tree": tree})
    tree_sha = tree_resp["sha"]
    print(f"[info] tree -> {tree_sha[:10]}")

    # 4) 建 commit（首个提交，无 parent）
    msg = f"PortForge {args.tag}: 世界级投资组合优化系统（作者：晨星）"
    commit = gh_api(
        "POST",
        "/git/commits",
        {"message": msg, "tree": tree_sha, "parents": []},
    )
    commit_sha = commit["sha"]
    print(f"[info] commit -> {commit_sha[:10]}")

    # 5) 建/更新分支 ref
    put_ref(f"refs/heads/{args.branch}", commit_sha)
    print(f"[ok] 分支 {args.branch} -> {commit_sha[:10]}")

    # 6) 建/更新 tag ref
    put_ref(f"refs/tags/{args.tag}", commit_sha)
    print(f"[ok] tag {args.tag} 已创建")

    # 7) 建 Release（gh 走 API，可达）
    import os
    import tempfile

    notes = args.release or f"PortForge {args.tag}"
    fd, np_ = tempfile.mkstemp(suffix=".md", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(notes)
    rel = subprocess.run(
        ["gh", "release", "create", args.tag, "--title", args.tag, "--notes-file", np_],
        capture_output=True,
        text=True,
    )
    if rel.returncode == 0:
        print(f"[ok] Release {args.tag} 已创建")
    else:
        print(f"[warn] Release 创建返回：{rel.stderr.strip()}")

    print(f"[done] https://github.com/CJX0712/{args.repo}")
    return 0


if __name__ == "__main__":
    OWNER_REPO = ""
    sys.exit(main())
