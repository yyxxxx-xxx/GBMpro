#!/usr/bin/env python3
"""
设备流登录助手 (绕过 gh 2.4.0 的轮询卡死问题)

用法:
    python3 scripts/gh_device_login.py
    -> 显示一次性代码
    -> 在浏览器输入授权
    -> 脚本轮询直到拿到 token
    -> 自动写入 ~/.config/gh/hosts.yml 并配置 git credential helper
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CLIENT_ID = "178c6fc778ccc68e1d6a"   # gh CLI 的公开 client_id
SCOPES = "repo read:org workflow gist"
DEVICE_CODE_URL = "https://github.com/login/device/code"
ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"


def _post(url: str, data: dict) -> dict:
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Accept": "application/json",
                 "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def main():
    print("正在申请设备码...", flush=True)
    resp = _post(DEVICE_CODE_URL, {
        "client_id": CLIENT_ID,
        "scope": SCOPES,
    })
    device_code = resp["device_code"]
    user_code = resp["user_code"]
    verify_uri = resp["verification_uri"]
    interval = resp.get("interval", 5)
    expires_in = resp.get("expires_in", 900)

    print()
    print("=" * 62)
    print(f"  一次性代码:  {user_code}")
    print(f"  验证地址:    {verify_uri}")
    print(f"  有效期:      {expires_in // 60} 分钟")
    print("=" * 62)
    print("\n请在浏览器打开上面的地址, 输入代码并授权。")
    print("本脚本会自动轮询直到拿到 token。\n", flush=True)

    deadline = time.time() + expires_in
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            r = _post(ACCESS_TOKEN_URL, {
                "client_id": CLIENT_ID,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            })
        except urllib.error.URLError as e:
            print(f"[{attempt}] 网络错误 {e}, 5 秒后重试", flush=True)
            time.sleep(5)
            continue

        if "access_token" in r:
            token = r["access_token"]
            scope = r.get("scope", "")
            print(f"\n[OK] 拿到 token (scope: {scope})", flush=True)
            _write_hosts(token, scope)
            _setup_git()
            return 0

        err = r.get("error", "unknown")
        if err == "authorization_pending":
            if attempt % 6 == 0:
                print(f"[{attempt}] 仍在等待授权...", flush=True)
            time.sleep(interval)
        elif err == "slow_down":
            interval += 5
            time.sleep(interval)
        elif err == "expired_token":
            print("[FAIL] 代码已过期, 请重跑脚本", flush=True)
            return 1
        elif err == "access_denied":
            print("[FAIL] 授权被拒绝", flush=True)
            return 1
        else:
            print(f"[FAIL] 错误: {err} - {r.get('error_description','')}", flush=True)
            time.sleep(interval)

    print("[FAIL] 超时, 未拿到 token", flush=True)
    return 1


def _write_hosts(token: str, scope: str):
    gh_dir = Path.home() / ".config" / "gh"
    gh_dir.mkdir(parents=True, exist_ok=True)
    hosts = gh_dir / "hosts.yml"
    content = (
        "github.com:\n"
        f"    oauth_token: {token}\n"
        "    user: yyxxxx-xxx\n"
        "    git_protocol: https\n"
        f"    scopes: {scope}\n"
    )
    hosts.write_text(content)
    hosts.chmod(0o600)
    print(f"[OK] 已写入 {hosts}", flush=True)


def _setup_git():
    subprocess.run(["git", "config", "--global",
                    "credential.helper", "store"], check=False)
    cred = Path.home() / ".git-credentials"
    line = "https://yyxxxx-xxx@github.com\n"
    existing = cred.read_text() if cred.exists() else ""
    if "github.com" not in existing:
        cred.write_text(existing + line)
    cred.chmod(0o600)
    print("[OK] 已配置 git credential helper", flush=True)


if __name__ == "__main__":
    sys.exit(main())
