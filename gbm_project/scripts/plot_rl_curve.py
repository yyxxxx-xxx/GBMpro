#!/usr/bin/env python3
"""
RL 训练曲线分析 / 绘图
======================

用法:
    python3 scripts/plot_rl_curve.py ../experiments/rl_m2_grpo
    python3 scripts/plot_rl_curve.py ../experiments/rl_m2_grpo --plot

产出:
    rl_curve.png       4 联图 (reward / KL / loss / 有效率)
    rl_summary.md      收敛判定报告

⚠️ 为什么必须看滑动平均 (MA):
   实测 3 步就出现 reward 0.19 -> 0.21 -> 0.11 的波动 —— 这**不是**
   收敛失败, 而是采样噪声 (每步只采 16 个分子, 有效 3~8 个)。
   单步值的波动幅度 ≈ 组内 reward 分布宽度, 与学习进展无关。
   判断收敛必须看 MA 曲线 + 线性回归斜率。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_log(run_dir: Path) -> list[dict]:
    path = run_dir / "train_log.jsonl"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    # 去重 (resume 会重复写同一步)
    seen = {}
    for r in rows:
        seen[r["step"]] = r
    return [seen[k] for k in sorted(seen)]


def moving_average(xs, window):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        chunk = xs[lo:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def linreg_slope(xs, ys):
    """最小二乘斜率 (y 对 x)。"""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


def analyze(rows, ma_window=5):
    steps = [r["step"] for r in rows]
    rew = [r.get("reward_mean", 0.0) for r in rows]
    rew_max = [r.get("reward_max", 0.0) for r in rows]
    ma = moving_average(rew, ma_window)
    ma_max = moving_average(rew_max, ma_window)

    kl = [r.get("kl", 0.0) for r in rows]
    loss = [r.get("loss", 0.0) for r in rows]
    valid = [r.get("n_valid_smiles", 0) for r in rows]
    total = [r.get("n_total_samples", 1) for r in rows]
    rate = [v / t if t else 0.0 for v, t in zip(valid, total)]

    slope = linreg_slope(steps, ma)
    contaminated = any(r.get("prior_contaminated") for r in rows)
    kl_end = kl[-1] if kl else 0.0
    ma_first = sum(ma[:ma_window]) / min(ma_window, len(ma))
    ma_last = sum(ma[-ma_window:]) / min(ma_window, len(ma))

    # 收敛判定
    if contaminated:
        verdict = "❌ 先验被污染 —— 结果不可信, 必须重跑"
    elif len(rows) < 5:
        verdict = "⚠️ 步数太少, 无法判定 (至少 15 步)"
    elif slope > 0.005:
        verdict = f"✅ 收敛中 (MA 斜率 {slope:+.5f}/step)"
    elif slope > 0.0:
        verdict = f"🟡 微升 (MA 斜率 {slope:+.5f}/step), 需更多步确认"
    elif slope > -0.005:
        verdict = f"🟡 平台期 (MA 斜率 {slope:+.5f}/step)"
    else:
        verdict = f"❌ 退化 (MA 斜率 {slope:+.5f}/step)"

    return {
        "steps": steps, "rew": rew, "rew_max": rew_max, "ma": ma,
        "ma_max": ma_max, "kl": kl, "loss": loss, "valid": valid,
        "total": total, "rate": rate,
        "slope": slope, "verdict": verdict, "contaminated": contaminated,
        "kl_end": kl_end, "ma_first": ma_first, "ma_last": ma_last,
        "ma_gain": ma_last - ma_first,
    }


def write_summary(a, rows, out_dir: Path, run_dir: Path):
    lines = [
        f"# RL 训练曲线分析 — {run_dir.name}",
        "",
        f"**判定**: {a['verdict']}",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 总步数 | {len(rows)} |",
        f"| MA 斜率 | {a['slope']:+.6f} /step |",
        f"| MA 前 {min(5,len(a['ma']))} 步均值 | {a['ma_first']:.4f} |",
        f"| MA 后 {min(5,len(a['ma']))} 步均值 | {a['ma_last']:.4f} |",
        f"| MA 净提升 | {a['ma_gain']:+.4f} |",
        f"| 末步 KL | {a['kl_end']:.6f} |",
        f"| 先验污染 | {'是 🔴' if a['contaminated'] else '否 ✅'} |",
        "",
        "## 逐步明细",
        "",
        "| step | loss | KL | reward | reward MA | max | 有效率 | 训练样本 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows):
        lines.append(
            f"| {r['step']} | {r.get('loss', 0):+.4f} | {r.get('kl', 0):.5f} "
            f"| {r.get('reward_mean', 0):.4f} | {a['ma'][i]:.4f} "
            f"| {r.get('reward_max', 0):.4f} "
            f"| {r.get('n_valid_smiles', 0)}/{r.get('n_total_samples', 0)} "
            f"| {r.get('n_train_seqs', 0)} |"
        )
    lines += [
        "",
        "## 解读注意",
        "",
        "- **单步 reward 波动不是收敛信号**: 每步只采 16-32 个分子 (有效 3-15 个),",
        "  组内 reward 分布宽度本身就决定了单步波动幅度。",
        "- 看 **reward MA 曲线** 和 **MA 斜率**, 不要看单点。",
        "- KL 持续上升 = 策略在偏离先验; KL 恒为 0 = 梯度没生效。",
        "",
    ]
    (out_dir / "rl_summary.md").write_text("\n".join(lines), encoding="utf-8")


def make_plot(a, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    steps = a["steps"]

    ax = axes[0][0]
    ax.plot(steps, a["rew"], "o-", alpha=0.35, ms=4, label="reward (单步)")
    ax.plot(steps, a["ma"], "s-", lw=2, label="reward MA")
    ax.plot(steps, a["ma_max"], "^--", alpha=0.6, label="max MA")
    ax.set_xlabel("step"); ax.set_ylabel("reward")
    ax.set_title(f"Reward (MA slope {a['slope']:+.5f}/step)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.plot(steps, a["kl"], "o-", lw=1.5)
    ax.set_xlabel("step"); ax.set_ylabel("KL")
    ax.set_title("KL(agent || prior) — 策略偏移量")
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    ax.plot(steps, a["loss"], "o-", lw=1.5, color="tab:purple")
    ax.set_xlabel("step"); ax.set_ylabel("loss")
    ax.set_title("Policy Loss")
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    ax.plot(steps, [r * 100 for r in a["rate"]], "o-", lw=1.5,
            color="tab:green")
    ax.set_xlabel("step"); ax.set_ylabel("valid SMILES (%)")
    ax.set_title("Rollout 有效率")
    ax.grid(alpha=0.3)

    fig.suptitle(f"RL Training — {a['verdict']}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "rl_curve.png", dpi=130)
    print(f"图已保存: {out_dir / 'rl_curve.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="训练输出目录 (含 train_log.jsonl)")
    ap.add_argument("--ma_window", type=int, default=5)
    ap.add_argument("--plot", action="store_true", help="生成 PNG")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    rows = load_log(run_dir)
    if not rows:
        raise SystemExit(f"没有日志: {run_dir / 'train_log.jsonl'}")

    a = analyze(rows, args.ma_window)
    out_dir = run_dir
    write_summary(a, rows, out_dir, run_dir)
    print(f"\n{'='*64}")
    print(f"步数        : {len(rows)}")
    print(f"MA 斜率     : {a['slope']:+.6f} /step")
    print(f"MA 提升     : {a['ma_gain']:+.4f}  ({a['ma_first']:.4f} -> {a['ma_last']:.4f})")
    print(f"末步 KL     : {a['kl_end']:.6f}")
    print(f"先验污染    : {'是 🔴' if a['contaminated'] else '否 ✅'}")
    print(f"判定        : {a['verdict']}")
    print(f"{'='*64}\n")
    print(f"报告: {out_dir / 'rl_summary.md'}")

    if args.plot:
        make_plot(a, out_dir)


if __name__ == "__main__":
    main()
