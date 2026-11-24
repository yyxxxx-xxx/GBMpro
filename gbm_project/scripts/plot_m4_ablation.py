"""
M4 四路消融对比图 (汇报用)
============================

把 4 个 run 的 reward MA 曲线 + KL 曲线画在一张图上, 方便汇报时
一屏说清"beta_kl 怎么影响训练".

用法:
    python3 scripts/plot_m4_ablation.py ../experiments
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib import font_manager as _fm

# 中文字体 (容器默认 DejaVu Sans 无 CJK 字形)
_have = {f.name for f in _fm.fontManager.ttflist}
for _cand in ("Noto Serif CJK SC", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
              "AR PL UMing CN"):
    if _cand in _have:
        matplotlib.rcParams["font.sans-serif"] = [_cand]
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["axes.unicode_minus"] = False
        break


RUNS = [
    ("M3  β=0.08, 30步", "rl_m3_kl008", "#2ca02c", "-"),
    ("A  β=0.08, 80步", "m4_a_kl008_long", "#1f77b4", "-"),
    ("B  β=0.15, 80步", "m4_b_kl015", "#d62728", "--"),
    ("C  β=0.08 关多样性", "m4_c_nodiv", "#ff7f0e", "-"),
    ("D  G=8, lr=5e-6", "m4_d_g8_lowlr", "#9467bd", "-"),
]


def load(run_dir: Path):
    p = run_dir / "train_log.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_root", help="experiments 目录")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.exp_root)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    for label, d, color, ls in RUNS:
        rows = load(root / d)
        if not rows:
            continue
        ma = np.array([r["reward_ma"] for r in rows])
        ax.plot(np.arange(len(ma)), ma, ls, color=color, lw=1.8, label=label, alpha=0.9)
    ax.axhline(0.2858, color="gray", ls=":", lw=1.2)
    ax.text(2, 0.288, "SFT 起点水平", fontsize=8, color="gray")
    ax.set_xlabel("step")
    ax.set_ylabel("reward MA (5步滑动平均)")
    ax.set_title("训练收敛对比")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for label, d, color, ls in RUNS:
        rows = load(root / d)
        if not rows:
            continue
        kl = np.array([r["kl"] for r in rows])
        ax.plot(np.arange(len(kl)), kl, ls, color=color, lw=1.8, label=label, alpha=0.9)
    ax.axhspan(0.05, 0.10, color="green", alpha=0.10)
    ax.text(1, 0.102, "健康区间 0.05–0.10", fontsize=8, color="green")
    ax.set_xlabel("step")
    ax.set_ylabel("KL(agent‖prior)")
    ax.set_title("策略漂移控制 (β_kl 的作用)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    ax = axes[2]
    labels, slopes, colors = [], [], []
    for label, d, color, ls in RUNS:
        rows = load(root / d)
        if not rows:
            continue
        ma = np.array([r["reward_ma"] for r in rows])
        x = np.arange(len(ma))
        sl = np.polyfit(x, ma, 1)[0]
        labels.append(label.split("  ")[0] + " " + label.split(" ")[1][:8])
        slopes.append(sl)
        colors.append(color)
    bars = ax.bar(range(len(slopes)), slopes, color=colors, alpha=0.85)
    for b, s in zip(bars, slopes):
        ax.text(b.get_x() + b.get_width() / 2,
                s + (0.00008 if s >= 0 else -0.00015),
                f"{s:+.5f}", ha="center", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=12)
    ax.set_ylabel("reward MA 线性斜率 (/step)")
    ax.set_title("学习速率排序")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("M4 四路消融: β_kl / 多样性过滤 / 组大小 的影响",
                 fontsize=13, y=1.02)
    fig.tight_layout()

    out = Path(args.out) if args.out else root / "m4_ablation.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"图已保存: {out}")


if __name__ == "__main__":
    main()
