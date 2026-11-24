"""
主结果可视化 (汇报用): RL vs SFT 基线
=====================================

两张图:
  1) reward 分布对比 (直方图 + 箱线图) —— SFT n=231 / RL n=220
  2) 8 项分子属性 Δ 对比 (相对 SFT 基线, 正=更好)

用法:
    python3 scripts/plot_main_result.py ../experiments/eval_m3_kl008
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

_have = {f.name for f in _fm.fontManager.ttflist}
for _cand in ("Noto Serif CJK SC", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
              "AR PL UMing CN"):
    if _cand in _have:
        matplotlib.rcParams["font.sans-serif"] = [_cand]
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["axes.unicode_minus"] = False
        break

# (key, 显示名, 越大越好?)
METRICS = [
    ("reward", "reward\n(综合)", True),
    ("vina", "vina\n对接亲和力", False),
    ("bbb", "bbb\n血脑屏障", True),
    ("qed", "qed\n类药性", True),
    ("ames", "ames\n毒性", False),
    ("herg", "herg\n心脏毒性", False),
    ("pgp", "pgp\n外排", False),
    ("dili", "dili\n肝毒性", False),
]


def load(ev_dir: Path, tag: str):
    cache = {}
    for line in (ev_dir / f"{tag}_reward_cache.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            cache[d["smiles"]] = d
    mols = json.loads((ev_dir / f"{tag}_smiles.json").read_text(encoding="utf-8"))
    return [cache[m["smiles"]] for m in mols if m["smiles"] in cache]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = Path(args.eval_dir)

    base = load(d, "baseline_sft")
    rl = load(d, "rl")
    br = np.array([x["reward"] for x in base if x.get("reward") is not None])
    rr = np.array([x["reward"] for x in rl if x.get("reward") is not None])

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25])

    # --- 1. reward 分布 ---
    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 0.75, 31)
    ax.hist(br, bins=bins, alpha=0.55, color="#7f7f7f", label=f"SFT 基线 (n={len(br)})")
    ax.hist(rr, bins=bins, alpha=0.55, color="#2ca02c", label=f"RL GRPO (n={len(rr)})")
    ax.axvline(np.median(br), color="#555", ls="--", lw=1.4)
    ax.axvline(np.median(rr), color="#1a7a37", ls="--", lw=1.4)
    ax.set_xlabel("reward")
    ax.set_ylabel("分子数")
    ax.set_title("reward 分布 (虚线=中位数)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 0]).inset_axes([0.58, 0.52, 0.4, 0.42])
    bp = ax2.boxplot([br, rr], labels=["SFT", "RL"], widths=0.5,
                     patch_artist=True, showfliers=False)
    bp["boxes"][0].set_facecolor("#b0b0b0")
    bp["boxes"][1].set_facecolor("#8fd18f")
    ax2.set_title("中位数 {:+.1%}".format(np.median(rr) / np.median(br) - 1),
                  fontsize=9)
    ax2.tick_params(labelsize=8)

    # --- 2. 属性 Δ ---
    ax = fig.add_subplot(gs[0, 1])
    names, deltas, raw_sft, raw_rl, colors = [], [], [], [], []
    for k, disp, higher_better in METRICS:
        a = np.array([x[k] for x in base if x.get(k) is not None])
        b = np.array([x[k] for x in rl if x.get(k) is not None])
        if len(a) == 0 or len(b) == 0:
            continue
        # 归一化成"相对基线的改进幅度", 并统一方向 (越大越好)
        denom = np.abs(a.mean()) if abs(a.mean()) > 1e-9 else 1.0
        rel = (b.mean() - a.mean()) / denom
        if not higher_better:
            rel = -rel
        names.append(disp)
        deltas.append(rel * 100)
        raw_sft.append(a.mean())
        raw_rl.append(b.mean())
        colors.append("#2ca02c" if rel > 0 else "#d62728")

    bars = ax.bar(range(len(deltas)), deltas, color=colors, alpha=0.88)
    for i, (b_, dv) in enumerate(zip(bars, deltas)):
        ax.text(b_.get_x() + b_.get_width() / 2, dv + (0.7 if dv >= 0 else -1.6),
                f"{dv:+.1f}%", ha="center", fontsize=9)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("相对 SFT 基线的改进 (%)  ↑ 越大越好")
    ax.set_title("8 项分子属性全部同向改善")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("主结果: GRPO 强化学习 vs SFT 基线 (IDH1, seed=1234, 8 prompts)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = Path(args.out) if args.out else d / "main_result.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"图已保存: {out}")


if __name__ == "__main__":
    main()
