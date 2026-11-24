#!/usr/bin/env python3
"""
Advantage 模式的离线效果对比
==============================

在**真实评估数据**上模拟不同 advantage 模式的梯度信号质量,
不需要 GPU, 也不需要重新训练。

目的: 在花 1.5 小时 GPU 训练之前, 先用已有数据判断 pairwise/rank
是否值得作为 M5-E 消融组。

对比维度:
  1. **信号强度** —— advantage 的绝对值均值 (越大梯度越强)
  2. **区分度**   —— 组内 advantage 的极差 (能分辨出最好和最差)
  3. **尺度一致性** —— 不同组之间 advantage 尺度是否可比
     (如果某组 reward 差异大、另一组差异小, zscore 会把两组拉到同一尺度,
      丢失"这组更值得学"的信息)
  4. **离群值敏感度** —— 加入极端 reward 时信号是否崩

用法:
    python3 scripts/compare_advantage_modes.py
    python3 scripts/compare_advantage_modes.py --eval_dir ../experiments/eval_m3_kl008
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

GBM_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GBM_ROOT / "src"))

from rl.advantage import compute_group_advantages

DEFAULT_EVAL = Path("/root/Llamole-main/experiments/eval_m3_kl008")
MODES = ("zscore", "pairwise", "pairwise_scaled", "rank")


def load_groups(eval_dir: Path):
    """从评估产物还原 (prompt_id -> rewards) 的组。"""
    cache = {}
    cache_path = eval_dir / "rl_reward_cache.jsonl"
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        cache[d["smiles"]] = d.get("reward", 0.0)

    mols = json.loads((eval_dir / "rl_smiles.json").read_text(encoding="utf-8"))
    groups = {}
    for m in mols:
        smi = m["smiles"]
        if smi in cache:
            groups.setdefault(m.get("prompt_id", -1), []).append(cache[smi])
    return groups


def analyze(groups, mode, min_std=1e-4):
    res = compute_group_advantages(groups, mode=mode, min_std=min_std)
    stats = []
    for pid, g in res.items():
        if g.skipped:
            continue
        advs = np.array(g.advantages)
        rews = np.array(g.rewards)
        stats.append({
            "pid": pid,
            "n": g.n_valid,
            "abs_mean": float(np.mean(np.abs(advs))),
            "span": float(advs.max() - advs.min()),
            "reward_span": float(rews.max() - rews.min()),
            "reward_std": float(rews.std()),
        })
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", default=str(DEFAULT_EVAL))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    groups = load_groups(eval_dir)
    print(f"评估目录: {eval_dir}")
    print(f"组数: {len(groups)}  各组样本数: "
          f"{[len(v) for v in groups.values()][:10]}")

    all_stats = {}
    for mode in MODES:
        st = analyze(groups, mode)
        all_stats[mode] = st
        if not st:
            print(f"{mode}: 所有组都被跳过")
            continue
        abs_mean = np.mean([s["abs_mean"] for s in st])
        span = np.mean([s["span"] for s in st])
        print(f"\n[{mode}]")
        print(f"  |advantage| 均值      : {abs_mean:.4f}")
        print(f"  advantage 极差均值    : {span:.4f}")
        print(f"  有效组数              : {len(st)}")

    # --- 关键对比: 组间尺度一致性 ---
    print(f"\n{'='*72}")
    print("组间尺度一致性 (跨组 advantage 尺度是否可比)")
    print(f"{'='*72}")
    print(f"{'模式':<12}{'组内尺度 CV':>14}{'解读'}")
    for mode in MODES:
        st = all_stats.get(mode) or []
        if not st:
            continue
        spans = np.array([s["span"] for s in st])
        cv = float(spans.std() / (spans.mean() + 1e-12))
        # reward 极差差异越大, 而 advantage 尺度差异越小 -> 说明该模式
        # 把不同"信息量"的组强行拉到同一尺度
        rspan = np.array([s["reward_span"] for s in st])
        r_cv = float(rspan.std() / (rspan.mean() + 1e-12))
        note = (f"reward 极差 CV={r_cv:.3f} → "
                f"{'丢失了组间信息量差异 ✅' if r_cv > cv * 1.5 else '保留了组间差异'}")
        print(f"{mode:<12}{cv:>14.3f}  {note}")

    # --- 离群值敏感度 ---
    print(f"\n{'='*72}")
    print("离群值敏感度 (给某组注入一个极端高分, 看其他组是否被影响)")
    print(f"{'='*72}")
    poisoned = {k: list(v) for k, v in groups.items()}
    target = max(poisoned, key=lambda k: len(poisoned[k]))
    poisoned[target] = poisoned[target] + [999.0]

    for mode in MODES:
        st_clean = analyze(groups, mode)
        st_pois = analyze(poisoned, mode)
        if not st_clean or not st_pois:
            continue
        clean_abs = np.mean([s["abs_mean"] for s in st_clean])
        pois_abs = np.mean([s["abs_mean"] for s in st_pois])
        ratio = pois_abs / (clean_abs + 1e-12)
        verdict = "❌ 信号被污染组带偏" if ratio > 1.5 else (
            "✅ 影响可控" if ratio < 1.2 else "⚠️ 有一定影响")
        print(f"{mode:<12} 组均 |adv| {clean_abs:.4f} → {pois_abs:.4f} "
              f"(×{ratio:.2f})  {verdict}")

    # --- 逐组明细 ---
    print(f"\n{'='*72}")
    print("逐组明细 (前 8 组)")
    print(f"{'='*72}")
    print(f"{'pid':>5}{'n':>5}{'reward跨度':>12}"
          f"{'zscore跨度':>12}{'pairwise跨度':>14}{'rank跨度':>11}")
    for i, pid in enumerate(sorted(groups.keys())[:8]):
        row = [f"{pid:>5}", f"{len(groups[pid]):>5}"]
        rw = np.array(groups[pid])
        row.append(f"{rw.max() - rw.min():>12.4f}")
        for mode in MODES:
            st = all_stats.get(mode) or []
            match = [s for s in st if s["pid"] == pid]
            row.append(f"{match[0]['span']:>12.4f}" if match else f"{'—':>12}")
        print("".join(row))

    if args.out:
        Path(args.out).write_text(json.dumps(
            {m: all_stats[m] for m in MODES}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"\n已保存: {args.out}")


if __name__ == "__main__":
    main()
