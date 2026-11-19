"""
独立评估的显著性检验: RL vs SFT 基线
=====================================

⚠️ 前提: 两组必须用**同一个 seed + 同一个 prompt 集**采样,
   否则采到的分子不同, 对比的不是"策略差异"而是"采样运气"。

检验方法:
  1. Mann-Whitney U (非参数, 不假设正态) —— 主检验
  2. Welch t-test (不等方差)
  3. 配对检验: 同一 prompt 内的最优分子 (每组 8 个 prompt 配对)
  4. 效应量 (Cliff's delta / Cohen's d)

用法:
    python3 scripts/compare_eval.py ../experiments/rl_m2_grpo/eval
"""

import json
import sys
from pathlib import Path

import numpy as np


def load_rewards(cache_path: Path, smiles_path: Path):
    """从 reward cache + smiles 还原 (reward, prompt_id) 列表。"""
    cache = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        cache[d["smiles"]] = d.get("reward", 0.0)
    mols = json.loads(smiles_path.read_text(encoding="utf-8"))
    rows = []
    for m in mols:
        smi = m["smiles"]
        if smi in cache:
            rows.append((m.get("prompt_id", -1), cache[smi], smi))
    return rows


def mann_whitney_u(a, b):
    """双侧 p 值 (正态近似 + tie 修正)。"""
    from itertools import chain
    a, b = np.asarray(a), np.asarray(b)
    n1, n2 = len(a), len(b)
    allv = np.concatenate([a, b])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u = min(u1, n1 * n2 - u1)
    mu = n1 * n2 / 2.0
    n = n1 + n2
    tie = 0
    _, counts = np.unique(allv, return_counts=True)
    tie = (counts ** 3 - counts).sum()
    sigma = np.sqrt(n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1))))
    if sigma <= 0:
        return u, 1.0
    z = (u - mu) / sigma
    from math import erfc, sqrt
    p = erfc(abs(z) / sqrt(2))
    return u, p


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) /
                 (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else 0.0


def main():
    eval_dir = Path(sys.argv[1] if len(sys.argv) > 1 else
                    "/root/Llamole-main/experiments/rl_m2_grpo/eval")

    base = load_rewards(eval_dir / "baseline_sft_reward_cache.jsonl",
                        eval_dir / "baseline_sft_smiles.json")
    rl = load_rewards(eval_dir / "rl_reward_cache.jsonl",
                      eval_dir / "rl_smiles.json")

    if not base or not rl:
        raise SystemExit("评估数据不全")

    br = np.array([r[1] for r in base])
    rr = np.array([r[1] for r in rl])

    print("=" * 70)
    print(f"RL vs SFT 基线 — 独立评估显著性检验 ({eval_dir})")
    print("=" * 70)
    print(f"\n样本量: SFT n={len(br)}  RL n={len(rr)}  "
          f"(同一 seed=1234, 同一 8 条 prompt)")

    print("\n1. 描述统计")
    print(f"   {'指标':<14}{'SFT':>12}{'RL':>12}{'Δ':>12}{'Δ%':>9}")
    for name, fa, fr in [
        ("mean", np.mean, np.mean),
        ("median", np.median, np.median),
        ("std", np.std, np.std),
    ]:
        a, b = fa(br), fr(rr)
        print(f"   {name:<14}{a:>12.4f}{b:>12.4f}{b-a:>+12.4f}"
              f"{(b-a)/a*100 if a else 0:>+8.1f}%")
    a, b = br.max(), rr.max()
    print(f"   {'max':<14}{a:>12.4f}{b:>12.4f}{b-a:>+12.4f}{(b-a)/a*100:>+8.1f}%")
    a, b = np.percentile(br, 90), np.percentile(rr, 90)
    print(f"   {'p90':<14}{a:>12.4f}{b:>12.4f}{b-a:>+12.4f}{(b-a)/a*100:>+8.1f}%")

    print("\n2. 独立样本检验 (全部样本, 不配对)")
    u, p_mw = mann_whitney_u(rr, br)
    print(f"   Mann-Whitney U  = {u:.1f}   p = {p_mw:.4f}  "
          f"{'✅ 显著 (p<0.05)' if p_mw < 0.05 else '❌ 不显著 (p>=0.05)'}")
    print(f"   Cliff's delta   = {cliffs_delta(rr, br):+.4f}  "
          f"(0=无差异, ±1=完全分离)")
    print(f"   Cohen's d       = {cohens_d(rr, br):+.4f}")

    print("\n3. 配对检验 (同一 prompt 内的最优分子, n=8 对)")
    b_best = {p: max(r for q, r, _ in base if q == p) for p in
              set(q for q, _, _ in base)}
    r_best = {p: max(r for q, r, _ in rl if q == p) for p in
              set(q for q, _, _ in rl)}
    common = sorted(set(b_best) & set(r_best))
    if common:
        diffs = np.array([r_best[p] - b_best[p] for p in common])
        print(f"   {'prompt':<8}{'SFT best':>12}{'RL best':>12}{'Δ':>10}")
        for p in common:
            print(f"   {p:<8}{b_best[p]:>12.4f}{r_best[p]:>12.4f}"
                  f"{r_best[p]-b_best[p]:>+10.4f}")
        u2, p2 = mann_whitney_u(diffs, np.zeros_like(diffs))
        print(f"\n   Δ 均值 {diffs.mean():+.4f}  中位数 {np.median(diffs):+.4f}  "
              f"胜 {sum(1 for d in diffs if d > 0)}/{len(diffs)}")
        print(f"   符号检验 p = {p2:.4f}  "
              f"{'✅ 显著' if p2 < 0.05 else '❌ 不显著 (样本太少)'}")

    print("\n4. 唯一性 (多样性)")
    ub = len(set(s for _, _, s in base))
    ur = len(set(s for _, _, s in rl))
    print(f"   SFT 唯一 {ub}/{len(br)} ({ub/len(br)*100:.1f}%)")
    print(f"   RL  唯一 {ur}/{len(rr)} ({ur/len(rr)*100:.1f}%)")

    print("\n" + "=" * 70)
    verdict = ("✅ RL 显著优于 SFT 基线" if p_mw < 0.05 else
               "🟡 RL 与 SFT 基线无统计显著差异 (需要更长训练或更大样本)")
    print(f"结论: {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
