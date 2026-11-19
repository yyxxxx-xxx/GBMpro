"""
评估器校准验证脚本 (2026-11-18)
================================

验证 GBMPhysicalEvaluator 重构后的三项关键指标:

1. 一致性: 评估器给出的 ADMET 概率 vs 论文评测口径 (rescore_parallel.py 用的
   同一套 admet-ai) 是否完全一致 —— 验证"反馈闭环与最终评测说同一套话"
2. 剪枝率: 真实上市药物被误杀的比例 (目标 < 10%)
3. Reward 分布: 是否连续可排序, 是否还有断崖

用法:
    python3 scripts/calibrate_evaluator.py [--n 300]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gbm_physical_evaluator import GBMPhysicalEvaluator  # noqa: E402


def load_real_drugs(n: int):
    p = PROJECT_ROOT / "data" / "lora_datasets" / "real_molecules.json"
    if not p.exists():
        p = PROJECT_ROOT.parent / "gbm_project" / "data" / "lora_datasets" / "real_molecules.json"
    data = json.load(open(p, encoding="utf-8"))
    return [d["smiles"] for d in data][:n]


def load_ours_molecules():
    import pandas as pd
    out = {}
    base = PROJECT_ROOT / "final_results" / "comparative" / "rescored" / "comparison_final"
    for t in ("EGFR", "VEGFR", "IDH1"):
        f = base / f"{t}.csv"
        if f.exists():
            df = pd.read_csv(f)
            out[t] = df["smiles"].tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    ev = GBMPhysicalEvaluator(vina_executable=None, receptor_pdbqt=None)
    print(f"admet-ai 后端: {'启用 ✓' if ev._admet.use_admet_ai else '未启用 ✗'}")
    print()

    # ============ 1. 真实药物 ============
    drugs = load_real_drugs(args.n)
    print(f"=== 1. 真实上市药物 (n={len(drugs)}) ===")
    rows = ev._admet.compute_batch(drugs)

    pruned = 0
    rewards = []
    for smi, ad in zip(drugs, rows):
        res = ev.evaluate(smi)
        pruned += res.is_pruned
        rewards.append(res.reward)

    print(f"  剪枝率          : {100 * pruned / len(drugs):.1f}%   (目标 < 10%)")
    print(f"  reward mean     : {st.mean(rewards):.3f}")
    print(f"  reward min/max  : {min(rewards):.3f} / {max(rewards):.3f}")

    # 一致性: 评估器 vs 评测口径
    print()
    print("  --- 与论文评测口径 (admet-ai) 的一致性 ---")
    mism = 0
    for smi, ad in zip(drugs, rows):
        res = ev.evaluate(smi)
        for key, attr in (("dili_prob", "dili_prob"), ("hERG_prob", "herg_prob"),
                          ("BBB_prob", "bbb_score"), ("Pgp_prob", "pgp_prob"),
                          ("AMES_prob", "ames_prob")):
            a, b = ad.get(key), getattr(res, attr, None)
            if a is None or b is None:
                continue
            if abs(float(a) - float(b)) > 1e-6:
                mism += 1
    total = len(drugs) * 5
    print(f"  端点值不一致数  : {mism} / {total}  ({100 * mism / total:.2f}%)")

    # ============ 2. Ours 生成分子 ============
    ours = load_ours_molecules()
    if ours:
        print()
        print("=== 2. Ours 生成分子 ===")
        allrows = []
        for t, smis in ours.items():
            r = ev.evaluate_batch(smis)
            pr = sum(1 for x in r if x.is_pruned)
            print(f"  {t:6s} n={len(smis):3d}  剪枝率={100 * pr / len(smis):5.1f}%  "
                  f"reward mean={st.mean([x.reward for x in r]):.3f}")
            allrows.extend(r)

        n = len(allrows)
        print()
        print("  --- 风险端点阳性率 (反馈闭环看到的就是这些数字) ---")
        for attr, label in (("herg_prob", "hERG"), ("dili_prob", "DILI"),
                            ("bbb_score", "BBB+"), ("pgp_prob", "Pgp+"),
                            ("ames_prob", "AMES+")):
            vals = [getattr(x, attr) for x in allrows
                    if getattr(x, attr, None) is not None and getattr(x, attr) == getattr(x, attr)]
            if vals:
                print(f"    {label:6s} >0.5 比例: {100 * sum(1 for v in vals if v > 0.5) / len(vals):5.1f}%")

        # ============ 3. Reward 单调性 ============
        print()
        print("=== 3. Reward 断崖检查 (其余端点固定, 只扫 TPSA) ===")
        mpo = ev._mpo
        prev = None
        for t in (40, 60, 80, 90, 100, 110, 115, 120, 130, 140):
            r = mpo.compute_reward(
                vina_norm=0.7, dili_norm=0.9, herg_norm=0.8, bbb_norm=0.6,
                tpsa_norm=mpo.normalize_tpsa(t), mw_norm=1.0,
                pgp_norm=0.8, ames_norm=0.9, qed_norm=0.6,
            )
            delta = "" if prev is None else f"  Δ={r - prev:+.4f}"
            print(f"    TPSA={t:5.0f}  norm={mpo.normalize_tpsa(t):+.3f}  reward={r:.4f}{delta}")
            prev = r


if __name__ == "__main__":
    main()
