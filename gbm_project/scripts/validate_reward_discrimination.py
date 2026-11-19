"""
MPO Reward 对靶点活性的判别力验证（严谨版）
=========================================

设计:
    用 **已验证活性的 PubChem 分子**（IC50 <= 10 uM，来自 BioAssay）作为正样本，
    用 **同靶点的随机真实药物**作为对照，比较:
        - MPO reward 的 AUC（活性 vs 非活性）
        - Vina 分数单独的 AUC（对照基线）
    以及 reward 与 Vina 的 Spearman 相关性。

    这比"reward vs Vina 相关性"更接近论文要回答的问题:
    **反馈闭环给的分数, 能不能区分出真正能结合的分子?**

    注意: 这里不期望 reward 的 AUC 超过 Vina —— reward 是多目标折中，
    一个分子结合强但毒性高，reward 低是完全正确的。关键是看 reward 是否
    **保留了足够的判别力**（AUC 不塌到 0.5）。

用法:
    python3 scripts/validate_reward_discrimination.py --target EGFR --n 60
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gbm_target_evaluator import build_evaluator_for_target  # noqa: E402


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def auc(pos: list, neg: list) -> float:
    """Mann-Whitney U 形式的 AUC（含 tie 处理）。"""
    if not pos or not neg:
        return float("nan")
    combined = [(v, 1) for v in pos] + [(v, 0) for v in neg]
    combined.sort(key=lambda t: t[0])
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    sum_pos = sum(ranks[i] for i, (_, lab) in enumerate(combined) if lab == 1)
    n1, n0 = len(pos), len(neg)
    return (sum_pos - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, default="EGFR")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--exhaustiveness", type=int, default=4)
    ap.add_argument("--cpu", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    ev = build_evaluator_for_target(
        args.target, exhaustiveness=args.exhaustiveness, cpu=args.cpu
    )
    if ev is None:
        print(f"无法为靶点 {args.target} 构建评估器")
        return

    # ---- 正样本: 已验证活性的分子 ----
    act_path = PROJECT_ROOT / "data" / "lora_datasets" / "target_active_molecules.json"
    actives = [d for d in json.loads(act_path.read_text(encoding="utf-8"))
               if d["target"] == args.target]
    random.shuffle(actives)
    actives = actives[: args.n]
    active_smiles = [a["smiles"] for a in actives]

    # ---- 负样本: 其他靶点的活性分子 ----
    #
    # 修复 (2026-11-18): 原实现用 real_molecules.json (LINCS 药物库) 当负样本，
    # 造成**标签泄漏** —— Gefitinib、Erlotinib 这些标杆 EGFR 抑制剂就在 LINCS
    # 库里, 被当成了"不结合 EGFR 的对照", 导致 AUC 完全失去意义
    # (Vina 的 AUC 甚至只有 0.488, 比随机还差)。
    #
    # 正确做法: 负样本用**其他靶点的活性分子**。它们是经过 PubChem BioAssay
    # 验证过的真实活性分子, 但作用于完全不同的靶点 (如 IDH1 的抑制剂不结合
    # EGFR 的 ATP 位点), 因此可验证地是阴性。
    all_actives = json.loads(act_path.read_text(encoding="utf-8"))
    other = [d for d in all_actives if d["target"] != args.target]
    random.shuffle(other)
    negatives = [d["smiles"] for d in other][: args.n]

    print(f"靶点={args.target}  正样本(该靶点已验证活性)={len(active_smiles)}  "
          f"负样本(其他靶点活性分子)={len(negatives)}  exh={args.exhaustiveness}\n")

    def run(smiles_list, label):
        out = []
        for i, s in enumerate(smiles_list):
            r = ev.evaluate(s)
            out.append({
                "smiles": s, "label": label,
                "vina": r.vina_score, "reward": r.reward,
                "dili": r.dili_prob, "herg": r.herg_prob, "bbb": r.bbb_score,
                "pgp": r.pgp_prob, "ames": r.ames_prob, "qed": r.qed,
                "pruned": r.is_pruned, "reason": r.prune_reason,
            })
            if (i + 1) % 10 == 0:
                print(f"  {label}: {i+1}/{len(smiles_list)}")
        return out

    rows = run(active_smiles, "active") + run(negatives, "drug")

    pos = [r for r in rows if r["label"] == "active" and r["vina"] is not None]
    neg = [r for r in rows if r["label"] == "drug" and r["vina"] is not None]

    print()
    print("=== 判别力 (AUC) ===")
    a_vina = auc([-r["vina"] for r in pos], [-r["vina"] for r in neg])
    a_rew = auc([r["reward"] for r in pos], [r["reward"] for r in neg])
    print(f"  Vina 分数 (取负, 越负越好): AUC = {a_vina:.3f}")
    print(f"  MPO reward                   : AUC = {a_rew:.3f}")
    if a_rew < 0.5:
        print("  ✗ reward 的判别力低于随机 —— 反馈闭环会给出误导性信号")
    elif a_rew >= 0.65:
        print("  ✓ reward 保留了有意义的判别力")
    else:
        print("  △ reward 判别力偏弱（多目标折中的代价）")

    print()
    print("=== reward 与 Vina 的 Spearman 相关性 ===")
    v = [r["vina"] for r in rows if r["vina"] is not None]
    rw = [r["reward"] for r in rows if r["vina"] is not None]
    print(f"  全部 (n={len(v)}): rho = {spearman(v, rw):+.3f}")
    keep = [r for r in rows if r["vina"] is not None and not r["pruned"]]
    if len(keep) > 5:
        print(f"  排除剪枝 (n={len(keep)}): rho = "
              f"{spearman([r['vina'] for r in keep], [r['reward'] for r in keep]):+.3f}")

    print()
    print("=== 活性分子 vs 药物 的均值 ===")
    for lab, grp in (("活性", pos), ("其他靶点", neg)):
        print(f"  {lab}: vina={sum(r['vina'] for r in grp)/len(grp):6.2f}  "
              f"reward={sum(r['reward'] for r in grp)/len(grp):.3f}  "
              f"dili={sum(r['dili'] for r in grp)/len(grp):.2f}  "
              f"herg={sum(r['herg'] for r in grp)/len(grp):.2f}  "
              f"bbb={sum(r['bbb'] for r in grp)/len(grp):.2f}  "
              f"pgp={sum(r['pgp'] for r in grp)/len(grp):.2f}")

    pr_a = sum(1 for r in pos if r["pruned"])
    pr_n = sum(1 for r in neg if r["pruned"])
    print()
    print(f"  剪枝率: 活性 {100*pr_a/len(pos):.1f}%  其他靶点 {100*pr_n/len(neg):.1f}%")

    out = PROJECT_ROOT / "final_results" / "mpo_validation" / f"{args.target}_discrimination.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()
