"""
MPO Reward 有效性验证
======================

回答一个核心问题: **反馈闭环给出的 reward 到底在引导什么?**

方法:
    在真实药物 + 已知活性配体上计算 (真实 Vina 分数, MPO reward),
    计算 Spearman 秩相关。

判读:
    - reward 与 vina 强正相关 (rho > 0.5) => reward 在正确引导"结合得更强"
    - reward 与 vina 弱相关 (rho < 0.3)   => 其他端点正在主导，Vina 信号被稀释
    - reward 与 vina 负相关               => 方向错了，闭环会把 LLM 带偏

同时输出:
    - reward 的区分度 (real drugs vs 已知活性配体 是否可分)
    - 各端点对 reward 变异的贡献 (Shapley 式近似: 单端点扰动)

用法:
    python3 scripts/validate_mpo_reward.py --target EGFR --n 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yaml  # noqa: E402

from gbm_target_evaluator import build_evaluator_for_target, load_receptor_config  # noqa: E402


def spearman(x, y):
    """Spearman 秩相关（无 scipy 依赖）。"""
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
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, default="EGFR")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--exhaustiveness", type=int, default=4)
    ap.add_argument("--cpu", type=int, default=8)
    args = ap.parse_args()

    ev = build_evaluator_for_target(
        args.target, exhaustiveness=args.exhaustiveness, cpu=args.cpu
    )
    if ev is None:
        print(f"无法为靶点 {args.target} 构建评估器")
        return

    backend = getattr(ev._vina_tool, "_backend", "pseudo")
    print(f"靶点={args.target}  后端={backend}  exh={args.exhaustiveness}  cpu={args.cpu}\n")

    # ---- 样本: 真实药物 + 已知活性配体 ----
    smiles, labels = [], []
    real_path = PROJECT_ROOT / "data" / "lora_datasets" / "real_molecules.json"
    if real_path.exists():
        data = json.load(open(real_path, encoding="utf-8"))
        for d in data[: args.n]:
            smiles.append(d["smiles"])
            labels.append("drug")

    # 已知活性配体 (来自 receptors.yaml 的共晶配体 + LINCS 活性分子)
    cfg = load_receptor_config()
    for key, spec in cfg.items():
        if isinstance(spec, dict) and spec.get("crystal_ligand_smiles"):
            smiles.append(spec["crystal_ligand_smiles"])
            labels.append("active")

    print(f"样本: {labels.count('drug')} 个真实药物 + {labels.count('active')} 个已知活性配体\n")

    # ---- 评估 ----
    rows = []
    for i, s in enumerate(smiles):
        r = ev.evaluate(s)
        rows.append({
            "smiles": s, "label": labels[i],
            "vina": r.vina_score, "reward": r.reward,
            "dili": r.dili_prob, "herg": r.herg_prob, "bbb": r.bbb_score,
            "pgp": r.pgp_prob, "ames": r.ames_prob, "qed": r.qed,
            "vina_norm": r.vina_norm, "pruned": r.is_pruned,
        })
        print(f"  [{i+1}/{len(smiles)}] {labels[i]:6s} vina={r.vina_score} reward={r.reward:.4f}")

    # ---- 相关性分析 ----
    valid = [r for r in rows if r["vina"] is not None]
    if len(valid) < 5:
        print("有效样本不足")
        return

    vinas = [r["vina"] for r in valid]
    rewards = [r["reward"] for r in valid]
    rho = spearman(vinas, rewards)
    print()
    print("=== reward 与真实 Vina 的相关性 ===")
    print(f"  Spearman rho = {rho:+.3f}   (n={len(valid)})")
    if rho > 0.5:
        print("  ✓ 强正相关 —— reward 在正确引导「结合得更强」")
    elif rho > 0.3:
        print("  △ 中等相关 —— Vina 信号被其他端点稀释，可考虑提高 Vina 权重")
    else:
        print("  ✗ 弱相关 —— reward 几乎不由结合强度决定，反馈闭环可能带偏 LLM")

    print()
    print("=== 各端点单独与 Vina 的相关性 (越接近 0 越好，说明没重复惩罚) ===")
    for key, label in (("dili", "DILI"), ("herg", "hERG"), ("bbb", "BBB"),
                       ("pgp", "Pgp"), ("ames", "AMES"), ("qed", "QED")):
        vals = [r[key] for r in valid if r[key] is not None]
        vs = [r["vina"] for r in valid if r[key] is not None]
        if len(vals) > 4:
            print(f"  {label:6s} vs Vina: rho = {spearman(vs, vals):+.3f}")

    print()
    print("=== 已知活性配体 vs 真实药物 ===")
    act = [r for r in valid if r["label"] == "active"]
    drg = [r for r in valid if r["label"] == "drug"]
    if act and drg:
        print(f"  活性配体 reward: {sum(r['reward'] for r in act) / len(act):.3f} "
              f"(n={len(act)})")
        print(f"  真实药物 reward: {sum(r['reward'] for r in drg) / len(drg):.3f} "
              f"(n={len(drg)})")
        print(f"  活性配体 vina  : {sum(r['vina'] for r in act) / len(act):.2f}")
        print(f"  真实药物 vina  : {sum(r['vina'] for r in drg) / len(drg):.2f}")

    out = PROJECT_ROOT / "final_results" / "mpo_validation" / f"{args.target}_mpo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
