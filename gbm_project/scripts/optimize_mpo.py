"""
MPO 聚合方式优化
=================

问题 (2026-11-18 实测):
    在 EGFR 上，MPO reward 与真实 Vina 分数的 Spearman rho 只有 +0.107 ——
    几乎不相关。根因:

        DILI vs Vina: rho = -0.586
        Pgp vs Vina: rho = -0.423
        hERG vs Vina: rho = -0.182

    结合更强的分子往往更大、更疏水、TPSA 更低 => 毒性与外排风险同时升高。
    **加权几何平均是乘性的**: 一个端点变差会成倍地拉低总分, 于是 Vina 的
    优势（vina_norm 从 0.5 升到 1.0, 只乘 2 倍）被 DILI 的劣势
    （dili_norm 从 0.9 降到 0.3, 除以 3 倍）彻底抵消。

    换句话说: 几何平均在"目标反向"的端点组合下会产生目标冲突抵消,
    这正是 REINVENT4 用几何平均时不会遇到的情形（它的端点基本同向）。

本脚本做的事:
    在已保存的对接结果上，网格搜索 (权重组合, 聚合方式)，找出
    rho 最大化、同时各端点仍保留有效权重的配置。

用法:
    python3 scripts/optimize_mpo.py --target EGFR
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

ENDPOINTS = ["vina", "dili", "herg", "bbb", "pgp", "ames", "qed", "physchem"]


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


def aggregate_one(sample: dict, weights: dict, mode: str, floor: float) -> float:
    """sample: 单个分子的 端点 -> [0,1] 归一化分数（已取补数，1=最好）"""
    if mode == "geometric":
        total = sum(weights[e] * np.log(max(sample[e], floor)) for e in weights)
        return float(np.exp(total))
    elif mode == "arithmetic":
        return float(sum(weights[e] * sample[e] for e in weights))
    elif mode == "harmonic":
        total = sum(weights[e] / max(sample[e], floor) for e in weights)
        return float(len(weights) / total)
    raise ValueError(mode)


def aggregate_all(eps: dict, weights: dict, mode: str, floor: float):
    """eps: 端点 -> 跨样本的列表。返回逐样本 reward 列表。"""
    n = len(next(iter(eps.values())))
    out = []
    for i in range(n):
        sample = {e: eps[e][i] for e in eps}
        out.append(aggregate_one(sample, weights, mode, floor))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, default="EGFR")
    ap.add_argument("--data", type=str, default=None)
    args = ap.parse_args()

    path = Path(args.data) if args.data else (
        PROJECT_ROOT / "final_results" / "mpo_validation" / f"{args.target}_mpo.json"
    )
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in rows if r.get("vina") is not None]
    print(f"样本 n={len(rows)}  ({path.name})\n")

    # ---- 构造归一化端点值 ----
    def vina_norm(v):
        return max(0.0, min(1.0, (-4.0 - v) / 8.0))

    def tpsa_norm(t):
        if t > 140: return -1.0
        if t > 90: return max(-1.0, 1.0 - 0.7 * (t - 90.0) / 50.0)
        if t >= 40: return 1.0
        if t >= 20: return 0.5 + 0.5 * (t - 20.0) / 20.0
        return -1.0

    def mw_norm(m):
        if 300 <= m <= 500: return 1.0
        if m < 300: return max(0.0, m / 300.0)
        return max(0.0, 1.0 - 0.7 * (m - 500.0) / 200.0)

    # TPSA/MW 需要描述符 —— 用 RDKit 现算
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp")
    from rdkit.Chem import Descriptors

    eps = {e: [] for e in ENDPOINTS}
    vinas = []
    for r in rows:
        m = Chem.MolFromSmiles(r["smiles"])
        t = Descriptors.TPSA(m) if m else 60.0
        w = Descriptors.MolWt(m) if m else 400.0
        tn, wn = tpsa_norm(t), mw_norm(w)
        phys = float(np.sqrt(max(tn, 0.05) * max(wn, 0.05)))
        eps["vina"].append(vina_norm(r["vina"]))
        eps["dili"].append(1.0 - (r["dili"] or 0.5))
        eps["herg"].append(1.0 - (r["herg"] or 0.5))
        eps["bbb"].append(r["bbb"] or 0.3)
        eps["pgp"].append(1.0 - (r["pgp"] or 0.5))
        eps["ames"].append(1.0 - (r["ames"] or 0.5))
        eps["qed"].append(r["qed"] or 0.5)
        eps["physchem"].append(phys)
        vinas.append(r["vina"])

    # ---- 当前配置基线 ----
    CURRENT = {
        "vina": 0.25, "dili": 0.12, "herg": 0.10, "bbb": 0.20,
        "pgp": 0.15, "ames": 0.08, "qed": 0.05, "physchem": 0.05,
    }
    print("=== 当前配置基线 ===")
    for mode in ("geometric", "arithmetic"):
        rew = aggregate_all(eps, CURRENT, mode, 0.05)
        print(f"  {mode:11s} rho = {spearman(vinas, rew):+.3f}")

    # ---- 网格搜索 ----
    print()
    print("=== 权重网格搜索 (目标: rho 最大化, 且各端点权重 > 0) ===")
    vina_grid = [0.25, 0.35, 0.45, 0.55, 0.65]
    risk_grid = [0.05, 0.10, 0.15]
    cns_grid = [0.10, 0.20, 0.30]

    results = []
    for v_w, r_w, c_w in itertools.product(vina_grid, risk_grid, cns_grid):
        # 结构: vina / 风险(dili+herg+ames) / CNS(bbb+pgp) / qed / physchem
        # 风险内部按 dili:herg:ames = 0.5:0.3:0.2 分
        risk_total = r_w
        w = {
            "vina": v_w,
            "dili": risk_total * 0.50,
            "herg": risk_total * 0.30,
            "ames": risk_total * 0.20,
            "bbb": c_w * 0.55,
            "pgp": c_w * 0.45,
            "qed": 0.05,
            "physchem": 0.05,
        }
        tot = sum(w.values())
        w = {k: v / tot for k, v in w.items()}
        for mode in ("geometric", "arithmetic", "harmonic"):
            rew = aggregate_all(eps, w, mode, 0.05)
            rho = spearman(vinas, rew)
            results.append((rho, mode, dict(w)))

    results.sort(key=lambda x: -x[0])
    print()
    print("  Top 10:")
    for rho, mode, w in results[:10]:
        ws = " ".join(f"{k}={v:.2f}" for k, v in w.items() if v >= 0.04)
        print(f"    rho={rho:+.3f}  {mode:11s} {ws}")

    print()
    print("  各聚合方式的最佳:")
    for mode in ("geometric", "arithmetic", "harmonic"):
        best = max([r for r in results if r[1] == mode], key=lambda x: x[0])
        ws = " ".join(f"{k}={v:.2f}" for k, v in best[2].items() if v >= 0.04)
        print(f"    {mode:11s} rho={best[0]:+.3f}  {ws}")

    # ---- 端点消融: 去掉每个端点后 rho 变化 ----
    print()
    print("=== 端点消融 (基于当前配置, 看哪些端点在稀释 Vina 信号) ===")
    for mode in ("geometric", "arithmetic"):
        base = aggregate_all(eps, CURRENT, mode, 0.05)
        base_rho = spearman(vinas, base)
        print(f"  {mode} 基线 rho={base_rho:+.3f}")
        for drop in ENDPOINTS:
            if drop == "vina":
                continue
            w2 = {k: v for k, v in CURRENT.items() if k != drop}
            tot = sum(w2.values())
            w2 = {k: v / tot for k, v in w2.items()}
            rew = aggregate_all(eps, w2, mode, 0.05)
            r2 = spearman(vinas, rew)
            print(f"    去掉 {drop:9s} -> rho={r2:+.3f}  ({r2 - base_rho:+.3f})")

    out = PROJECT_ROOT / "final_results" / "mpo_validation" / f"{args.target}_optimize.json"
    out.write_text(json.dumps(
        [{"rho": r, "mode": m, "weights": w} for r, m, w in results[:50]],
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTop50 已保存: {out}")


if __name__ == "__main__":
    main()
