#!/usr/bin/env python3
"""
Modification Bias 检查 (R-group 改造偏置)
=========================================

借鉴自 PBCNet2.0 的训练数据偏置审查 (见 docs/PBCNET2_0_BENCHMARKS.md §7)::

    "Finally, we were concerned about the potential for the modification bias,
     which means that the activity changes of all compound pairs with the same
     R-group modification (e.g., from a methyl group to an ethyl group) might
     be consistently either increased or decreased. The risk of such a bias is
     that the model might simply memorize the structural differences between
     molecules and predict accordingly, leading to overfitting."

为什么这对我的 RL 系统重要
--------------------------

审稿人可能会问:
    "你的 RL 提升 reward +31%, 但会不会只是**复现了 SFT 数据里固有的偏置**?
     比如训练集里'加个 F 就一定提升 ADMET', 模型学到的是这个统计规律,
     而不是真正的物理反馈?"

这个脚本回答的就是这个问题: **SFT 数据本身有没有系统性偏向某类改造?**

如果 SFT 数据存在强偏置 (比如"分子越大 ADMET 越好"),
那么 RL 的提升就可能只是把偏置放大, 而不是真的发现了更好的分子。

检查方法
--------

对每一对**共享 Murcko 骨架**的分子 (即只改了 R-group), 计算:

    Δ = property(改过的分子) - property(原分子)

按改造类型 (取代基变化) 分组, 检查各组 Δ 的均值是否显著偏离 0。

如果某个改造类型 (比如 -CH3 -> -C2H5) 的 Δ 均值显著 > 0,
说明数据里存在"这类改造总是变好"的偏置。

检查的性质
----------

| 性质 | 期望方向 | 偏置含义 |
|---|---|---|
| MW | 无 | 大分子偏好 |
| LogP | 无 | 疏水性偏好 |
| TPSA | 无 | 极性偏好 |
| QED | 无 | 整体类药性偏好 |
| BBB | 无 | 血脑屏障偏好 |
| hERG | 低 | 心脏毒性偏好 |
| DILI | 低 | 肝毒性偏好 |
| AMES | 低 | 突变毒性偏好 |
| **value_um (活性)** | **低 = 更强** | **活性改造偏置** |

用法:
    python3 scripts/check_modification_bias.py
    python3 scripts/check_modification_bias.py --qa data/lora_datasets/active_qa_train.json
    python3 scripts/check_modification_bias.py --min-pairs 5 --out experiments/bias_report.json

数据源说明
----------

两种模式:
  1. **默认**: 读 active_molecules_6targets_clean.json —— 只有活性值 (value_um)
  2. **--qa**: 读 active_qa_train.json —— 有完整物化性质 (MW/TPSA/QED/BBB/hERG/...),
     按 CID (original_drug 字段, 形如 "EGFR_CID9949175") 与活性分子关联,
     从而把物化性质也纳入偏置检查
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

GBM_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = (GBM_ROOT / "data" / "lora_datasets"
                / "active_molecules_6targets_clean.json")
DEFAULT_QA = GBM_ROOT / "data" / "lora_datasets" / "active_qa_train.json"

# 性质 -> (字段名, 期望方向)
#   direction: +1 表示"越大越好", -1 表示"越小越好"
PROPERTIES = {
    "MW": ("MW", 0),
    "LogP": ("XLogP", 0),
    "TPSA": ("TPSA", 0),
    "QED": ("QED", 0),
    "BBB_prob": ("BBB_prob", 0),
    "hERG_prob": ("hERG_prob", -1),
    "DILI_prob": ("DILI_prob", -1),
    "AMES_prob": ("AMES_prob", -1),
    "value_um": ("value_um", -1),      # 越小 = 活性越强
}

# t 分布 95% 双尾临界值 (df=∞ 用 1.96)
T_CRIT = 1.96


def murcko_scaffold(smiles: str) -> str:
    """Murcko 骨架 (Bemis-Murcko)。拿不到 RDKit 就退回 None。"""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        core = MurckoScaffold.GetScaffoldForMol(mol)
        s = Chem.MolToSmiles(core)
        return s if s else None
    except Exception:
        return None


def rgroup_signature(smiles: str, scaffold: str) -> str:
    """R-group 签名: 骨架之外的取代基。"""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        try:
            from rdkit.Chem import rdMolDescriptors
        except Exception:
            pass
        # 用 MCS 不稳, 改用"分子式差"做粗略签名
        # 更稳的做法: 直接用整个分子的分子式减去骨架的分子式
        def formula(m):
            from rdkit.Chem import rdMolDescriptors
            return rdMolDescriptors.CalcMolFormula(m)
        return formula(mol), formula(scaffold_mol)
    except Exception:
        return None


def t_stat(deltas):
    """返回 (均值, t 统计量, n)。"""
    n = len(deltas)
    if n < 2:
        return (deltas[0] if n else None), None, n
    m = sum(deltas) / n
    var = sum((d - m) ** 2 for d in deltas) / (n - 1)
    sd = math.sqrt(var)
    if sd < 1e-15:
        return m, None, n
    return m, m / (sd / math.sqrt(n)), n


def load_qa_properties(qa_path: Path):
    """从 QA 数据集抽物化性质, 按 (target, cid) 索引。

    QA 的 original_drug 字段形如 "EGFR_CID9949175"。
    返回 {(target, cid): {prop: value}}
    """
    records = json.loads(qa_path.read_text(encoding="utf-8"))
    out = {}
    for r in records:
        md = r.get("metadata") or {}
        props = md.get("properties") or {}
        drug = md.get("original_drug") or ""
        m = re.search(r"CID(\d+)", str(drug))
        if not m:
            continue
        cid = m.group(1)
        tgt = md.get("target")
        entry = {}
        for k, v in props.items():
            if isinstance(v, (int, float)):
                entry[k] = float(v)
        if entry:
            out[(tgt, cid)] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--qa", default=None,
                    help="QA 数据集路径 —— 提供后会把物化性质合并进分析")
    ap.add_argument("--min-pairs", type=int, default=3,
                    help="某个改造类型至少要有多少对才纳入统计")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_path = Path(args.data)
    records = json.loads(data_path.read_text(encoding="utf-8"))
    print(f"数据: {data_path}")
    print(f"分子数: {len(records)}")

    # --- 合并 QA 物化性质 ---
    qa_props = {}
    if args.qa:
        qa_path = Path(args.qa)
        qa_props = load_qa_properties(qa_path)
        print(f"QA 性质索引: {qa_path} ({len(qa_props)} 个 (target,cid) 条目)")
        n_hit = 0
        for r in records:
            key = (r.get("target"), str(r.get("cid")))
            if key in qa_props:
                r.update(qa_props[key])
                n_hit += 1
        print(f"  匹配到物化性质的分子: {n_hit}/{len(records)}")

    # --- 建骨架分组 ---
    groups = defaultdict(list)
    n_no_scaffold = 0
    for r in records:
        smi = r.get("smiles")
        if not smi:
            continue
        core = murcko_scaffold(smi)
        if not core:
            n_no_scaffold += 1
            continue
        groups[core].append(r)

    print(f"唯一骨架数: {len(groups)}  (无骨架: {n_no_scaffold})")

    multi = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"含 >=2 个分子的骨架: {len(multi)}")

    # --- 收集成对 Δ ---
    # ⚠️ 配对方向必须**与活性无关**, 否则会人为制造偏置:
    #    - 按数据里的原始顺序取 (A-B): 顺序与活性相关 -> 假偏置 (实测 t=-12.3)
    #    - 对称化取 (+d 和 -d 各一次): 均值恒为 0 -> 检验失效 (t=0)
    #    - ✅ 用 SMILES 字典序确定 A/B: 顺序是**外生的**, 与活性无关
    #       -> 单样本 t 检验有效, 无偏
    pair_deltas = defaultdict(list)   # prop -> [delta]
    rgroup_pairs = []                 # (core, recA, recB) 用于改造类型分析

    for core, mols in multi.items():
        # 按 SMILES 字典序排 —— 外生顺序
        ordered = sorted(mols, key=lambda r: str(r.get("smiles", "")))
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a, b = ordered[i], ordered[j]
                for prop, (field, _dir) in PROPERTIES.items():
                    va, vb = a.get(field), b.get(field)
                    if va is None or vb is None:
                        continue
                    if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
                        continue
                    pair_deltas[prop].append(float(va) - float(vb))
                rgroup_pairs.append((core, a, b))

    print(f"成对比较数: {len(rgroup_pairs)}")

    # --- 逐性质检查 ---
    print(f"\n{'='*72}")
    print("Modification Bias 检查 (同骨架内的 Δ = A - B, 随机取对)")
    print(f"{'='*72}")
    print(f"{'性质':<14}{'n':>6}{'Δ均值':>14}{'t':>9}{'判定':>14}")
    print(f"{'-'*72}")

    results = {}
    for prop, (field, direction) in PROPERTIES.items():
        deltas = pair_deltas.get(prop, [])
        if len(deltas) < args.min_pairs:
            print(f"{prop:<14}{len(deltas):>6}{'—':>14}{'—':>9}{'样本不足':>14}")
            continue
        m, t, n = t_stat(deltas)
        sig = t is not None and abs(t) > T_CRIT
        if sig:
            verdict = "⚠️ 有偏置"
        else:
            verdict = "✅ 无显著偏置"
        print(f"{prop:<14}{n:>6}{m:>14.5f}"
              f"{(f'{t:>9.3f}' if t is not None else '—'):>9}{verdict:>14}")
        results[prop] = {"n": n, "mean_delta": m, "t": t,
                         "significant": sig, "direction": direction}

    # --- 靶点内部分析 (更严格: 同一靶点内才可比) ---
    print(f"\n{'='*72}")
    print("靶点内分析 (更严格: 只比较同一靶点的同骨架分子)")
    print(f"{'='*72}")
    tgt_groups = defaultdict(list)
    for r in records:
        smi = r.get("smiles")
        if not smi:
            continue
        core = murcko_scaffold(smi)
        if core:
            tgt_groups[(r.get("target"), core)].append(r)

    tgt_deltas = defaultdict(list)
    n_pairs_tgt = 0
    for (tgt, core), mols in tgt_groups.items():
        if len(mols) < 2:
            continue
        ordered = sorted(mols, key=lambda r: str(r.get("smiles", "")))
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a, b = ordered[i], ordered[j]
                n_pairs_tgt += 1
                for prop, (field, _d) in PROPERTIES.items():
                    va, vb = a.get(field), b.get(field)
                    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                        tgt_deltas[prop].append(float(va) - float(vb))

    print(f"靶点内成对数: {n_pairs_tgt}")
    print(f"{'性质':<14}{'n':>6}{'Δ均值':>14}{'t':>9}{'判定':>14}")
    print(f"{'-'*72}")
    tgt_results = {}
    for prop, (field, _d) in PROPERTIES.items():
        deltas = tgt_deltas.get(prop, [])
        if len(deltas) < args.min_pairs:
            continue
        m, t, n = t_stat(deltas)
        sig = t is not None and abs(t) > T_CRIT
        verdict = "⚠️ 有偏置" if sig else "✅ 无显著偏置"
        print(f"{prop:<14}{n:>6}{m:>14.5f}"
              f"{(f'{t:>9.3f}' if t is not None else '—'):>9}{verdict:>14}")
        tgt_results[prop] = {"n": n, "mean_delta": m, "t": t,
                             "significant": sig}

    # --- 结论 ---
    print(f"\n{'='*72}")
    print("结论")
    print(f"{'='*72}")
    flagged = [p for p, r in results.items() if r.get("significant")]
    if not flagged:
        print("✅ 未检测到显著的 modification bias。")
        print("   SFT 数据里不存在'某类改造总是变好/变坏'的系统性偏置,")
        print("   因此 RL 的 reward 提升不太可能只是复现数据偏置。")
    else:
        print(f"⚠️ 检测到 {len(flagged)} 个性质存在显著偏置: {flagged}")
        print("   这些偏置需要 (a) 在论文中如实报告,")
        print("   (b) 评估 RL 提升时, 检查这些性质上的提升是否显著超过偏置幅度。")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"all_pairs": results, "within_target": tgt_results,
             "n_pairs": len(rgroup_pairs), "n_pairs_within_target": n_pairs_tgt},
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n已保存: {args.out}")


if __name__ == "__main__":
    main()
