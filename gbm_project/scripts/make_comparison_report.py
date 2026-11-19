#!/usr/bin/env python3
"""
对比实验结果报告生成
====================

输入: final_results/comparative/rescored/ 下的重打分结果
输出: 论文可直接使用的 Markdown 报告 + CSV

报告包含:
    1. 主表: 每个靶点的四方法对比 (Vina / ADMET / 描述符 / 多样性)
    2. 统计检验: Vina 分数的 Mann-Whitney U 检验 (我们 vs 各基线)
    3. 管线验证: 共晶配体重对接 (证明对接管线正确)
    4. 数据集审计: 训练数据多样性 (审稿人必问)

用法:
    python scripts/make_comparison_report.py
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GBM_ROOT = PROJECT_ROOT / "gbm_project"
sys.path.insert(0, str(GBM_ROOT / "src"))
sys.path.insert(0, str(GBM_ROOT))

TARGETS = [
    ("EGFR", "EGFR (2JIV)"),
    ("VEGF_VEGFR", "VEGF/VEGFR (3WZD)"),
    ("IDH1_IDH2", "IDH1/IDH2 (6U4J)"),
]


def load_rescored(rescored_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """-> {target_key: {method: df}}"""
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for method_dir in sorted(rescored_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        for csv_path in sorted(method_dir.glob("*.csv")):
            stem = csv_path.stem.upper()
            tk = None
            # 注意顺序: 必须先匹配更具体的名字, 否则 "VEGF_VEGFR" 会被
            # "EGFR" 子串误判成 EGFR 靶点。
            if "IDH" in stem:
                tk = "IDH1_IDH2"
            elif "VEGF" in stem:
                tk = "VEGF_VEGFR"
            elif "EGFR" in stem:
                tk = "EGFR"
            if tk is None:
                continue
            df = pd.read_csv(csv_path)
            out.setdefault(tk, {})[method_dir.name] = df
    return out


def stat_row(df: pd.DataFrame) -> Dict:
    """单方法汇总。"""
    o = {"n": len(df)}
    v = pd.to_numeric(df.get("vina_docking_score"), errors="coerce").dropna()
    o["vina_n"] = len(v)
    o["vina_mean"] = round(v.mean(), 2) if len(v) else None
    o["vina_std"] = round(v.std(), 2) if len(v) else None
    o["vina_median"] = round(v.median(), 2) if len(v) else None
    o["vina_best"] = round(v.min(), 2) if len(v) else None
    o["vina_hit_rate"] = (round((v <= -7.0).mean() * 100, 1) if len(v) else None)

    def yes(col):
        if col not in df.columns:
            return None
        s = df[col].astype(str).str.strip().str.lower()
        m = s.isin(["yes", "no"])
        return round((s[m] == "yes").sum() / m.sum() * 100, 1) if m.any() else None

    o["bbb_pct"] = yes("BBB permeant")
    o["pgp_pct"] = yes("Pgp substrate")
    o["herg_pct"] = yes("hERG risk")
    o["dili_pct"] = yes("DILI risk")

    def num(col):
        if col not in df.columns:
            return None
        v = pd.to_numeric(df[col], errors="coerce").dropna()
        return round(v.mean(), 2) if len(v) else None

    o["qed"] = num("QED")
    o["mw"] = num("MW")
    o["tpsa"] = num("TPSA")
    o["logp"] = num("LogP")
    o["sa"] = num("Synthetic Accessibility")
    o["pains"] = num("PAINS #alerts")
    o["brenk"] = num("Brenk #alerts")

    if "Scaffold" in df.columns:
        sc = df["Scaffold"].astype(str)
        o["n_scaffolds"] = int(sc.nunique())
    else:
        o["n_scaffolds"] = None
    if "ScaffoldEntropy" in df.columns:
        e = pd.to_numeric(df["ScaffoldEntropy"], errors="coerce").dropna()
        o["scaffold_entropy"] = round(e.iloc[0], 3) if len(e) else None
    else:
        o["scaffold_entropy"] = None
    if "MaxTanimoto_ChEMBL" in df.columns:
        o["novelty"] = num("MaxTanimoto_ChEMBL")
    else:
        o["novelty"] = None

    # 类药预筛通过率：被挡下的分子无法进入对接阶段。
    # 这衡量的是"生成了多少真正可成药的分子"，是通用 LLM 基线的关键短板。
    if "vina_pose" in df.columns:
        pose = df["vina_pose"].astype(str)
        prefiltered = pose.str.startswith("prefilter:")
        o["prefilter_fail"] = int(prefiltered.sum())
        o["prefilter_pass_rate"] = round(
            (1 - prefiltered.mean()) * 100, 1) if len(df) else None
    else:
        o["prefilter_fail"] = 0
        o["prefilter_pass_rate"] = 100.0

    return o


def mwu_test(a: pd.Series, b: pd.Series) -> str:
    """Mann-Whitney U 检验（双侧），返回可读字符串。"""
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 3 or len(b) < 3:
        return "n/a"
    try:
        from scipy.stats import mannwhitneyu
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        return f"p={p:.4f} ({sig})"
    except Exception:
        return "n/a"


DISPLAY = {
    "PlayMolecule_AI": "PlayMolecule AI",
    "llasmol": "LlaSMol",
    "llamole": "Llamole (ablation)",
    "comparison_final": "GBM-PharmaMind (Ours)",
    "REINVENT": "REINVENT4 (Vina-RL)",
    "REINVENT4_TLRL": "REINVENT4 (TL+RL)",
    "base_mistral": "Mistral-7B (Base)",
    "base_qwen": "Qwen2-7B (Base)",
}
ORDER = ["PlayMolecule_AI", "llasmol", "llamole", "REINVENT",
         "REINVENT4_TLRL", "base_mistral", "base_qwen", "comparison_final"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescored_dir", type=str,
                        default=str(GBM_ROOT / "final_results/comparative/rescored"))
    parser.add_argument("--output_dir", type=str,
                        default=str(GBM_ROOT / "final_results/comparative/report"))
    args = parser.parse_args()

    rescored = Path(args.rescored_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_rescored(rescored)
    if not data:
        print(f"⚠ {rescored} 无数据")
        return

    md = ["# GBM 靶向药物分子生成 —— 对比实验报告", ""]
    md.append("## 0. 评测管线与方法学")
    md.append("| 项目 | 配置 |")
    md.append("|---|---|")
    md.append("| 分子对接 | AutoDock Vina 1.2.7 + Meeko 0.8.0；exhaustiveness=8，n_poses=5，"
              "box 24×24×24 Å 中心取共晶配体质心 |")
    md.append("| ADMET | admet-ai（52 端点），BBB/Pgp/hERG/DILI/AMES 阈值 0.5 |")
    md.append("| 描述符 | RDKit（MW/TPSA/LogP/HBD/HBA/RotB/PAINS/Brenk/NIH） |")
    md.append("| 合成可及性 | Ertl & Schuffenhauer SA_Score（RDKit Contrib） |")
    md.append("| 多样性 | Bemis-Murcko 骨架去重数 + 归一化 Shannon 熵 |")
    md.append("| 新颖性 | 与 ChEMBL 的最大 Tanimoto 相似度（越低越新颖） |")
    md.append("")
    md.append("> **公平性声明**：所有方法（含基线）均由**同一版本、同一参数**的本地管线"
              "重新评测。基线 CSV 中原有的 `vina_docking_score` 来自早期的经验公式"
              "伪对接，已被全部覆盖（保留在 `vina_docking_score_legacy` 列供追溯）。")
    md.append("")
    md.append("> **类药预筛**：进入对接前，所有方法共用同一道门槛"
              "（MW ≤ 500、重原子 ≤ 40、可旋转键 ≤ 10）。该门槛不是为了削弱基线，"
              "而是因为极端柔性分子（如 147 个可旋转键的长链烷烃）会让 Vina 的"
              "扭转树搜索指数级爆炸而无法完成对接 —— 这类结构本身也不具备"
              "成药价值。预筛通过率作为独立指标报告（见 §1.4）。")
    md.append("")
    md.append("> ⚠ **关于 REINVENT4 (Vina-RL) 的解读**：该基线在训练时**以真实 Vina "
              "对接分数作为强化学习的奖励信号**（DAP 策略梯度），即它的优化目标"
              "与评测指标 Vina 分数是同一个量。因此它在 Vina 上的优势是"
              "「用评测指标做训练目标」的直接结果，而非「更好的先验表示」。"
              "在论文中它应被视为**对接引导生成（docking-guided generation）的"
              "性能上界参照**，而不是与我们的模型在同等条件下的直接对比。"
              "把它与我们的方法并列比较时，必须同时说明这一不对称性。"
              "同样地，PlayMolecule AI 的分子在 EGFR/VEGFR 上呈现单一骨架"
              "（骨架熵 = 0.0），其高 Vina 分数伴随严重的模式坍缩。")
    md.append("")
    md.append("> **补充**：REINVENT4 并**没有**出现模式坍缩 —— 三个靶点上均为 "
              "20/20 唯一骨架、0% 近重复，因为其 RL 阶段启用了 "
              "IdenticalMurckoScaffold 多样性过滤器。因此它在结合能力与化学多样性上"
              "同时领先，是本次对比中表现最强的基线，也意味着审稿人更可能追问"
              "「为什么不对我们的模型也做同样的对接引导」。请参见 §7 建议第 2 条。")
    md.append("")

    # ---------- 主表 ----------
    md.append("## 1. 主结果")
    all_stats = {}
    for tk, tname in TARGETS:
        if tk not in data:
            continue
        md.append(f"### 1.{[t for t, _ in TARGETS].index(tk) + 1} {tname}")
        md.append("")
        rows = []
        for key in ORDER:
            if key not in data[tk]:
                continue
            st = stat_row(data[tk][key])
            st["_key"] = key
            rows.append(st)
        if not rows:
            continue
        tab = pd.DataFrame(rows)
        all_stats[tk] = tab

        md.append("| 方法 | n | Vina mean±std | Vina 中位数 | Vina 最优 | Vina≤-7 (%) | "
                  "BBB+ (%) | Pgp+ (%) | hERG+ (%) | DILI+ (%) | QED | MW | TPSA | LogP | SA | "
                  "PAINS | 唯一骨架 | 骨架熵 |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in tab.iterrows():
            name = DISPLAY.get(r["_key"], r["_key"])
            md.append(
                f"| {name} | {r['n']} | "
                f"{r['vina_mean'] if r['vina_mean'] is not None else '—'}±"
                f"{r['vina_std'] if r['vina_std'] is not None else '—'} | "
                f"{r['vina_median'] if r['vina_median'] is not None else '—'} | "
                f"{r['vina_best'] if r['vina_best'] is not None else '—'} | "
                f"{r['vina_hit_rate'] if r['vina_hit_rate'] is not None else '—'} | "
                f"{r['bbb_pct'] if r['bbb_pct'] is not None else '—'} | "
                f"{r['pgp_pct'] if r['pgp_pct'] is not None else '—'} | "
                f"{r['herg_pct'] if r['herg_pct'] is not None else '—'} | "
                f"{r['dili_pct'] if r['dili_pct'] is not None else '—'} | "
                f"{r['qed'] if r['qed'] is not None else '—'} | "
                f"{r['mw'] if r['mw'] is not None else '—'} | "
                f"{r['tpsa'] if r['tpsa'] is not None else '—'} | "
                f"{r['logp'] if r['logp'] is not None else '—'} | "
                f"{r['sa'] if r['sa'] is not None else '—'} | "
                f"{r['pains'] if r['pains'] is not None else '—'} | "
                f"{r['n_scaffolds'] if r['n_scaffolds'] is not None else '—'} | "
                f"{r['scaffold_entropy'] if r['scaffold_entropy'] is not None else '—'} |"
            )
        md.append("")

    # ---------- 通用 LLM 基线专章 ----------
    md.append("## 1.4 通用 LLM 基线的生成分子合格率")
    md.append("")
    md.append("通用 LLM（未做化学领域微调）在本任务上的表现需要单独说明："
              "它们能产出语法合法的 SMILES，但相当一部分是**非类药结构**"
              "（超长烷烃链、超高 MW、超多可旋转键），这类分子既无法进入"
              "对接阶段，也不具备成药价值。")
    md.append("")
    md.append("| 靶点 | 方法 | 生成数 | 通过类药预筛 (%) | 预筛失败数 | "
              "有效对接数 | Vina 中位数 |")
    md.append("|---|---|---|---|---|---|---|")
    for tk, tname in TARGETS:
        if tk not in data:
            continue
        for key in ("base_qwen", "base_mistral"):
            if key not in data[tk]:
                continue
            st = stat_row(data[tk][key])
            name = DISPLAY.get(key, key)
            md.append(
                f"| {tname} | {name} | {st['n']} | "
                f"{st['prefilter_pass_rate']} | {st['prefilter_fail']} | "
                f"{st['vina_n']} | "
                f"{st['vina_median'] if st['vina_median'] is not None else '—'} |"
            )
    md.append("")

    # ---------- 多目标综合排序 ----------
    md.append("## 1.5 多目标综合排序（结合能力 × 类药性 × 多样性）")
    md.append("")
    md.append("单一 Vina 指标不足以评价一个生成分子模型。真实 CNS 药物需要同时"
              "满足：与靶点口袋结合、具备类药物化性质（BBB 通透、低 Pgp 底物、"
              "低 hERG/DILI 风险）、且化学空间不坍缩。下表把三个维度合成一个"
              "0–1 的综合得分（每维度先做 min-max 归一化后等权平均）。")
    md.append("")
    md.append("**综合得分定义**：")
    md.append("")
    md.append("```")
    md.append("Affinity  = 归一化(|Vina 均值|)           # 结合越强越高")
    md.append("Druglike  = (BBB+ + (1-Pgp%) + (1-hERG%) + (1-DILI%)) / 4")
    md.append("            # CNS 药物需要能过血脑屏障、且无外排/心脏/肝毒性风险")
    md.append("Diversity = 归一化(骨架熵)                    # 骨架熵越高越不坍缩")
    md.append("Composite = (Affinity + Druglike + Diversity) / 3")
    md.append("```")
    md.append("")
    md.append("| 靶点 | 排名 | 方法 | Affinity | Druglike | Diversity | "
              "**Composite** |")
    md.append("|---|---|---|---|---|---|---|")
    for tk, tname in TARGETS:
        if tk not in data:
            continue
        recs = []
        for key in ORDER:
            if key not in data[tk]:
                continue
            st = stat_row(data[tk][key])
            vm = st["vina_mean"]
            recs.append({
                "key": key,
                "name": DISPLAY.get(key, key),
                "aff": (-vm) if vm is not None else None,
                "bbb": st["bbb_pct"],
                "pgp": st["pgp_pct"],
                "herg": st["herg_pct"],
                "dili": st["dili_pct"],
                "div": st["scaffold_entropy"],
            })
        if not recs:
            continue

        # Druglike 分数：BBB 通透是加分项，Pgp/hERG/DILI 风险是减分项
        for r in recs:
            parts = []
            if r["bbb"] is not None:
                parts.append(r["bbb"] / 100.0)
            for k in ("pgp", "herg", "dili"):
                if r[k] is not None:
                    parts.append(1.0 - r[k] / 100.0)
            r["drug"] = sum(parts) / len(parts) if parts else None

        affs = [r["aff"] for r in recs if r["aff"] is not None]
        divs = [r["div"] for r in recs if r["div"] is not None]
        a_lo, a_hi = (min(affs), max(affs)) if affs else (0, 1)
        d_lo, d_hi = (min(divs), max(divs)) if divs else (0, 1)
        for r in recs:
            r["aff_n"] = ((r["aff"] - a_lo) / (a_hi - a_lo)) if (
                r["aff"] is not None and a_hi - a_lo > 1e-9) else None
            r["div_n"] = ((r["div"] - d_lo) / (d_hi - d_lo)) if (
                r["div"] is not None and d_hi - d_lo > 1e-9) else None
            parts = [r["aff_n"], r["drug"], r["div_n"]]
            parts = [p for p in parts if p is not None]
            r["comp"] = round(sum(parts) / len(parts), 3) if parts else None
        recs.sort(key=lambda r: (r["comp"] is None, -(r["comp"] or 0)))
        for i, r in enumerate(recs):
            md.append(
                f"| {tname} | {i+1} | {r['name']} | "
                f"{round(r['aff_n'], 3) if r['aff_n'] is not None else '—'} | "
                f"{round(r['drug'], 3) if r['drug'] is not None else '—'} | "
                f"{round(r['div_n'], 3) if r['div_n'] is not None else '—'} | "
                f"**{r['comp'] if r['comp'] is not None else '—'}** |"
            )
        md.append("")

    # ---------- 统计检验 ----------
    md.append("## 2. 统计显著性检验")
    md.append("")
    md.append("对 Vina 结合分数做 Mann-Whitney U 检验（双侧），"
              "比较我们的方法与各基线。**分数越负越好**，因此 `p` 值显著即表示"
              "我们的分布整体更靠左（结合更强）。")
    md.append("")
    md.append("| 靶点 | 对比 | 中位数差 | p 值 | 结论 |")
    md.append("|---|---|---|---|---|")
    for tk, tname in TARGETS:
        if tk not in data or "comparison_final" not in data[tk]:
            continue
        ours = pd.to_numeric(data[tk]["comparison_final"]["vina_docking_score"],
                             errors="coerce").dropna()
        for key in ORDER:
            if key == "comparison_final" or key not in data[tk]:
                continue
            other = pd.to_numeric(data[tk][key]["vina_docking_score"],
                                  errors="coerce").dropna()
            if len(ours) == 0 or len(other) == 0:
                continue
            diff = round(ours.median() - other.median(), 2)
            p = mwu_test(ours, other)
            name = DISPLAY.get(key, key)
            better = "我们更优" if diff < 0 else "基线更优"
            md.append(f"| {tname} | {name} vs Ours | {diff:+} | {p} | {better} |")
    md.append("")

    # ---------- 管线验证 ----------
    md.append("## 3. 对接管线验证（共晶配体重对接）")
    md.append("")
    sanity_path = GBM_ROOT / "experiments" / "receptor_sanity_check.json"
    if sanity_path.exists():
        sj = json.loads(sanity_path.read_text(encoding="utf-8"))
        md.append("重对接共晶配体，验证对接管线能复现已知的结合模式"
                  "（若管线错误，共晶配体分数会显著劣于生成分子）。")
        md.append("")
        md.append("| 靶点 | PDB | 共晶配体 | 重对接分数 (kcal/mol) |")
        md.append("|---|---|---|---|")
        for k, v in sj.items():
            md.append(f"| {k} | {v.get('pdb_id', '—')} | {v.get('ligand', '—')} | "
                      f"{v.get('crystal_redock_score', v.get('score', '—'))} |")
    else:
        md.append("（未找到 sanity check 结果，请运行 scripts/prepare_receptors.py）")
    md.append("")

    # ---------- 多样性面板 ----------
    div_path = GBM_ROOT / "experiments" / "diversity_analysis.json"
    if div_path.exists():
        dj = json.loads(div_path.read_text(encoding="utf-8"))
        md.append("## 4. 化学空间多样性")
        md.append("")
        md.append("Tanimoto 相似度基于 Morgan 指纹 (r=2, 2048 bit) 的**两两**计算。"
                  "`IntDiv = 1 - 平均两两 Tanimoto`，越高表示内部覆盖越广。"
                  "「近重复%」指 Tanimoto > 0.9 的分子对占比 —— 衡量「换皮不换骨」的程度。")
        md.append("")
        for tk_d, tname_d in [("EGFR", "EGFR (2JIV)"),
                              ("VEGFR", "VEGF/VEGFR (3WZD)"),
                              ("IDH1", "IDH1/IDH2 (6U4J)")]:
            if tk_d not in dj:
                continue
            md.append(f"### {tname_d}")
            md.append("")
            md.append("| 方法 | n | 唯一 SMILES | 唯一骨架 | Tanimoto 均值 | 最大 | "
                      "近重复 (%) | IntDiv |")
            md.append("|---|---|---|---|---|---|---|---|")
            for key in ORDER:
                if key not in dj[tk_d]:
                    continue
                r = dj[tk_d][key]
                if not r:
                    continue
                name = DISPLAY.get(key, key)
                md.append(f"| {name} | {r['n_input']} | {r['n_unique_smiles']} | "
                          f"{r['n_unique_scaffolds']} | {r['tanimoto_mean']} | "
                          f"{r['tanimoto_max']} | {r['near_duplicate_pct']} | "
                          f"{r['intdiv']} |")
            md.append("")

    # ---------- 数据审计 ----------
    audit_path = GBM_ROOT / "experiments" / "data_audit" / "training_data_audit.json"
    if audit_path.exists():
        aj = json.loads(audit_path.read_text(encoding="utf-8"))
        md.append("## 5. 训练数据多样性审计")
        md.append("")
        md.append("| 指标 | 数值 |")
        md.append("|---|---|")
        md.append(f"| 总样本数 | {aj['n_samples']} |")
        md.append(f"| 唯一分子数 | {aj['n_unique_smiles']} |")
        md.append(f"| 唯一 instruction 数 | {aj['n_unique_instructions']} |")
        md.append(f"| 唯一 Murcko 骨架数 | {aj.get('n_unique_scaffolds')} |")
        if aj.get("n_unique_source_drugs"):
            md.append(f"| metadata 源药物数 | {aj['n_unique_source_drugs']} |")
        md.append("")

    # ---------- 保存 ----------
    md_path = out_dir / "comparison_report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    for tk, tab in all_stats.items():
        tab.to_csv(out_dir / f"stats_{tk}.csv", index=False)

    print("\n".join(md))
    print(f"\n-> {md_path}")


if __name__ == "__main__":
    main()
