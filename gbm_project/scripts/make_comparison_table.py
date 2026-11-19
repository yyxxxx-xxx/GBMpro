#!/usr/bin/env python3
"""
对比实验汇总表
==============

把重打分后的各方法结果汇总成论文用的主表 + 补充表。

主表（每个靶点一张）:
    方法 | n | Vina (mean±std) | BBB+ (%) | Pgp- (%) | hERG- (%) | DILI- (%) |
          QED | MW | TPSA | LogP | SA | PAINS+ | 骨架熵 | 唯一骨架

统计口径（写入论文 Method）:
    Vina 越负越好（kcal/mol，AutoDock Vina 1.2.7, exhaustiveness=8, n_poses=5）
    BBB+/Pgp-/hERG-/DILI- 为 admet-ai 预测概率 >0.5 的比例
    骨架熵为归一化 Shannon 熵，越高越分散

用法:
    python scripts/make_comparison_table.py --rescored_dir final_results/comparative/rescored
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GBM_ROOT = PROJECT_ROOT / "gbm_project"

# 靶点 -> (展示名, 展示顺序)
TARGET_ORDER = [
    ("EGFR", "EGFR"),
    ("VEGFR", "VEGF/VEGFR"),
    ("VEGF_VEGFR", "VEGF/VEGFR"),
    ("IDH1", "IDH1/IDH2"),
    ("IDH1_IDH2", "IDH1/IDH2"),
]

METHOD_ORDER = [
    "PlayMolecule AI",
    "LlaSMol",
    "Llamole (ablation)",
    "REINVENT4 (Vina-RL)",
    "REINVENT4 (TL+RL)",
    "Mistral-7B (Base)",
    "Qwen2-7B (Base)",
    "GBM-PharmaMind (Ours)",
]

METHOD_DISPLAY = {
    "PlayMolecule_AI": "PlayMolecule AI",
    "PlayMolecule AI": "PlayMolecule AI",
    "llasmol": "LlaSMol",
    "llamole": "Llamole (ablation)",
    "comparison_final": "GBM-PharmaMind (Ours)",
    "REINVENT": "REINVENT4 (Vina-RL)",
    "REINVENT4_TLRL": "REINVENT4 (TL+RL)",
    "base_mistral": "Mistral-7B (Base)",
    "base_qwen": "Qwen2-7B (Base)",
}


def _yes_rate(df: pd.DataFrame, col: str) -> Optional[float]:
    if col not in df.columns:
        return None
    vals = df[col].astype(str).str.strip().str.lower()
    valid = vals.isin(["yes", "no"])
    if not valid.any():
        return None
    return round((vals[valid] == "yes").sum() / valid.sum() * 100, 1)


def summarize(df: pd.DataFrame) -> Dict:
    """对单个方法-靶点的 CSV 计算汇总统计。"""
    out: Dict = {"n": len(df)}

    if "vina_docking_score" in df.columns:
        v = pd.to_numeric(df["vina_docking_score"], errors="coerce").dropna()
        if len(v):
            out["vina_mean"] = round(v.mean(), 2)
            out["vina_std"] = round(v.std(), 2)
            out["vina_median"] = round(v.median(), 2)
            out["vina_best"] = round(v.min(), 2)
            out["vina_n"] = len(v)
        else:
            out["vina_mean"] = out["vina_std"] = out["vina_median"] = None
            out["vina_best"] = None
            out["vina_n"] = 0

    for col, key in [
        ("BBB permeant", "bbb_pct"),
        ("Pgp substrate", "pgp_pct"),
        ("hERG risk", "herg_pct"),
        ("DILI risk", "dili_pct"),
        ("AMES risk", "ames_pct"),
    ]:
        out[key] = _yes_rate(df, col)

    # 概率均值（比 yes/no 更有信息量）
    for col, key in [
        ("BBB_prob", "bbb_prob"),
        ("Pgp_prob", "pgp_prob"),
        ("hERG_prob", "herg_prob"),
        ("DILI_prob", "dili_prob"),
        ("AMES_prob", "ames_prob"),
    ]:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            out[key] = round(v.mean(), 3) if len(v) else None
        else:
            out[key] = None

    for col, key in [
        ("QED", "qed"),
        ("MW", "mw"),
        ("TPSA", "tpsa"),
        ("LogP", "logp"),
        ("Synthetic Accessibility", "sa"),
    ]:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            out[key] = round(v.mean(), 2) if len(v) else None
        else:
            out[key] = None

    for col, key in [("PAINS #alerts", "pains"), ("Brenk #alerts", "brenk")]:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            out[key] = round(v.mean(), 2) if len(v) else None
        else:
            out[key] = None

    # 骨架多样性
    if "Scaffold" in df.columns:
        scaffolds = df["Scaffold"].dropna().astype(str)
        out["unique_scaffolds"] = int(scaffolds.nunique())
        out["scaffold_entropy"] = (
            round(float(pd.to_numeric(df["ScaffoldEntropy"], errors="coerce").dropna().iloc[0]), 3)
            if "ScaffoldEntropy" in df.columns
            and pd.to_numeric(df["ScaffoldEntropy"], errors="coerce").dropna().size
            else None
        )
    else:
        out["unique_scaffolds"] = None
        out["scaffold_entropy"] = None

    if "MaxTanimoto_ChEMBL" in df.columns:
        v = pd.to_numeric(df["MaxTanimoto_ChEMBL"], errors="coerce").dropna()
        out["max_tanimoto_chembl"] = round(v.mean(), 3) if len(v) else None
    else:
        out["max_tanimoto_chembl"] = None

    return out


def fmt(v, nd=2, pct=False, dash="—"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return dash
    if pct:
        return f"{v:.1f}"
    return f"{v:.{nd}f}"


def build_table(rescored_dir: Path) -> Dict[str, pd.DataFrame]:
    """扫描重打分目录，构建 每个靶点 -> (方法, 统计) 的表。"""
    data: Dict[str, Dict[str, Dict]] = {}

    for method_dir in sorted(rescored_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        raw_name = method_dir.name
        method = METHOD_DISPLAY.get(raw_name, raw_name.replace("_", " "))
        for csv_path in sorted(method_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            if "smiles" not in df.columns or len(df) == 0:
                continue
            # 靶点判定（顺序重要: VEGF_VEGFR 必须先于 EGFR，
            # 否则会被 "EGFR" 子串误判成 EGFR 靶点）
            target = None
            stem = csv_path.stem.upper()
            if "IDH" in stem:
                target = "IDH1/IDH2"
            elif "VEGF" in stem:
                target = "VEGF/VEGFR"
            elif "EGFR" in stem:
                target = "EGFR"
            if target is None:
                continue
            data.setdefault(target, {})[method] = summarize(df)

    tables = {}
    for target, methods in data.items():
        rows = []
        for method in METHOD_ORDER:
            if method not in methods:
                continue
            rows.append({"方法": method, **methods[method]})
        # 未在 METHOD_ORDER 里的也带上
        for method, s in methods.items():
            if method not in METHOD_ORDER:
                rows.append({"方法": method, **s})
        tables[target] = pd.DataFrame(rows)
    return tables


def render(table: pd.DataFrame) -> str:
    """渲染成 Markdown 表（论文可直接用）。"""
    if table.empty:
        return "（无数据）"

    cols = [
        ("方法", "方法", None),
        ("n", "n", None),
        ("vina_mean", "Vina (kcal/mol)", None),
        ("vina_std", "±", None),
        ("vina_median", "Vina 中位数", None),
        ("vina_best", "Vina 最优", None),
        ("bbb_pct", "BBB+ (%)", True),
        ("pgp_pct", "Pgp+ (%)", True),
        ("herg_pct", "hERG+ (%)", True),
        ("dili_pct", "DILI+ (%)", True),
        ("qed", "QED", None),
        ("mw", "MW", None),
        ("tpsa", "TPSA", None),
        ("logp", "LogP", None),
        ("sa", "SA", None),
        ("pains", "PAINS", None),
        ("unique_scaffolds", "唯一骨架", None),
        ("scaffold_entropy", "骨架熵", None),
    ]
    header = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]

    for _, r in table.iterrows():
        cells = []
        for key, _, pct in cols:
            if key == "方法":
                cells.append(str(r.get(key, "—")))
            elif key == "n":
                cells.append(str(int(r.get(key, 0))))
            elif key == "vina_std":
                cells.append(fmt(r.get(key)))
            elif key == "vina_mean":
                cells.append(fmt(r.get(key)))
            else:
                cells.append(fmt(r.get(key), 2 if key not in ("qed", "scaffold_entropy") else 3, pct))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Build comparison tables")
    parser.add_argument("--rescored_dir", type=str,
                        default=str(GBM_ROOT / "final_results" / "comparative" / "rescored"))
    parser.add_argument("--output_dir", type=str,
                        default=str(GBM_ROOT / "final_results" / "comparative" / "tables"))
    args = parser.parse_args()

    rescored = Path(args.rescored_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = build_table(rescored)
    if not tables:
        print(f"⚠ {rescored} 下没有可用数据")
        return

    all_md = ["# 对比实验汇总表", "",
              f"数据来源: `{rescored}`", "",
              "> 口径: AutoDock Vina 1.2.7 (exhaustiveness=8, n_poses=5) 真实对接；",
              "> ADMET 为 admet-ai 预测 (概率阈值 0.5)；描述符为 RDKit；SA 为 Ertl SA_Score。",
              "> 所有方法均由同一版本管线统一重算。", ""]

    for target in ["EGFR", "VEGF/VEGFR", "IDH1/IDH2"]:
        if target not in tables:
            continue
        table = tables[target]
        all_md.append(f"## {target}")
        all_md.append("")
        all_md.append(render(table))
        all_md.append("")
        table.to_csv(out_dir / f"{target.replace('/', '_')}.csv", index=False)

    md_path = out_dir / "comparison_tables.md"
    md_path.write_text("\n".join(all_md), encoding="utf-8")

    # 打印到终端
    print("=" * 100)
    for target in ["EGFR", "VEGF/VEGFR", "IDH1/IDH2"]:
        if target in tables:
            print(f"\n### {target}")
            print(render(tables[target]))
    print("\n" + "=" * 100)
    print(f"-> {md_path}")


if __name__ == "__main__":
    main()
