#!/usr/bin/env python3
"""
化学空间多样性分析
==================

多维度的多样性评估（比单一骨架熵更可靠）:

    1. 唯一 SMILES 数           —— 简单去重
    2. 唯一 Murcko 骨架数       —— 骨架层面去重
    3. 两两 Tanimoto 相似度分布 —— 内部化学空间覆盖（Morgan FP, r=2, 2048 bit）
    4. 近重复率 (>0.9)          —— "换皮不换骨" 的程度
    5. 内部多样性 (IntDiv)      —— 1 - 平均两两 Tanimoto
    6. 对 ChEMBL 的新颖性        —— 与真实药物库的最大相似度

输出: experiments/diversity_analysis.json + 控制台对比表
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GBM_ROOT = PROJECT_ROOT / "gbm_project"
sys.path.insert(0, str(GBM_ROOT / "src"))

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp")

DISPLAY = {
    "PlayMolecule_AI": "PlayMolecule AI",
    "llasmol": "LlaSMol",
    "llamole": "Llamole (ablation)",
    "comparison_final": "GBM-PharmaMind (Ours)",
    "REINVENT": "REINVENT4 (Vina-RL)",
    "base_mistral": "Mistral-7B (Base)",
    "base_qwen": "Qwen2-7B (Base)",
}
ORDER = ["PlayMolecule_AI", "llasmol", "llamole", "REINVENT",
         "base_mistral", "base_qwen", "comparison_final"]


def analyze(df: pd.DataFrame) -> Dict:
    smiles = [str(s) for s in df["smiles"] if pd.notna(s)]
    mols = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mols.append(m)
    if not mols:
        return {}

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(m) for m in mols]

    sims = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sims.append(AllChem.DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    sims = np.array(sims) if sims else np.array([0.0])

    scaffolds = []
    for m in mols:
        try:
            sc = MurckoScaffold.GetScaffoldForMol(m)
            scaffolds.append(Chem.MolToSmiles(sc) if sc and sc.GetNumAtoms() else "")
        except Exception:
            scaffolds.append("")

    out = {
        "n_input": len(smiles),
        "n_valid": len(mols),
        "n_unique_smiles": len({Chem.MolToSmiles(m) for m in mols}),
        "n_unique_scaffolds": len({s for s in scaffolds if s}),
        "tanimoto_mean": round(float(sims.mean()), 3),
        "tanimoto_max": round(float(sims.max()), 3),
        "tanimoto_median": round(float(np.median(sims)), 3),
        "near_duplicate_pct": round(float((sims > 0.9).mean() * 100), 1),
        "intdiv": round(1 - float(sims.mean()), 3),
    }
    if "MaxTanimoto_ChEMBL" in df.columns:
        v = pd.to_numeric(df["MaxTanimoto_ChEMBL"], errors="coerce").dropna()
        out["max_tanimoto_chembl_mean"] = round(v.mean(), 3) if len(v) else None
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescored_dir", type=str,
                        default=str(GBM_ROOT / "final_results/comparative/rescored"))
    parser.add_argument("--output", type=str,
                        default=str(GBM_ROOT / "experiments/diversity_analysis.json"))
    args = parser.parse_args()

    rescored = Path(args.rescored_dir)
    results: Dict[str, Dict[str, Dict]] = {}

    for method_dir in sorted(rescored.iterdir()):
        if not method_dir.is_dir():
            continue
        for csv_path in sorted(method_dir.glob("*.csv")):
            stem = csv_path.stem.upper()
            if "IDH" in stem:
                tk = "IDH1"
            elif "VEGF" in stem:
                tk = "VEGFR"
            elif "EGFR" in stem:
                tk = "EGFR"
            else:
                continue
            df = pd.read_csv(csv_path)
            results.setdefault(tk, {})[method_dir.name] = analyze(df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 控制台对比表 ----
    for tk in ["EGFR", "VEGFR", "IDH1"]:
        if tk not in results:
            continue
        print(f"\n{'=' * 100}\n### {tk} 化学空间多样性\n{'=' * 100}")
        print(f"{'方法':<26}{'n':>5}{'唯一SMILES':>12}{'唯一骨架':>10}"
              f"{'Tanimoto均值':>14}{'最大':>8}{'近重复%':>10}{'IntDiv':>9}{'ChEMBL相似':>12}")
        print("-" * 100)
        for key in ORDER:
            if key not in results[tk]:
                continue
            r = results[tk][key]
            if not r:
                continue
            print(f"{DISPLAY.get(key, key):<26}{r['n_input']:>5}{r['n_unique_smiles']:>12}"
                  f"{r['n_unique_scaffolds']:>10}{r['tanimoto_mean']:>14}"
                  f"{r['tanimoto_max']:>8}{r['near_duplicate_pct']:>10}"
                  f"{r['intdiv']:>9}{str(r.get('max_tanimoto_chembl_mean', '—')):>12}")

    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
