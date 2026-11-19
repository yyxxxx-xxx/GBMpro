#!/usr/bin/env python3
"""
对接管线验证：共晶配体重对接
============================

论文必备的验证步骤 —— 证明对接分数不是随机噪声，而是真的复现了
已知的结合模式。

做法: 把 PDB 里的共晶配体重新对接回自己的口袋。
      若管线正确，重对接分数应与生成分子相当甚至更好（因为共晶配体
      是经过实验优化的真实结合构象）。

输出: experiments/receptor_sanity_check.json
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GBM_ROOT = PROJECT_ROOT / "gbm_project"
sys.path.insert(0, str(GBM_ROOT / "src"))
sys.path.insert(0, str(GBM_ROOT))

from property_pipeline.docking import VinaDocking
from scripts.compute_properties import load_receptor_config, build_dockers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=str(GBM_ROOT / "configs/receptors.yaml"))
    parser.add_argument("--output", type=str,
                        default=str(GBM_ROOT / "experiments/receptor_sanity_check.json"))
    args = parser.parse_args()

    cfg = load_receptor_config(Path(args.cfg))
    results = {}

    for key in ["EGFR", "IDH1_IDH2", "VEGF_VEGFR"]:
        spec = cfg.get(key)
        if not spec:
            continue
        lig = spec.get("crystal_ligand_smiles")
        if not lig:
            results[key] = {
                "pdb_id": spec.get("pdb_id"),
                "ligand": spec.get("crystal_ligand_name"),
                "error": "crystal_ligand_smiles 未配置",
            }
            print(f"  ! {key}: 未配置共晶配体 SMILES，跳过")
            continue

        print(f"\n▶ {key} ({spec.get('pdb_id')}, 共晶配体 {spec.get('crystal_ligand_name')})")
        dockers = build_dockers(cfg, [key])
        if key not in dockers:
            results[key] = {"pdb_id": spec.get("pdb_id"), "error": "受�� PDBQT 缺失"}
            continue

        score, err = dockers[key].dock(lig)
        print(f"    重对接分数: {score} kcal/mol  {err[:80]}")
        results[key] = {
            "pdb_id": spec.get("pdb_id"),
            "ligand": spec.get("crystal_ligand_name"),
            "crystal_ligand_smiles": lig,
            "crystal_redock_score": score,
            "error": err,
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
