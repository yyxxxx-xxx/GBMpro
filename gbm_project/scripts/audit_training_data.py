#!/usr/bin/env python3
"""
训练数据多样性审计
==================

这个脚本回答一个审稿人一定会问的问题：
    "你的 LoRA 训练数据有多少个不同的分子？"

审计内容:
    1. 训练集的唯一分子数 vs 总样本数
    2. 唯一 instruction 数
    3. metadata 里原始药物数 vs 最终 SMILES 数（多样性在哪一步丢失）
    4. 重复度最高的分子
    5. 与生成结果的重叠度（模型是在背答案还是在泛化）

输出: JSON + Markdown 报告
"""

import os
import sys
import json
import argparse
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GBM_ROOT = PROJECT_ROOT / "gbm_project"
sys.path.insert(0, str(GBM_ROOT / "src"))

from property_pipeline.smiles_repair import repair_smiles


def extract_smiles_from_response(resp: str) -> str:
    resp = resp.strip()
    if resp.upper().startswith("SMILES:"):
        resp = resp[7:].strip()
    return resp.split("\n")[0].strip()


def audit_dataset(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = len(data)
    responses = [extract_smiles_from_response(r["response"]) for r in data]
    instructions = [r["instruction"] for r in data]

    resp_counter = Counter(responses)
    instr_counter = Counter(instructions)

    out = {
        "path": str(path),
        "n_samples": n,
        "n_unique_smiles": len(resp_counter),
        "n_unique_instructions": len(instr_counter),
        "smiles_redundancy": round(1 - len(resp_counter) / n, 4) if n else None,
        "top_smiles": resp_counter.most_common(15),
        "top_instructions": [
            (i[:120], c) for i, c in instr_counter.most_common(6)
        ],
    }

    # metadata 审计：原始药物 -> 最终 SMILES
    if "metadata" in data[0]:
        drugs = [r["metadata"].get("original_drug") for r in data]
        drug_counter = Counter([d for d in drugs if d])
        out["n_unique_source_drugs"] = len(drug_counter)
        out["drug_to_smiles_ratio"] = (
            round(len(resp_counter) / len(drug_counter), 4) if drug_counter else None
        )
        out["top_source_drugs"] = drug_counter.most_common(10)

    # 骨架层面
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        scaffolds = []
        for s in resp_counter:
            fixed, _ = repair_smiles(s)
            if not fixed:
                continue
            m = Chem.MolFromSmiles(fixed)
            if m is None:
                continue
            try:
                sc = MurckoScaffold.GetScaffoldForMol(m)
                scaffolds.append(Chem.MolToSmiles(sc) if sc and sc.GetNumAtoms() else "")
            except Exception:
                pass
        out["n_unique_scaffolds"] = len([s for s in scaffolds if s])
    except Exception:
        out["n_unique_scaffolds"] = None

    return out


def overlap_with_generated(gen_csv: Path, train_smiles: List[str]) -> Dict:
    """生成结果与训练集的重叠度 —— 过拟合的直接证据。"""
    import pandas as pd

    df = pd.read_csv(gen_csv)
    gen = set(df["smiles"].astype(str).str.strip())
    train = set(train_smiles)
    inter = gen & train
    return {
        "n_generated": len(gen),
        "n_overlap_with_train": len(inter),
        "overlap_pct": round(len(inter) / len(gen) * 100, 1) if gen else None,
        "examples": list(inter)[:10],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default=str(GBM_ROOT / "data/lora_datasets/gbm_evaluation_expanded_lora_train.json"))
    parser.add_argument("--generated_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str,
                        default=str(GBM_ROOT / "experiments" / "data_audit"))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("训练数据多样性审计")
    print("=" * 70)

    result = audit_dataset(Path(args.dataset))

    print(f"\n数据集: {result['path']}")
    print(f"  总样本数            : {result['n_samples']}")
    print(f"  唯一分子数          : {result['n_unique_smiles']}")
    print(f"  唯一 instruction 数 : {result['n_unique_instructions']}")
    print(f"  分子重复率          : {result['smiles_redundancy']:.1%}")
    if result.get("n_unique_source_drugs"):
        print(f"  metadata 源药物数   : {result['n_unique_source_drugs']}")
        print(f"  源药物→分子坍缩率   : {result['drug_to_smiles_ratio']:.4f}"
              f"  (即 {result['n_unique_source_drugs']} 个源药物只产出 "
              f"{result['n_unique_smiles']} 个不同分子)")
    if result.get("n_unique_scaffolds") is not None:
        print(f"  唯一 Murcko 骨架    : {result['n_unique_scaffolds']}")

    print("\n  出现最多的分子:")
    for smi, cnt in result["top_smiles"][:8]:
        print(f"    {cnt:5d} x {smi[:70]}")

    if args.generated_csv:
        data = json.load(open(args.dataset))
        train_smiles = [extract_smiles_from_response(r["response"]) for r in data]
        ov = overlap_with_generated(Path(args.generated_csv), train_smiles)
        result["generated_overlap"] = ov
        print(f"\n生成结果与训练集重叠:")
        print(f"  生成 {ov['n_generated']} 个，其中 {ov['n_overlap_with_train']} 个 "
              f"直接出现在训练集 ({ov['overlap_pct']}%)")
        for s in ov["examples"][:5]:
            print(f"    {s}")

    out_json = out_dir / "training_data_audit.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {out_json}")


if __name__ == "__main__":
    main()
