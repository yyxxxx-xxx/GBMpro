#!/usr/bin/env python3
"""
按真实蛋白靶点抓取活性化合物（PubChem BioAssay）
==================================================

为什么需要这一步
--------------
LINCS 的 1679 个药名里，真正带靶点关键词的只有 19 个（EGFR 5 / VEGFR 6 / ...）。
如果按关键词兜底把 905 个泛瘤药都标成 "EGFR"，就是在教模型**幻觉** ——
instruction 说这是 EGFR 抑制剂，response 却是一个完全无关的分子。
条件生成能力会因此学崩。

正确做法是反过来：**从靶点出发找分子**。
PubChem 的 protein/accession/<UniProt>/concise 接口直接给出
该靶点的活性化合物（CID + IC50 + 活性类型），是**实测**的靶点-分子关系。

流程
----
1. 对每个靶点查 PubChem，拿到 (CID, SMILES, IC50, Activity Outcome)
2. 按 IC50 过滤（<= 10 uM 且 Outcome == Active）
3. 与 LINCS 分子库取交集（保留有 GBM 细胞状态标注的）
4. 交集为空时，保留 PubChem 活性分子（标注 target 但不带细胞状态，
   用 "GBM cells" 作为泛化 phenotype）

用法
----
    python scripts/fetch_target_active_molecules.py \
        --out data/lora_datasets/target_active_molecules.json
"""
import os
import re
import sys
import json
import time
import argparse
import warnings
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GBM_ROOT = PROJECT_ROOT / "gbm_project"

# 靶点 -> (UniProt accession, 显示名, 活性阈值 uM)
TARGETS = {
    "EGFR": ("P00533", "EGFR tyrosine kinase receptor", 10.0),
    "VEGFR": ("P15692", "VEGFR2/KDR tyrosine kinase receptor", 10.0),
    "IDH1": ("P06401", "isocitrate dehydrogenase IDH1 (R132H mutant)", 10.0),
    "MGMT": ("P16455", "O6-methylguanine-DNA methyltransferase", 10.0),
    "PI3K_AKT_mTOR": ("P42336", "PI3K catalytic subunit alpha", 10.0),
    "p53_MDM2": ("Q00987", "MDM2 proto-oncogene", 1.0),
}

CONCISE = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/protein/accession/{}/concise/JSON")
CID_PROPS = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/"
             "CanonicalSMILES,ConnectivitySMILES,MolecularWeight,XLogP,TPSA/JSON")


def fetch_json(url: str, timeout: int = 90, retries: int = 3) -> Optional[dict]:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "gbm-builder/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except Exception as e:
            if attempt == retries - 1:
                print(f"    ! 请求失败: {type(e).__name__} {str(e)[:100]}")
                return None
            time.sleep(2 + attempt * 2)
    return None


def parse_concise(data: dict) -> List[dict]:
    """解析 concise 表格"""
    table = data.get("Table", {})
    cols = table.get("Columns", {}).get("Column", [])
    rows = table.get("Row", [])
    out = []
    for r in rows:
        cells = r.get("Cell", [])
        d = dict(zip(cols, cells))
        out.append(d)
    return out


def get_active_compounds(acc: str, max_um: float) -> List[dict]:
    """取该靶点的活性化合物"""
    data = fetch_json(CONCISE.format(acc))
    if not data:
        return []
    rows = parse_concise(data)

    out = []
    for d in rows:
        outcome = (d.get("Activity Outcome") or "").strip()
        if outcome != "Active":
            continue
        val_str = (d.get("Activity Value [uM]") or "").strip()
        if not val_str:
            continue
        try:
            val = float(val_str)
        except ValueError:
            continue
        if val <= 0 or val > max_um:
            continue
        cid = d.get("CID", "").strip()
        if not cid:
            continue
        out.append({
            "cid": cid,
            "value_um": val,
            "activity_name": d.get("Activity Name", ""),
            "assay_name": d.get("Assay Name", ""),
            "pubmed": d.get("PubMed ID", ""),
        })
    # 按活性值排序，取最活跃的
    out.sort(key=lambda x: x["value_um"])
    return out


def cid_to_smiles(cid: str) -> Optional[str]:
    data = fetch_json(CID_PROPS.format(cid), timeout=45, retries=2)
    if not data:
        return None
    try:
        p = data["PropertyTable"]["Properties"][0]
        return p.get("ConnectivitySMILES") or p.get("CanonicalSMILES")
    except Exception:
        return None


def canonicalize(smiles: str) -> Optional[str]:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        n = mol.GetNumHeavyAtoms()
        if n < 12 or n > 70:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lincs", type=str,
                    default=str(GBM_ROOT / "data/lora_datasets/real_molecules.json"))
    ap.add_argument("--out", type=str,
                    default=str(GBM_ROOT / "data/lora_datasets/target_active_molecules.json"))
    ap.add_argument("--per_target", type=int, default=120)
    args = ap.parse_args()

    print("=" * 70)
    print("按真实靶点抓取活性化合物")
    print("=" * 70)

    # LINCS 库：drug -> (smiles, cell_states)
    lincs: Dict[str, dict] = {}
    if os.path.exists(args.lincs):
        for m in json.load(open(args.lincs)):
            key = m["drug"].lower()
            if key not in lincs:
                lincs[key] = {"smiles": m["smiles"], "cell_state": m["cell_state"],
                              "connectivity_score": m.get("connectivity_score")}
            else:
                # 同一药多个细胞状态时保留连接分数最高的
                if (m.get("connectivity_score") or 0) > \
                        (lincs[key].get("connectivity_score") or 0):
                    lincs[key]["cell_state"] = m["cell_state"]
                    lincs[key]["connectivity_score"] = m.get("connectivity_score")
    print(f"LINCS 库: {len(lincs)} 个药名")

    all_records = []
    for target, (acc, disp, max_um) in TARGETS.items():
        print(f"\n[{target}] UniProt {acc} ...")
        actives = get_active_compounds(acc, max_um)
        print(f"  活性化合物 (IC50 <= {max_um} uM): {len(actives)}")

        kept = []
        n_lincs_hit = 0
        for a in actives:
            if len(kept) >= args.per_target:
                break
            cid = a["cid"]
            smi = cid_to_smiles(cid)
            if not smi:
                continue
            smi = canonicalize(smi)
            if not smi:
                continue

            # 尝试在 LINCS 库里找到同药名 -> 带细胞状态标注
            cell_state = None
            conn = None
            drug_name = None
            for d, info in lincs.items():
                # 用 CID 无法直接匹配药名，改用分子级匹配：LINCS 里是否有相同 SMILES
                if info["smiles"] == smi:
                    drug_name = d
                    cell_state = info["cell_state"]
                    conn = info.get("connectivity_score")
                    n_lincs_hit += 1
                    break

            kept.append({
                "target": target,
                "target_display": disp,
                "cid": cid,
                "smiles": smi,
                "value_um": a["value_um"],
                "activity_name": a["activity_name"],
                "cell_state": cell_state,
                "connectivity_score": conn,
                "original_drug": drug_name,
                "in_lincs": cell_state is not None,
            })
        print(f"  保留 {len(kept)} 个（其中 {n_lincs_hit} 个与 LINCS 库结构重合）")
        all_records.extend(kept)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_records, open(args.out, "w"), ensure_ascii=False, indent=1)

    import collections
    print("\n" + "=" * 70)
    print(f"总计 {len(all_records)} 条 -> {args.out}")
    print(f"唯一 SMILES: {len({r['smiles'] for r in all_records})}")
    print(f"靶点分布: {dict(collections.Counter(r['target'] for r in all_records))}")
    print(f"与 LINCS 重合: {sum(1 for r in all_records if r['in_lincs'])}")


if __name__ == "__main__":
    main()
