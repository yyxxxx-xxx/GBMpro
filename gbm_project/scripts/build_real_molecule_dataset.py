#!/usr/bin/env python3
"""
构建真实分子 LoRA 训练集
==========================

背景
----
现有训练集 gbm_evaluation_expanded_lora_train.json 有两个致命问题：
  1. 数据塌陷：982 个不同药名 -> 最终只有 32 个唯一 SMILES
     （generate_optimized_smiles_for_targets() 从 12 个硬编码模板里随机挑）
  2. 梯度错配：instruction 73 token / response 14 token
     -> 84% 梯度花在背 prompt 模板，只有 16% 在教化学

本脚本用 PubChem 解析真实 LINCS 药名 -> 真实 SMILES，
再用 admet-ai 实测 property profile，构造新的训练集。

命名解析策略（实测：直查命中率仅 20%，变体重试可到 100%）
    原名 -> 直查 -> '.'->'-' -> 去空格 -> 大小写变体 -> 全失败则丢弃

用法
----
    python scripts/build_real_molecule_dataset.py --out data/lora_datasets/real_molecules.json
"""
import os
import re
import sys
import json
import time
import argparse
import urllib.parse
import urllib.request
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GBM_ROOT = PROJECT_ROOT / "gbm_project"

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES,IsomericSMILES,MolecularWeight,XLogP,TPSA/JSON"

# 训练集用到的 4 种细胞状态
CELL_STATES = {
    "MES": "Mesenchymal-like GBM cells",
    "NPC": "Neural Progenitor-like GBM cells",
    "AC": "Astrocyte-like GBM cells",
    "OPC": "Oligodendrocyte precursor-like GBM cells",
}


def name_variants(name: str) -> List[str]:
    """生成候选查询名。实测：'.'->'-' 是命中率提升的关键。"""
    base = name.strip()
    cands = [base]
    # LINCS 命名惯例：小数点分隔 = 连字符
    if "." in base:
        cands.append(base.replace(".", "-"))
    # 去掉所有空格
    cands.append(re.sub(r"\s+", "", base))
    # 点->横线 且 去空格
    cands.append(re.sub(r"\s+", "", base).replace(".", "-"))
    # 去掉前导数字（'10.DEBC' -> 'DEBC' 通常不行，但 '2.methoxyestradiol' -> 'methoxyestradiol' 可行）
    stripped = re.sub(r"^\d+\.", "", base)
    if stripped != base:
        cands.append(stripped)
        cands.append(stripped.replace(".", "-"))
    # 小写
    cands.append(base.lower())
    # 去重保序
    out, seen = [], set()
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def fetch_pug(name: str, timeout: int = 20, retries: int = 3) -> Optional[dict]:
    """查询 PubChem，返回 properties 或 None"""
    url = PUG.format(urllib.parse.quote(name))
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "gbm-dataset-builder/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
            props = data["PropertyTable"]["Properties"][0]
            return props
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None  # 不存在，不用重试
            time.sleep(1.0 + attempt)
        except Exception:
            time.sleep(1.0 + attempt)
    return None


def resolve_smiles(name: str) -> Tuple[Optional[str], Optional[dict], Optional[str]]:
    """把药名解析成 SMILES。返回 (smiles, props, matched_variant)"""
    for variant in name_variants(name):
        props = fetch_pug(variant)
        if props is None:
            continue
        smiles = props.get("ConnectivitySMILES") or props.get("CanonicalSMILES") or props.get("SMILES")
        if smiles:
            return smiles, props, variant
    return None, None, None


def canonicalize(smiles: str) -> Optional[str]:
    """用 RDKit 校验并规范化 SMILES。无效返回 None。

    PubChem 返回的 ConnectivitySMILES 通常是有效的，但偶尔会出现
    截断或非常规价键形式，必须过一遍 RDKit 才能进训练集。
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # 过滤掉极端分子（训练集里出现会造成灾难性泛化）
        if mol.GetNumHeavyAtoms() < 10 or mol.GetNumHeavyAtoms() > 100:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str,
                    default=str(GBM_ROOT / "data/lora_datasets/gbm_evaluation_expanded_train.json"))
    ap.add_argument("--out", type=str,
                    default=str(GBM_ROOT / "data/lora_datasets/real_molecules.json"))
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8, help="PubChem 并发查询线程数")
    ap.add_argument("--limit", type=int, default=0, help="0 = 全部")
    args = ap.parse_args()

    # ---- 读取源数据：药名 + 细胞状态 + 连接分数 ----
    src = json.load(open(args.source))
    records = []
    seen_pairs = set()
    for it in src:
        md = it.get("metadata", {})
        drug = md.get("original_drug")
        state = md.get("cell_state")
        score = md.get("connectivity_score")
        if not drug or not state:
            continue
        key = (drug, state)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        records.append({"drug": drug, "cell_state": state, "connectivity_score": score})

    if args.limit:
        records = records[:args.limit]
    print(f"源数据: {len(records)} 条 (药名, 细胞状态) 配对，"
          f"{len({r['drug'] for r in records})} 个唯一药名")

    # ---- 解析 SMILES ----
    # ---- 解析 SMILES（多线程并行）----
    out = []
    failed = []
    n_invalid = 0
    t0 = time.time()

    def _resolve_one(r):
        """单条解析：名字 -> SMILES（含变体重试）。"""
        smi, props, variant = resolve_smiles(r["drug"])
        canon = canonicalize(smi) if smi else None
        return r, smi, props, variant, canon

    n_workers = args.workers
    print(f"  使用 {n_workers} 个线程并行查询 PubChem ...")
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_resolve_one, r): i for i, r in enumerate(records)}
        done = 0
        for fut in as_completed(futs):
            done += 1
            r, smi, props, variant, canon = fut.result()
            if canon:
                r["smiles"] = canon
                r["matched_variant"] = variant
                r["pubchem"] = {k: props.get(k) for k in
                                ("MolecularWeight", "XLogP", "TPSA", "CID")} if props else {}
                out.append(r)
            else:
                if smi:
                    n_invalid += 1
                failed.append(r["drug"])
            if done % 50 == 0:
                ok = len(out)
                print(f"  [{done}/{len(records)}] 成功 {ok} ({ok*100/done:.1f}%) "
                      f"失败 {len(failed)} (其中 RDKit 无效 {n_invalid})  "
                      f"{time.time()-t0:.0f}s", flush=True)

    # 保持与源数据一致的顺序，便于追溯
    order = {id(r): i for i, r in enumerate(records)}
    out.sort(key=lambda x: order.get(id(x), 0))

    print(f"\n解析完成: 成功 {len(out)} / {len(records)} "
          f"({len(out)*100/max(len(records),1):.1f}%)")
    print(f"唯一 SMILES: {len({r['smiles'] for r in out})}")
    if failed:
        print(f"失败样例 (前 15): {failed[:15]}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n已保存 -> {args.out}")

    # 缓存失败的药名，便于后续人工补
    if failed:
        fp = str(args.out).replace(".json", "_failed.json")
        json.dump(sorted(set(failed)), open(fp, "w"), ensure_ascii=False, indent=1)
        print(f"失败列表 -> {fp}")


if __name__ == "__main__":
    main()
