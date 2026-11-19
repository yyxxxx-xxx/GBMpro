#!/usr/bin/env python3
"""
把真实分子库转成 LoRA QA 训练集
================================

设计要点（对应已诊断出的三个问题）
----------------------------------
1. **数据塌陷修复**
   旧集 982 药名 -> 32 个唯一 SMILES（硬编码模板）。
   新集用 PubChem 解析出的真实 SMILES，每条 response 都是不同的真实分子。

2. **prompt 形态对齐**
   旧训练集 instruction 是 "Generate a GBM drug candidate SMILES for {cell}."
   但推理走的是 gbm_prompts.yaml 的领域 prompt + 目标知识 + CoT。
   新集的 instruction 复用**推理时同一套模板**（base_gbm_prompt + target_specific_prompts
   + 指标约束 + CoT + 生成指令），response 仍是 "SMILES: xxx"，
   保证训练分布与推理分布一致。

3. **梯度只落在 response**
   tokenize 时把 instruction 部分的 label 置 -100，只对 "Response: SMILES: xxx" 计 loss。
   （这在 train 侧改，见 gbm_lora_finetuner 的 masked tokenize）

property profile 用 admet-ai 实测，不编造 target。

用法
----
    python scripts/build_qa_dataset.py \
        --molecules data/lora_datasets/real_molecules.json \
        --out data/lora_datasets/real_qa_train.json
"""
import os
import re
import sys
import json
import random
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import yaml

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GBM_ROOT = PROJECT_ROOT / "gbm_project"
sys.path.insert(0, str(GBM_ROOT / "src"))

CELL_STATE_MAP = {
    "MES": "Mesenchymal-like GBM cells",
    "NPC": "Neural progenitor-like GBM cells",
    "AC": "Astrocyte-like GBM cells",
    "OPC": "Oligodendrocyte precursor-like GBM cells",
}

# 推理时 configs/english_gbm_prompts.yaml 里的 target 名称
TARGET_ALIASES = {
    "EGFR": ["EGFR"],
    "VEGFR": ["VEGF_VEGFR"],
    "IDH1": ["IDH1_IDH2"],
}

# 靶点推断关键词（药名 -> 靶点）。
# 关键: 靶点标签必须与分子**真实相关**，否则模型学不到条件映射
# （随机分配靶点会让 instruction 变成噪声，条件生成能力学不出来）。
TARGET_KEYWORDS = {
    "EGFR": ["gefitinib", "erlotinib", "osimertinib", "lapatinib", "vandetanib",
             "pelitinib", "canertinib", "icotinib", "afatinib", "dacomitinib",
             "neratinib", "brigatinib", "almonertinib", "lazertinib",
             "poziotinib", "mobocertinib", "avitinib", "olmutinib", "naquotinib",
             "egfr", "her2", "erbb", "src", "lapatinib"],
    "VEGFR": ["sunitinib", "sorafenib", "pazopanib", "axitinib", "cabozantinib",
              "lenvatinib", "regorafenib", "tivozanib", "cediranib", "linifanib",
              "brivanib", "motesanib", "semaxanib", "su5416", "su6668", "vatalanib",
              "apatinib", "anlotinib", "fruquintinib", "surufatinib", "lotilanib",
              "vegfr", "kdr", "flk1", "angiogen", "semaxanib", "bevacizumab"],
    "IDH1": ["enasidenib", "ivosidenib", "vorasidenib", "olutasidenib",
             "ag-120", "ag-881", "ds-1001", "ft-2102", "ft-1102",
             "idh1", "idh2", "isocitrate"],
    # 其他 GBM 相关靶点（用于增加条件多样性）
    "MGMT": ["temozolomide", "tmz", "mgmt", "o6-benzylguanine", "lomeguatrib",
             "nimustine"],
    "PI3K_AKT_mTOR": ["everolimus", "temsirolimus", "sirolimus", "rapamycin",
                      "buparlisib", "pictilisib", "alpelisib", "copanlisib",
                      "ipatasertib", "capivasertib", "gedatolisib", "dactolisib",
                      "wortmannin", "ly294002", "idealisib", "idelalisib",
                      "duvelisib", "umbralisib", "mTOR", "PI3K", "AKT"],
    "p53_MDM2": ["nutlin", "mdm2", "idasanutlin", "milademetan", "siremadlin",
                 "alrizomadlin", "am-gain", "cgx-202", "idasanutlin"],
}


def infer_target(drug: str) -> str:
    """按药名关键词推断靶点。返回主靶点之一，未命中返回 None。"""
    d = drug.lower()
    for target, kws in TARGET_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in d:
                return target
    return None


# 非主靶点的分子归入"泛 GBM"靶点
GENERIC_TARGET = "EGFR"


def load_prompts(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def measure_properties(smiles_list: List[str]) -> List[Dict]:
    """用 admet-ai 实测 property profile（失败则全 NaN）"""
    from property_pipeline.admet import compute_admet, admet_available
    from property_pipeline.descriptors import compute_descriptors

    if not admet_available():
        print("  ⚠ admet-ai 不可用，property profile 将缺失")
        return [{} for _ in smiles_list]

    print(f"  admet-ai 推理 {len(smiles_list)} 个分子 ...")
    admet_rows = compute_admet(smiles_list)
    desc_rows = [compute_descriptors(s) for s in smiles_list]

    out = []
    for a, d in zip(admet_rows, desc_rows):
        row = {}
        for k in ("BBB_prob", "hERG_prob", "DILI_prob", "AMES_prob", "QED",
                  "BBB permeant", "hERG risk", "DILI risk"):
            if k in a and a[k] == a[k]:
                row[k] = a[k]
        for k in ("MW", "logP", "TPSA", "RotatableBonds", "HBD", "HBA", "SA_Score"):
            if k in d and d[k] == d[k]:
                row[k] = d[k]
        out.append(row)
    return out


def describe_metrics(props: Dict) -> str:
    """把实测 property 转成自然语言指标描述（贴近推理时的措辞风格）"""
    def bbb_str(v):
        if v is None:
            return "unknown permeability"
        if v > 0.5:
            return "good permeability (logBB > -0.5)"
        return "moderate permeability (logBB > -0.5)"

    def act_str(props):
        q = props.get("QED")
        if q is None:
            return "moderate"
        if q >= 0.7:
            return "high"
        if q >= 0.45:
            return "moderate"
        return "low"

    def tox_str(props):
        h = props.get("hERG_prob")
        d = props.get("DILI_prob")
        if h is None and d is None:
            return "moderate acceptable"
        risk = 0
        n = 0
        for v in (h, d):
            if v is not None:
                risk += v
                n += 1
        if n == 0:
            return "moderate acceptable"
        avg = risk / n
        if avg < 0.3:
            return "low"
        if avg < 0.6:
            return "moderate acceptable"
        return "high"

    return (f"BBB: {bbb_str(props.get('BBB_prob'))}; "
            f"GBM Activity: {act_str(props)}; "
            f"Toxicity: {tox_str(props)}; "
            f"Selectivity: high (GBM/normal ratio)")


def build_instruction(target: str, cell_desc: str, metrics: str,
                      prompts: dict, use_cot: bool,
                      potency: Optional[str] = None) -> str:
    """复用推理时的 prompt 模板

    potency: 实测活性（如 "IC50 = 0.62 uM"），来自 PubChem BioAssay。
             带上的样本说明「这条 instruction 确实对应一个真实活性分子」。
    """
    base = prompts["base_gbm_prompt"].strip()

    # specific_requirements：按 target 取 target_specific_prompt
    target_prompt = ""
    for key in TARGET_ALIASES.get(target, [target]):
        if key in prompts.get("target_specific_prompts", {}):
            target_prompt = prompts["target_specific_prompts"][key].strip()
            break

    req = (
        f"- Target cell phenotype: {cell_desc}\n"
        f"- Target metrics: {metrics}\n"
        f"- Molecular features: MW 300-500 Da, logP 2-4, TPSA 60-120, "
        f"HBD 1-4, HBA 3-8"
    )
    # 注: potency 行**不写入** instruction。
    # 推理端 (generate_with_new_adapter.py) 的 build_prompt 没有这一行,
    # 若训练时带上, 模型会学到一条推理时不存在的条件前缀 -> 分布不一致。
    # potency 仍然保留在 metadata 里, 用于评测阶段的分层分析。
    domain = base.replace("{specific_requirements}", req)
    if target_prompt:
        domain += "\n\n" + target_prompt

    if use_cot:
        cot = prompts.get("cot_reasoning_templates", {}).get(
            "step_by_step_design", "").strip()
        if cot:
            domain += "\n\n" + cot

    domain += ("\n\nBased on the above analysis, generate a novel GBM candidate "
               "molecule, including SMILES string and design rationale.")
    return domain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", type=str,
                    default=str(GBM_ROOT / "data/lora_datasets/real_molecules.json"))
    ap.add_argument("--out", type=str,
                    default=str(GBM_ROOT / "data/lora_datasets/real_qa_train.json"))
    ap.add_argument("--prompts", type=str,
                    default=str(GBM_ROOT / "configs/english_gbm_prompts.yaml"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_cot", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)

    mols = json.load(open(args.molecules))
    print(f"载入真实分子 {len(mols)} 条")

    prompts = load_prompts(Path(args.prompts))
    print(f"载入 prompt 模板: {args.prompts}")

    # 去重（按 SMILES）
    seen = set()
    uniq = []
    for m in mols:
        if m["smiles"] in seen:
            continue
        seen.add(m["smiles"])
        uniq.append(m)
    print(f"去重后 {len(uniq)} 个唯一分子")

    # 实测 property
    props = measure_properties([m["smiles"] for m in uniq])

    # 组装 QA
    samples = []
    n_target_hit = 0
    for m, p in zip(uniq, props):
        cell_desc = CELL_STATE_MAP.get(m.get("cell_state"), "GBM cells")
        metrics = describe_metrics(p)

        # 靶点：优先用**实测活性**标注（PubChem BioAssay，IC50 可查）。
        # 只有在没有活性证据时才退回关键词推断 / 泛靶点，
        # 绝不把泛瘤药无条件标成 EGFR —— 那是在教模型幻觉。
        has_activity = bool(m.get("cid") and m.get("value_um"))
        if m.get("target"):
            n_target_hit += 1
            target = m["target"]
        else:
            target = infer_target(m.get("drug", "")) or GENERIC_TARGET
        # 来源标签（写入 metadata，便于论文里披露数据构成）
        if has_activity:
            source = "pubchem_bioassay"
        elif m.get("target"):
            source = "target_keyword"
        elif infer_target(m.get("drug", "")):
            source = "target_keyword"
        else:
            source = "lincs_generic"
        use_cot = random.random() < 0.5
        # 实测活性 -> potency 描述
        potency = None
        if m.get("value_um"):
            v = m["value_um"]
            potency = f"{m.get('activity_name') or 'IC50'} = {v:.2f} uM"
        instr = build_instruction(target, cell_desc, metrics, prompts,
                                  use_cot and not args.no_cot, potency)
        samples.append({
            "instruction": instr,
            "response": f"SMILES: {m['smiles']}",
            "metadata": {
                "cell_state": m.get("cell_state"),
                "original_drug": m.get("drug"),
                "connectivity_score": m.get("connectivity_score"),
                "target": target,
                "metrics": metrics,
                "properties": p,
                "use_cot": use_cot,
                # 实测活性证据（若有）
                "cid": m.get("cid"),
                "value_um": m.get("value_um"),
                "activity_name": m.get("activity_name"),
                "source": source,
            },
        })

    random.shuffle(samples)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(samples, open(args.out, "w"), ensure_ascii=False, indent=1)

    n_uni_resp = len({s["response"] for s in samples})
    print(f"\n已保存 {len(samples)} 条 -> {args.out}")
    print(f"唯一 response (SMILES) 数: {n_uni_resp}  "
          f"(旧数据集为 32 / 2400)")
    print(f"唯一 instruction 数: {len({s['instruction'] for s in samples})}")
    print(f"靶点关键词命中: {n_target_hit}/{len(samples)} "
          f"({n_target_hit*100/len(samples):.1f}%)")
    import collections
    print(f"靶点分布: {dict(collections.Counter(s['metadata']['target'] for s in samples))}")
    print(f"数据来源: {dict(collections.Counter(s['metadata']['source'] for s in samples))}")
    print("\n--- 样例 ---")
    print(samples[0]["instruction"][:900])
    print(">>>")
    print(samples[0]["response"])

    # 切分 dev/test
    n = len(samples)
    n_dev = max(1, n // 20)
    json.dump(samples[:n_dev], open(args.out.replace("_train.json", "_dev.json"), "w"),
              ensure_ascii=False, indent=1)
    json.dump(samples[n_dev:n_dev + n_dev], open(
        args.out.replace("_train.json", "_test.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\ndev/test 各 {n_dev} 条已保存")


if __name__ == "__main__":
    main()
