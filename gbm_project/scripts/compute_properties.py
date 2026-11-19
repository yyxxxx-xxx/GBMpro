#!/usr/bin/env python3
"""
性质计算主脚本（对比实验）
==========================

对任意 baseline 的 CSV 批量补齐全部本地可计算的分子性质列。

新增列（相对原有 comparative CSV）:
    RDKit:   HBD, HBA, RotB, NIH #alerts
    admet-ai: BBB_prob, Pgp_prob, hERG_prob, DILI_prob, AMES_prob, QED,
              Caco2, Solubility, BBB permeant, Pgp substrate, hERG risk, DILI risk
    Vina:    vina_docking_score (真实对接，替代原伪对接)
    骨架:    Scaffold, ScaffoldEntropy, FragmentEntropy, NovelScaffold

用法:
    python scripts/compute_properties.py --input_dir experiments/cmp_test --output_dir experiments/cmp_test/scored
    python scripts/compute_properties.py --input_file a.csv --output_file b.csv --target EGFR
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GBM_ROOT = PROJECT_ROOT / "gbm_project"
sys.path.insert(0, str(GBM_ROOT / "src"))
sys.path.insert(0, str(GBM_ROOT))

from property_pipeline.descriptors import compute_descriptors, sa_source, DESCRIPTOR_COLUMNS
from property_pipeline.admet import compute_admet, admet_available, ADMET_COLUMNS
from property_pipeline.docking import VinaDocking, vina_available, DOCKING_COLUMNS
from property_pipeline.scaffold import compute_scaffold_metrics, SCAFFOLD_COLUMNS


def load_receptor_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_target(target_raw: str, cfg: dict) -> Optional[str]:
    """把生成脚本的靶点名映射到 receptor config 的 key。"""
    aliases = cfg.get("target_aliases", {})
    if target_raw in aliases:
        return aliases[target_raw]
    if target_raw in cfg:
        return target_raw
    return None


def infer_target_from_filename(path: Path, cfg: dict) -> Optional[str]:
    stem = path.stem.upper()
    for key, spec in cfg.items():
        if not isinstance(spec, dict) or "pdb_id" not in spec:
            continue
        if key.upper() in stem:
            return key
        disp = str(spec.get("display_name", "")).upper()
        for part in disp.replace("/", "_").split("_"):
            if part and part in stem:
                return key
    return None


def build_dockers(cfg: dict, targets: List[str], log=print) -> Dict[str, VinaDocking]:
    """为每个靶点创建 Vina 对接器。"""
    dockers = {}
    for key in targets:
        spec = cfg.get(key)
        if not spec:
            continue
        manifest_path = GBM_ROOT / "data" / "receptors" / "receptors_manifest.json"
        pdbqt = None
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry = manifest.get(key, {})
            if entry.get("status") == "ok":
                pdbqt = entry.get("pdbqt")
        if not pdbqt:
            pdbqt = spec.get("pdbqt")
        if not pdbqt or not os.path.isfile(pdbqt):
            log(f"  [skip] {key}: 受体 PDBQT 缺失，先运行 scripts/prepare_receptors.py")
            continue
        dockers[key] = VinaDocking(
            receptor_pdbqt=pdbqt,
            center=tuple(spec["box_center"]),
            size=tuple(spec["box_size"]),
        )
        log(f"  [ok] {key}: {os.path.basename(pdbqt)}")
    return dockers


def process_file(
    csv_path: Path,
    out_path: Path,
    cfg: dict,
    target_key: Optional[str],
    do_vina: bool,
    do_admet: bool,
    log=print,
) -> Optional[pd.DataFrame]:
    """对单个 CSV 计算全部性质并写出。"""
    df = pd.read_csv(csv_path)
    if "smiles" not in df.columns:
        log(f"  [skip] {csv_path.name}: 无 smiles 列")
        return None

    smiles_list = [str(s) if pd.notna(s) else "" for s in df["smiles"]]
    n = len(smiles_list)
    log(f"  [{csv_path.name}] n={n}")

    # ---- 1. RDKit 描述符 ----
    log("    · RDKit 描述符 ...")
    desc_rows = [compute_descriptors(s) for s in smiles_list]
    for c in DESCRIPTOR_COLUMNS:
        df[c] = [r.get(c, float("nan")) for r in desc_rows]

    # ---- 2. admet-ai ----
    if do_admet and admet_available():
        log("    · admet-ai ADMET ...")
        admet_rows = compute_admet(smiles_list)
        for c in ADMET_COLUMNS:
            df[c] = [r.get(c, float("nan")) for r in admet_rows]
    elif do_admet:
        log("    ! admet-ai 不可用，跳过 ADMET 列")

    # ---- 3. Vina 真实对接 ----
    if do_vina and target_key:
        docker = DOCKER_REGISTRY.get(target_key)
        if docker is not None:
            log(f"    · Vina 对接 ({target_key}) —— 每个分子约 15-30 秒 ...")
            t0 = time.time()
            dock_rows = []
            for i, smi in enumerate(smiles_list):
                score, err = docker.dock(smi)
                row = {c: float("nan") for c in DOCKING_COLUMNS}
                if score is not None:
                    row["vina_docking_score"] = round(score, 3)
                    row["vina_affinity"] = round(score, 3)
                dock_rows.append(row)
                if (i + 1) % 5 == 0:
                    el = time.time() - t0
                    log(f"      {i + 1}/{n} (用时 {el:.0f}s)")
            for c in DOCKING_COLUMNS:
                df[c] = [r.get(c, float("nan")) for r in dock_rows]
            df["vina_error"] = ""
        else:
            log(f"    ! 无 {target_key} 受体，跳过 Vina")
    elif do_vina:
        log("    ! 未识别靶点，跳过 Vina")

    # ---- 4. 骨架 / 多样性 ----
    log("    · 骨架与多样性 ...")
    scaf_rows = compute_scaffold_metrics(smiles_list)
    for c in SCAFFOLD_COLUMNS:
        df[c] = [r.get(c, float("nan")) for r in scaf_rows]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log(f"    ✓ 已保存: {out_path.name}")
    return df


DOCKER_REGISTRY: Dict[str, VinaDocking] = {}


def main():
    parser = argparse.ArgumentParser(description="Compute all local molecular properties")
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--target", type=str, default=None, help="强制指定靶点")
    parser.add_argument("--config", type=str, default=str(GBM_ROOT / "configs" / "receptors.yaml"))
    parser.add_argument("--no_vina", action="store_true", help="跳过 Vina 对接")
    parser.add_argument("--no_admet", action="store_true", help="跳过 admet-ai")
    args = parser.parse_args()

    cfg = load_receptor_config(Path(args.config))
    print("=" * 70)
    print("性质计算管线")
    print("=" * 70)
    print(f"  SA score 来源: {sa_source()}")
    print(f"  vina 可用: {vina_available()}   admet-ai 可用: {admet_available()}")

    # 收集待处理文件
    files: List[Path] = []
    if args.input_file:
        files = [Path(args.input_file)]
    elif args.input_dir:
        files = sorted(p for p in Path(args.input_dir).glob("*.csv") if p.name != "generation_manifest.json")
    else:
        print("必须提供 --input_dir 或 --input_file")
        return

    # 靶点 -> 对接器
    global DOCKER_REGISTRY
    needed = set()
    for f in files:
        tk = args.target or resolve_target(f.stem, cfg) or infer_target_from_filename(f, cfg)
        if tk:
            needed.add(tk)
    print("\n🔧 准备对接器...")
    DOCKER_REGISTRY = build_dockers(cfg, sorted(needed))

    # 处理
    for f in files:
        target_key = args.target or resolve_target(f.stem, cfg) or infer_target_from_filename(f, cfg)
        print(f"\n▶ {f.name}  (target={target_key})")
        if args.output_file:
            out = Path(args.output_file)
        elif args.output_dir:
            out = Path(args.output_dir) / f.name
        else:
            out = f.with_name(f.stem + "_scored.csv")
        process_file(
            f, out, cfg, target_key,
            do_vina=not args.no_vina,
            do_admet=not args.no_admet,
        )

    print("\n完成。")


if __name__ == "__main__":
    main()
