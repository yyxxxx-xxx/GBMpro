#!/usr/bin/env python3
"""
基线统一重打分（公平性关键）
=============================

为什么要重打分:
    现有 comparative CSV 里的 vina_docking_score 来自不同的测量途径 ——
    部分是外部网站查询，部分是项目里的伪对接 (_pseudo_dock)。
    不同方法用不同口径的分数做对比，是审稿人最容易攻击的点。

    本脚本把**所有** baseline 和我们的模型放到同一套本地管线下重算:
        RDKit 描述符 -> admet-ai -> AutoDock Vina -> 骨架多样性
    这样表格里的每一个数字都来自同一个版本、同一个参数、同一次运行。

用法:
    python scripts/rescore_baselines.py --output_dir final_results/comparative/rescored
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

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GBM_ROOT = PROJECT_ROOT / "gbm_project"
sys.path.insert(0, str(GBM_ROOT / "src"))
sys.path.insert(0, str(GBM_ROOT))

from property_pipeline.descriptors import compute_descriptors, sa_source, DESCRIPTOR_COLUMNS
from property_pipeline.admet import compute_admet, admet_available, ADMET_COLUMNS
from property_pipeline.docking import VinaDocking, vina_available, DOCKING_COLUMNS
from property_pipeline.scaffold import compute_scaffold_metrics, SCAFFOLD_COLUMNS
from scripts.compute_properties import (  # noqa: E402
    load_receptor_config,
    resolve_target,
    build_dockers,
    infer_target_from_filename,
)

# 基线目录 -> 方法名
BASELINE_DIRS = {
    "PlayMolecule AI": "PlayMolecule AI",
    "llasmol": "LlaSMol",
    "llamole": "Llamole (ablation)",
}


def process(df: pd.DataFrame, target_key: Optional[str], docker: Optional[VinaDocking],
            do_vina: bool, do_admet: bool, log=print) -> pd.DataFrame:
    smiles_list = [str(s) if pd.notna(s) else "" for s in df["smiles"]]
    n = len(smiles_list)

    log(f"    · RDKit 描述符 (n={n})")
    desc_rows = [compute_descriptors(s) for s in smiles_list]
    for c in DESCRIPTOR_COLUMNS:
        df[c] = [r.get(c, float("nan")) for r in desc_rows]

    if do_admet and admet_available():
        log("    · admet-ai ...")
        admet_rows = compute_admet(smiles_list)
        for c in ADMET_COLUMNS:
            df[c] = [r.get(c, float("nan")) for r in admet_rows]
    else:
        log("    ! 跳过 ADMET")

    if do_vina and docker is not None:
        log(f"    · Vina 真实对接 ({target_key}) ...")
        t0 = time.time()
        rows = []
        for i, smi in enumerate(smiles_list):
            score, err = docker.dock(smi)
            row = {c: float("nan") for c in DOCKING_COLUMNS}
            if score is not None:
                row["vina_docking_score"] = round(score, 3)
                row["vina_affinity"] = round(score, 3)
            rows.append(row)
            if (i + 1) % 10 == 0:
                log(f"      {i + 1}/{n} ({time.time() - t0:.0f}s)")
        for c in DOCKING_COLUMNS:
            df[c] = [r.get(c, float("nan")) for r in rows]
    else:
        log("    ! 跳过 Vina")

    scaf_rows = compute_scaffold_metrics(smiles_list)
    for c in SCAFFOLD_COLUMNS:
        df[c] = [r.get(c, float("nan")) for r in scaf_rows]

    return df


def main():
    parser = argparse.ArgumentParser(description="Rescore all baselines with one unified pipeline")
    parser.add_argument("--comparative_dir", type=str,
                        default=str(GBM_ROOT / "final_results" / "comparative"))
    parser.add_argument("--output_dir", type=str,
                        default=str(GBM_ROOT / "final_results" / "comparative" / "rescored"))
    parser.add_argument("--config", type=str, default=str(GBM_ROOT / "configs" / "receptors.yaml"))
    parser.add_argument("--no_vina", action="store_true")
    parser.add_argument("--no_admet", action="store_true")
    parser.add_argument("--extra", type=str, default=None,
                        help="额外的目录: 路径=方法名（用于加入我们自己的生成结果）")
    args = parser.parse_args()

    cfg = load_receptor_config(Path(args.config))
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("基线统一重打分")
    print("=" * 70)
    print(f"  SA score 来源: {sa_source()}")
    print(f"  vina: {vina_available()}   admet-ai: {admet_available()}")

    # 收集所有任务
    tasks: List[tuple] = []  # (方法名, csv路径, 靶点key)
    comp = Path(args.comparative_dir)
    for dirname, method in BASELINE_DIRS.items():
        d = comp / dirname
        if not d.exists():
            continue
        for csv_path in sorted(d.glob("*.csv")):
            tasks.append((method, csv_path, None))

    if args.extra:
        for item in args.extra.split(","):
            path, method = item.split("=", 1)
            d = Path(path)
            files = [d] if d.is_file() else sorted(d.glob("*.csv"))
            for csv_path in files:
                tasks.append((method, csv_path, None))

    # 解析靶点
    resolved = []
    needed_targets = set()
    for method, csv_path, forced in tasks:
        tk = forced or resolve_target(csv_path.stem, cfg) or infer_target_from_filename(csv_path, cfg)
        if tk:
            needed_targets.add(tk)
        resolved.append((method, csv_path, tk))

    print("\n🔧 准备对接器...")
    dockers = build_dockers(cfg, sorted(needed_targets))

    # 处理
    for method, csv_path, tk in resolved:
        print(f"\n▶ [{method}] {csv_path.parent.name}/{csv_path.name}  (target={tk})")
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [FAIL] 读取失败: {e}")
            continue
        if "smiles" not in df.columns:
            print("  [skip] 无 smiles 列")
            continue

        # 保留原有的 drugclip_score（外部指标），重命名清楚
        if "drugclip_score" in df.columns:
            df["drugclip_score_external"] = df["drugclip_score"]
        if "vina_docking_score" in df.columns:
            df["vina_docking_score_legacy"] = df["vina_docking_score"]

        df = process(
            df, tk, dockers.get(tk),
            do_vina=not args.no_vina, do_admet=not args.no_admet,
        )

        out_dir = out_root / method.replace(" ", "_").replace("/", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / csv_path.name
        df.to_csv(out_path, index=False)
        print(f"  ✓ {out_path}")

    print("\n完成。")


if __name__ == "__main__":
    main()
