#!/usr/bin/env python3
"""
受体制备脚本（真 Vina 前置步骤）
=================================
从 RCSB 下载 PDB -> 剥离 HETATM/水/杂原子 -> 选链 -> meeko 生成受体 PDBQT。

输出的 PDBQT 供 AutoDock Vina 真实对接使用，替代 gbm_physical_evaluator.py 中
原先基于 MW/LogP 经验公式的"伪对接"（_pseudo_dock）。

用法:
    python scripts/prepare_receptors.py
    python scripts/prepare_receptors.py --targets EGFR,IDH1_IDH2
"""

import os
import sys
import json
import argparse
import subprocess
import urllib.request
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GBM_ROOT = PROJECT_ROOT / "gbm_project"
sys.path.insert(0, str(GBM_ROOT))

DEFAULT_CONFIG = GBM_ROOT / "configs" / "receptors.yaml"
DEFAULT_OUTDIR = GBM_ROOT / "data" / "receptors"


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_pdb(pdb_id: str, outdir: Path) -> Path:
    """从 RCSB 下载 PDB 文件（带缓存）。"""
    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / f"{pdb_id}.pdb"
    if target.exists() and target.stat().st_size > 1000:
        print(f"  [cache] {target}")
        return target

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"  [download] {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        content = resp.read()
    target.write_bytes(content)
    print(f"  [saved] {target} ({len(content)} bytes)")
    return target


def strip_to_protein(pdb_path: Path, chain: str) -> Tuple[Path, int]:
    """
    剥离所有 HETATM（含共晶配体、水、缓冲液），只保留指定链的 ATOM 记录。

    共晶配体必须剥离，否则会占据活性位点导致对接盒子判断错误。
    """
    lines = pdb_path.read_text().splitlines()
    kept: List[str] = []
    for line in lines:
        if line.startswith(("HETATM", "HET ", "CONECT", "MODEL", "ENDMDL", "ANISOU")):
            continue
        if len(line) > 21 and line[21] != chain:
            continue
        if not line.startswith("ATOM"):
            continue
        kept.append(line)
    kept.append("END")

    out = pdb_path.with_name(f"{pdb_path.stem}_chain{chain}.pdb")
    out.write_text("\n".join(kept) + "\n")
    return out, len(kept)


def prepare_receptor_pdbqt(
    protein_pdb: Path,
    out_prefix: Path,
    center: List[float],
    size: List[float],
) -> Optional[Path]:
    """
    调用 meeko 的 mk_prepare_receptor 生成刚性受体 PDBQT。

    -a (--allow_bad_res): 允许删除无法匹配模板的残基（缺失侧链等）
    --default_altloc A: 固定选择 altloc A 变体
    """
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "meeko.cli.mk_prepare_receptor",
        "--read_pdb", str(protein_pdb),
        "-o", str(out_prefix),
        "-p",
        "--box_size", *[str(s) for s in size],
        "--box_center", *[str(c) for c in center],
        "-a",
        "--default_altloc", "A",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    pdbqt = out_prefix.with_suffix(".pdbqt")
    if not pdbqt.exists():
        print(f"  [FAIL] meeko 未生成 PDBQT\n{result.stdout[-1500:]}\n{result.stderr[-1500:]}")
        return None
    print(f"  [ok] {pdbqt}")
    return pdbqt


def verify_box_contains_receptor_atoms(pdbqt: Path, center: List[float], size: List[float]) -> Dict:
    """验证盒子中心附近确实有受体原子（避免盒子放错位置）。"""
    coords = []
    with open(pdbqt) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                coords.append((
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ))
    if not coords:
        return {"n_atoms": 0, "n_in_box": 0, "min_dist": None}

    import math
    half = [s / 2 for s in size]
    n_in = 0
    min_dist = float("inf")
    for c in coords:
        d = math.dist(c, center)
        min_dist = min(min_dist, d)
        if all(abs(c[i] - center[i]) <= half[i] for i in range(3)):
            n_in += 1
    return {"n_atoms": len(coords), "n_in_box": n_in, "min_dist": round(min_dist, 2)}


def main():
    parser = argparse.ArgumentParser(description="Prepare receptor PDBQT files for real Vina docking")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    parser.add_argument("--targets", type=str, default=None, help="逗号分隔的靶点 key")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    targets = (
        [t.strip() for t in args.targets.split(",")]
        if args.targets
        else [k for k in config if isinstance(config.get(k), dict) and "pdb_id" in config[k]]
    )

    manifest = {}
    for key in targets:
        spec = config[key]
        print(f"\n=== {key} ({spec['pdb_id']}) ===")

        pdb_path = download_pdb(spec["pdb_id"], outdir)
        protein_pdb, n_lines = strip_to_protein(pdb_path, spec.get("chain", "A"))
        print(f"  [strip] chain={spec.get('chain','A')} kept {n_lines} ATOM lines -> {protein_pdb.name}")

        prefix = outdir / f"{key}_{spec['pdb_id']}_chain{spec.get('chain','A')}"
        pdbqt = prepare_receptor_pdbqt(
            protein_pdb, prefix, spec["box_center"], spec["box_size"]
        )
        if pdbqt is None:
            manifest[key] = {"status": "failed"}
            continue

        check = verify_box_contains_receptor_atoms(pdbqt, spec["box_center"], spec["box_size"])
        print(f"  [verify] 受体原子 {check['n_atoms']}，盒内 {check['n_in_box']}，最近距离 {check['min_dist']} A")

        manifest[key] = {
            "status": "ok",
            "pdb_id": spec["pdb_id"],
            "chain": spec.get("chain", "A"),
            "pdbqt": str(pdbqt),
            "center": spec["box_center"],
            "size": spec["box_size"],
            "ligand_residue": spec.get("ligand_residue"),
            "crystal_ligand_smiles": spec.get("crystal_ligand_smiles"),
            "n_receptor_atoms": check["n_atoms"],
            "n_atoms_in_box": check["n_in_box"],
        }

    mf = outdir / "receptors_manifest.json"
    mf.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n清单已写入: {mf}")
    return manifest


if __name__ == "__main__":
    main()
