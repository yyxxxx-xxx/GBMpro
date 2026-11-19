"""
按靶点构建 GBMPhysicalEvaluator（真实对接版）
==============================================

背景 (2026-11-18):
    反馈闭环此前用的是"伪对接"（基于 MW/LogP 的经验公式），
    导致 ToT + 反馈管线（Vina -5.1~-8.3）反而劣于不用 ToT 的简单采样
    (-8.85/-7.66/-9.01) —— 反馈闭环收益为负。

    本模块把真实 Vina 对接接入反馈闭环，并复用
    property_pipeline.docking.VinaDocking（vina Python API 1.2.7 + Meeko），
    关键优化: 复用已计算的 grid maps（maps 计算占单次对接 80%+ 时间）。

用法:
    from gbm_target_evaluator import build_evaluator_for_target
    ev = build_evaluator_for_target("EGFR")
    result = ev.evaluate("CCO")
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gbm_physical_evaluator import GBMPhysicalEvaluator  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "configs" / "receptors.yaml"
MANIFEST_PATH = PROJECT_ROOT / "data" / "receptors" / "receptors_manifest.json"


def load_receptor_config(path: Optional[Path] = None) -> dict:
    import yaml
    p = Path(path) if path else CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_target_key(target_raw: str, cfg: dict) -> Optional[str]:
    """把生成脚本输出的靶点名映射到 receptor config 的 key。"""
    aliases = cfg.get("target_aliases", {})
    if target_raw in aliases:
        return aliases[target_raw]
    if target_raw in cfg:
        return target_raw
    # 大小写不敏感兜底
    up = target_raw.upper()
    for k, v in aliases.items():
        if k.upper() == up:
            return v
    return None


def find_pdbqt(target_key: str, spec: dict) -> Optional[str]:
    """定位该靶点的受体 PDBQT（优先用 manifest 里的绝对路径）。"""
    if MANIFEST_PATH.exists():
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            entry = manifest.get(target_key, {})
            if entry.get("status") == "ok" and entry.get("pdbqt"):
                p = entry["pdbqt"]
                if os.path.isfile(p):
                    return p
                # manifest 里可能是相对路径
                cand = PROJECT_ROOT / p
                if cand.exists():
                    return str(cand)
        except Exception:
            pass
    pdbqt = spec.get("pdbqt")
    if pdbqt and os.path.isfile(pdbqt):
        return pdbqt
    # 最后按命名约定猜
    disp = str(spec.get("display_name", "")).replace("/", "_")
    for cand in (PROJECT_ROOT / "data" / "receptors").glob(f"*{spec.get('pdb_id','')}*chainA.pdbqt"):
        return str(cand)
    return None


def build_evaluator_for_target(
    target_raw: str,
    exhaustiveness: int = 8,
    cpu: int = 4,
    n_poses: int = 5,
    enable_vina: bool = True,
    use_admet_ai: bool = True,
    cfg: Optional[dict] = None,
) -> Optional[GBMPhysicalEvaluator]:
    """
    为指定靶点构建评估器（含真实 Vina 对接）。

    Returns:
        GBMPhysicalEvaluator；若靶点或受体缺失则返回 None。
    """
    cfg = cfg or load_receptor_config()
    key = resolve_target_key(target_raw, cfg)
    if key is None or not isinstance(cfg.get(key), dict):
        return None

    spec = cfg[key]
    pdbqt = find_pdbqt(key, spec)
    if not pdbqt:
        return None

    return GBMPhysicalEvaluator(
        vina_executable=None,
        receptor_pdbqt=pdbqt,
        vina_center=tuple(spec["box_center"]),
        vina_size=tuple(spec["box_size"]),
        enable_vina=enable_vina,
        use_admet_ai=use_admet_ai,
        vina_exhaustiveness=exhaustiveness,
        vina_cpu=cpu,
    )


def build_all_evaluators(
    targets=None,
    exhaustiveness: int = 8,
    cpu: int = 4,
    use_admet_ai: bool = True,
) -> Dict[str, GBMPhysicalEvaluator]:
    """为所有靶点构建评估器。"""
    cfg = load_receptor_config()
    if targets is None:
        targets = [k for k, v in cfg.items()
                   if isinstance(v, dict) and "pdb_id" in v]
    out = {}
    for t in targets:
        ev = build_evaluator_for_target(
            t, exhaustiveness=exhaustiveness, cpu=cpu,
            use_admet_ai=use_admet_ai, cfg=cfg,
        )
        if ev is not None:
            out[t] = ev
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--exhaustiveness", type=int, default=8)
    ap.add_argument("--cpu", type=int, default=4)
    ap.add_argument("--target", type=str, default=None)
    args = ap.parse_args()

    evs = build_all_evaluators(
        targets=[args.target] if args.target else None,
        exhaustiveness=args.exhaustiveness,
        cpu=args.cpu,
    )
    for k, ev in evs.items():
        backend = getattr(ev._vina_tool, "_backend", "pseudo")
        print(f"{k:14s} 后端={backend:10s} "
              f"受体={os.path.basename(ev._vina_tool.receptor_pdbqt)} "
              f"中心={ev._vina_tool.center} 盒子={ev._vina_tool.size}")
