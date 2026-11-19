#!/usr/bin/env python3
"""
并行重打分器
============

问题: 串行 Vina 对接太慢（60 核只用了 4 核）。
方案: 把「每个 CSV 文件」拆成独立 worker 进程，每个 worker 独占一个 Vina
      实例（maps 跨分子复用，见 property_pipeline/docking.py 的性能修复），
      并行跑。ADMET 推理在 GPU 上串行执行（避免显存争抢），
      Vina 在 CPU 上并行执行 —— 两者天然互补。

用法:
    python scripts/rescore_parallel.py --workers 12
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GBM_ROOT = PROJECT_ROOT / "gbm_project"
sys.path.insert(0, str(GBM_ROOT / "src"))
sys.path.insert(0, str(GBM_ROOT))

import pandas as pd

from property_pipeline.docking import VinaDocking, DOCKING_COLUMNS
from property_pipeline.descriptors import compute_descriptors
from property_pipeline.scaffold import compute_scaffold_metrics
from scripts.compute_properties import load_receptor_config, resolve_target, build_dockers


# ---------------------------------------------------------------- Vina worker
_WORKER_DOCKERS: Dict[str, VinaDocking] = {}
_WORKER_CFG = None


def _get_docker(target_key: str, cfg) -> VinaDocking:
    """每个 worker 进程内缓存 Vina 实例（maps 复用）。"""
    global _WORKER_DOCKERS, _WORKER_CFG
    if _WORKER_CFG is None:
        _WORKER_CFG = cfg
    if target_key not in _WORKER_DOCKERS:
        dockers = build_dockers(_WORKER_CFG, [target_key])
        _WORKER_DOCKERS[target_key] = dockers[target_key]
    return _WORKER_DOCKERS[target_key]


# ---------------------------------------------------------------- 药物样预筛

# 标准类药上限（Lipinski / Veber 规则）。对所有方法一致生效。
MAX_MW = 500.0
MAX_HEAVY_ATOMS = 40
MAX_ROTATABLE_BONDS = 10


def _is_druglike(smiles: str) -> Tuple[bool, str]:
    """
    预筛：把不可能作为类药分子的结构挡在对接之前。

    这不是"给基线放水"，而是所有方法共用同一道门槛；
    被挡下的比例本身会作为一个指标报告出来（`prefilter_pass_rate`）。
    """
    if not smiles:
        return False, "empty"
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "invalid_smiles"
        mw = Descriptors.MolWt(mol)
        heavy = mol.GetNumHeavyAtoms()
        rot = Descriptors.NumRotatableBonds(mol)
        if mw > MAX_MW:
            return False, f"mw={mw:.0f}"
        if heavy > MAX_HEAVY_ATOMS:
            return False, f"heavy={heavy}"
        if rot > MAX_ROTATABLE_BONDS:
            return False, f"rot={rot}"
    except Exception as e:
        return False, f"descriptor_error: {e}"
    return True, ""


def _dock_worker(task: Tuple[str, str, str, int, int]) -> List[Dict]:
    """
    子进程任务：对接一组分子。

    task = (csv_path, target_key, receptor_cfg_path, exhaustiveness, cpu)
    """
    csv_path, target_key, cfg_path, exh, cpu = task
    from scripts.compute_properties import load_receptor_config
    cfg = load_receptor_config(cfg_path)
    docker = _get_docker(target_key, cfg)

    df = pd.read_csv(csv_path)
    smiles_list = [str(s) if pd.notna(s) else "" for s in df["smiles"]]

    rows = []
    for smi in smiles_list:
        row = {c: float("nan") for c in DOCKING_COLUMNS}
        # ---- 药物样预筛（对所有方法一视同仁）----
        # Vina 的扭转树搜索对极端柔性分子（如长链烷烃 rot=147）会指数级
        # 爆炸，单分子可达数十分钟且无法完成。这类分子本身不是类药分子，
        # 通用 LLM 基线会大量产出此类退化产物。
        # 采用标准的类药上限（MW ≤ 500, 重原子 ≤ 40, 可旋转键 ≤ 10），
        # 对所有方法一致生效，并把预筛通过率作为独立指标报告。
        ok, reason = _is_druglike(smi)
        if not ok:
            row["vina_pose"] = f"prefilter: {reason}"
            rows.append(row)
            continue

        score, err = docker.dock(smi)
        if score is not None:
            row["vina_docking_score"] = round(score, 3)
            row["vina_affinity"] = round(score, 3)
        else:
            row["vina_pose"] = err[:200]
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str,
                        default=str(GBM_ROOT / "final_results" / "comparative"))
    parser.add_argument("--output_dir", type=str,
                        default=str(GBM_ROOT / "final_results" / "comparative" / "rescored"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--cpu_per_worker", type=int, default=4)
    parser.add_argument("--extra_csvs", type=str, default=None,
                        help="逗号分隔的额外 CSV（我们的方法），格式 dir/name.csv")
    parser.add_argument("--no_vina", action="store_true")
    parser.add_argument("--no_cache", action="store_true",
                        help="忽略磁盘上的 Vina 缓存，强制重算")
    args = parser.parse_args()

    cfg_path = GBM_ROOT / "configs" / "receptors.yaml"
    cfg = load_receptor_config(cfg_path)

    # 收集任务
    tasks: List[Tuple[str, str]] = []   # (method, csv_path)
    in_root = Path(args.input_dir)
    for method_dir in sorted(in_root.iterdir()):
        if not method_dir.is_dir() or method_dir.name == "rescored":
            continue
        # report/ 和 tables/ 是本脚本的输出目录，不是输入
        if method_dir.name in ("report", "tables"):
            continue
        for csv_path in sorted(method_dir.glob("*.csv")):
            tasks.append((method_dir.name, csv_path))

    if args.extra_csvs:
        for item in args.extra_csvs.split(","):
            item = item.strip()
            if not item:
                continue
            p = Path(item)
            if p.exists():
                tasks.append((p.parent.name, p))

    print(f"共 {len(tasks)} 个 CSV 待处理，{args.workers} workers，"
          f"exhaustiveness={args.exhaustiveness}\n")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- 阶段 1: Vina 并行对接 ----
    vina_results: Dict[str, List[Dict]] = {}

    # 载入缓存（对接很贵，避免重复计算）
    cache_path = out_root / "_vina_cache.json"
    if cache_path.exists() and not args.no_cache:
        try:
            vina_results = json.loads(cache_path.read_text(encoding="utf-8"))
            print(f"▶ 载入 Vina 缓存: {len(vina_results)} 个文件")
        except Exception as e:
            print(f"  ! 缓存读取失败: {e}")
            vina_results = {}

    if not args.no_vina:
        dock_tasks = []
        task_meta = []
        for method, csv_path in tasks:
            df = pd.read_csv(csv_path)
            tk = (resolve_target(csv_path.stem, cfg)
                  or _infer_target(csv_path, cfg))
            if tk is None:
                print(f"  ! 跳过（靶点未知）: {csv_path}")
                continue
            dock_tasks.append((str(csv_path), tk, str(cfg_path),
                               args.exhaustiveness, args.cpu_per_worker))
            task_meta.append((method, csv_path, tk))

        print(f"▶ 阶段 1/2: Vina 对接 ({len(dock_tasks)} 个文件, "
              f"{args.workers} workers × {args.cpu_per_worker} 核)")
        t0 = time.time()
        done = 0
        # 只对缓存里没有的任务真正计算
        pending = [t for t in dock_tasks
                   if str((t[0], t[1])) not in vina_results]
        cached = len(dock_tasks) - len(pending)
        if cached:
            print(f"  命中缓存 {cached} 个，待算 {len(pending)} 个")
        if pending:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(_dock_worker, t): t for t in pending}
                for fut in as_completed(futs):
                    t = futs[fut]
                    try:
                        rows = fut.result()
                    except Exception as e:
                        print(f"  ✗ 失败 {t[0]}: {e}")
                        continue
                    key = str((t[0], t[1]))
                    vina_results[key] = rows
                    done += 1
                    print(f"  [{done}/{len(pending)}] {Path(t[0]).parent.name}/"
                          f"{Path(t[0]).name}  ({time.time() - t0:.0f}s)")
                    # 增量落盘 —— 每完成一个文件就写一次缓存，
                    # 进程被杀也不会丢掉已算好的对接结果
                    try:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(vina_results, f)
                    except Exception:
                        pass
        print(f"  Vina 阶段完成，用时 {time.time() - t0:.0f}s\n")

        # 持久化缓存 —— 对接很贵，失败重跑时不必重算
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(vina_results, f)
            print(f"  缓存 -> {cache_path}\n")
        except Exception as e:
            print(f"  ! 缓存写入失败: {e}")

    # ---- 阶段 2: 描述符 + ADMET + 骨架（串行，ADMET 用 GPU）----
    print("▶ 阶段 2/2: 描述符 + ADMET + 骨架")
    from property_pipeline.admet import compute_admet, ADMET_COLUMNS
    from property_pipeline.descriptors import DESCRIPTOR_COLUMNS
    from property_pipeline.scaffold import SCAFFOLD_COLUMNS

    for method, csv_path, tk in (task_meta if not args.no_vina else
                                 [(m, c, resolve_target(c.stem, cfg) or _infer_target(c, cfg))
                                  for m, c in tasks]):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "smiles" not in df.columns:
            print(f"  ! 跳过（无 smiles 列）: {csv_path}")
            continue
        smiles_list = [str(s) if pd.notna(s) else "" for s in df["smiles"]]

        # 保留旧值
        if "drugclip_score" in df.columns:
            df["drugclip_score_external"] = df["drugclip_score"]
        if "vina_docking_score" in df.columns:
            df["vina_docking_score_legacy"] = df["vina_docking_score"]

        # 描述符
        desc_rows = [compute_descriptors(s) for s in smiles_list]
        for c in DESCRIPTOR_COLUMNS:
            if desc_rows and c in desc_rows[0]:
                df[c] = [r.get(c, float("nan")) for r in desc_rows]

        # ADMET
        try:
            admet_rows = compute_admet(smiles_list)
            for c in ADMET_COLUMNS:
                df[c] = [r.get(c, float("nan")) for r in admet_rows]
        except Exception as e:
            print(f"  ! ADMET 失败 {csv_path.name}: {e}")

        # 骨架
        try:
            scaf_rows = compute_scaffold_metrics(smiles_list)
            for c in SCAFFOLD_COLUMNS:
                if scaf_rows and c in scaf_rows[0]:
                    df[c] = [r.get(c, float("nan")) for r in scaf_rows]
        except Exception as e:
            print(f"  ! 骨架失败: {e}")

        # Vina
        key = str((str(csv_path), tk))
        if key in vina_results:
            for c in DOCKING_COLUMNS:
                df[c] = [r.get(c, float("nan")) for r in vina_results[key]]
        else:
            print(f"  ! 无 Vina 结果: {csv_path} (key={key})")

        out_dir = out_root / method.replace(" ", "_").replace("/", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / csv_path.name
        df.to_csv(out_path, index=False)
        n_vina = df["vina_docking_score"].notna().sum() if "vina_docking_score" in df.columns else 0
        print(f"  ✓ {method}/{csv_path.name}  n={len(df)}  vina_ok={n_vina}")

    print(f"\n完成 -> {out_root}")


def _infer_target(csv_path: Path, cfg) -> Optional[str]:
    from scripts.compute_properties import infer_target_from_filename
    return infer_target_from_filename(csv_path, cfg)


if __name__ == "__main__":
    main()
