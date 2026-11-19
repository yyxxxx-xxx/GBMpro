#!/usr/bin/env python3
"""
REINVENT 结果提取 → 标准基线 CSV
================================

把 REINVENT 的 RL 输出转成 `final_results/comparative/` 格式，
使其能被 `scripts/rescore_parallel.py` 用同一套管线重打分。

筛选策略（与我们的模型公平对齐）:
    - 去掉 Score=0 的（被 custom_alerts 过滤掉的）
    - 去掉 Vina 对接失败的
    - 取 Score 最高的前 N 个
    - 去重

用法:
    python scripts/extract_reinvent_results.py --target EGFR --n 20
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import List

import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GBM_ROOT = PROJECT_ROOT / "gbm_project"

EXP_DIR = GBM_ROOT / "experiments" / "reinvent_baseline"
OUT_DIR = GBM_ROOT / "final_results" / "comparative" / "REINVENT"

DISPLAY = "REINVENT4 (Vina-RL)"
DISPLAY_TLRL = "REINVENT4 (TL+RL)"


def extract(target: str, n: int, method_dir: Path = None,
            prefix: str = "_rl", display: str = DISPLAY) -> pd.DataFrame:
    exp_dir = method_dir or EXP_DIR
    csv_path = exp_dir / f"{target}{prefix}_1.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    # 只保留成功打分且 Vina 对接成功的
    df = df[pd.to_numeric(df["Score"], errors="coerce") > 0].copy()
    df["Score_num"] = pd.to_numeric(df["Score"], errors="coerce")
    df["Vina_num"] = pd.to_numeric(df["Vina (raw)"], errors="coerce")
    df = df[df["Vina_num"].notna()].copy()
    df = df[df["Vina_num"] > 0].copy()

    # 去重（按规范 SMILES）
    df = df.drop_duplicates(subset=["SMILES"]).copy()

    # 取分数最高的前 n 个（REINVENT 的分数已含 Vina + QED + MW）
    df = df.sort_values("Score_num", ascending=False).head(n)

    out = pd.DataFrame({
        "entity_id": target,
        "mol_id": [f"{display}_{target}_{i+1}" for i in range(len(df))],
        "library_name": display,
        "target": target,
        "smiles": df["SMILES"].astype(str).values,
    })
    out["reinvent_score"] = df["Score_num"].round(4).values
    # Vina (raw) 是归一化后的 [0,1]；换算回 kcal/mol 便于核对
    out["reinvent_vina_norm"] = df["Vina_num"].round(4).values
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--prefix", type=str, default="_rl",
                        help="REINVENT 结果文件前缀：_rl (纯 RL) 或 _tlrl (TL+RL)")
    parser.add_argument("--display", type=str, default=DISPLAY)
    parser.add_argument("--out_name", type=str, default=None,
                        help="输出文件名（不含 .csv），默认与 target 同名")
    args = parser.parse_args()

    df = extract(args.target, args.n, prefix=args.prefix, display=args.display)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{(args.out_name or args.target)}.csv"
    df.to_csv(out_path, index=False)
    print(f"✓ {out_path}  ({len(df)} 个分子)")
    for s in df["smiles"].head(5):
        print(f"   {s}")


if __name__ == "__main__":
    main()
