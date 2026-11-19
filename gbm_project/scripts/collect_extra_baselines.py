#!/usr/bin/env python3
"""
把 comparative_extra 下的 base 模型结果整理成统一基线目录
==========================================================

`generate_extra_baselines.py` 输出的是扁平文件（base_qwen_EGFR.csv），
而 `rescore_parallel.py` 期望的是 `<方法名>/<靶点>.csv` 结构。
这个脚本负责转换，并把 base 模型重命名为有意义的展示名。

用法:
    python scripts/collect_extra_baselines.py
"""

import shutil
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

GBM_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = GBM_ROOT / "final_results" / "comparative_extra"
DST_ROOT = GBM_ROOT / "final_results" / "comparative"

# 源文件前缀 -> 输出目录名
METHOD_MAP = {
    "base_qwen": "base_qwen",
    "base_mistral": "base_mistral",
}

# 靶点文件名归一
TARGET_CANON = {
    "EGFR": "EGFR",
    "VEGFR": "VEGFR",
    "IDH1": "IDH1",
}


def main():
    if not SRC_DIR.exists():
        print(f"! 源目录不存在: {SRC_DIR}")
        return

    for src in sorted(SRC_DIR.glob("base_*.csv")):
        stem = src.stem  # base_qwen_EGFR
        parts = stem.split("_")
        if len(parts) < 3:
            print(f"  ! 跳过（命名不识别）: {src.name}")
            continue
        method = METHOD_MAP.get(f"{parts[0]}_{parts[1]}")
        target = TARGET_CANON.get(parts[2].upper())
        if not method or not target:
            print(f"  ! 跳过: {src.name}")
            continue

        df = pd.read_csv(src)
        if len(df) == 0:
            print(f"  ! 跳过（空文件）: {src.name}")
            continue

        out_dir = DST_ROOT / method
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{target}.csv"
        df.to_csv(out_path, index=False)
        print(f"✓ {out_path}  ({len(df)} 个分子)")


if __name__ == "__main__":
    main()
