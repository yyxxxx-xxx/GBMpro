"""
GBM Molecule Scoring Script
============================
对 model_test 和 ablation 实验结果进行自动化评分，按实验分组、按靶点分文件，
按总分从高到低排名。

评分函数参照 GBM_molecule_scoring_function.md：
  - DrugClip 子分 (30%)：归一化至正值 P95
  - Vina Dock 子分 (30%)：P5-P95 稳健归一化，越低越好
  - BBB 渗透分 (15%)：Yes=1, No=0, missing=0.5
  - 理化性质分 (8%)：MW + TPSA + LogP 三项平均
  - P-gp 外排风险分 (5%)：No=1, Yes=0.4, missing=0.5
  - 结构警报分 (7%)：PAINS + Brenk 加权
  - 可合成性分 (5%)：(6-SA)/5

输出文件命名规范：
  {experiment_name}_{target_name}_ranked.csv

用法：
  python score_experiments.py
"""

import os
import sys
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# 评分函数
# =============================================================================

def clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def linear_score(x, x0, x1, y0=0.0, y1=1.0):
    """线性插值 from (x0,y0) to (x1,y1)."""
    if x1 == x0:
        return y1
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def score_mw(mw):
    if pd.isna(mw):
        return np.nan
    if 250 <= mw <= 500:
        return 1.0
    if 150 <= mw < 250:
        return clip(linear_score(mw, 150, 250, 0, 1))
    if 500 < mw <= 650:
        return clip(linear_score(mw, 500, 650, 1, 0))
    return 0.0


def score_tpsa(tpsa):
    if pd.isna(tpsa):
        return np.nan
    if 20 <= tpsa <= 90:
        return 1.0
    if 0 <= tpsa < 20:
        return clip(linear_score(tpsa, 0, 20, 0.7, 1.0))
    if 90 < tpsa <= 140:
        return clip(linear_score(tpsa, 90, 140, 1.0, 0.0))
    return 0.0


def score_logp(logp):
    if pd.isna(logp):
        return np.nan
    if 1.0 <= logp <= 4.0:
        return 1.0
    if 0.0 <= logp < 1.0:
        return clip(linear_score(logp, 0.0, 1.0, 0.5, 1.0))
    if 4.0 < logp <= 6.0:
        return clip(linear_score(logp, 4.0, 6.0, 1.0, 0.0))
    return 0.0


def score_bbb(val):
    if pd.isna(val):
        return 0.5
    if str(val).strip().lower() == "yes":
        return 1.0
    return 0.0


def score_pgp(val):
    if pd.isna(val):
        return 0.5
    if str(val).strip().lower() == "no":
        return 1.0
    return 0.4


def score_pains(n):
    if pd.isna(n):
        return 0.5
    n = int(n)
    if n == 0:
        return 1.0
    if n == 1:
        return 0.2
    return 0.0


def score_brenk(n):
    if pd.isna(n):
        return 0.5
    return clip(1 - int(n) / 5)


def score_alert(pains_n, brenk_n):
    return 0.60 * score_pains(pains_n) + 0.40 * score_brenk(brenk_n)


def score_sa(sa):
    if pd.isna(sa):
        return np.nan
    return clip((6 - float(sa)) / 5)


def score_one_molecule(row, target_stats):
    """
    计算单个分子的综合评分。
    target_stats: dict with 'drugclip_positive_p95', 'vina_p5', 'vina_p95'
    """
    drugclip = row["DrugClip"]
    vina = row["Vina"]

    # DrugClip < 0 → 总分直接归零
    if pd.isna(drugclip) or drugclip < 0:
        return 0.0

    # DrugClip 子分
    p95_pos = target_stats.get("drugclip_positive_p95", None)
    if p95_pos and p95_pos > 0:
        s_drugclip = clip(drugclip / p95_pos)
    else:
        s_drugclip = 1.0

    # Vina 子分
    v_good = target_stats.get("vina_p5", None)
    v_bad = target_stats.get("vina_p95", None)
    if (
        pd.notna(vina)
        and v_good is not None
        and v_bad is not None
        and (v_bad - v_good) != 0
    ):
        s_vina = clip((v_bad - vina) / (v_bad - v_good))
    else:
        s_vina = np.nan

    # BBB 子分
    s_bbb = score_bbb(row.get("BBB"))

    # 理化性质子分
    s_mw = score_mw(row.get("MW"))
    s_tpsa = score_tpsa(row.get("TPSA"))
    s_logp = score_logp(row.get("LogP"))
    physchem_valid = [v for v in [s_mw, s_tpsa, s_logp] if not np.isnan(v)]
    s_physchem = np.mean(physchem_valid) if physchem_valid else np.nan

    # P-gp 子分
    s_pgp = score_pgp(row.get("Pgp"))

    # 结构警报子分
    s_alert = score_alert(row.get("PAINS"), row.get("Brenk"))

    # SA 子分
    s_sa = score_sa(row.get("SA"))

    # 加权总分
    raw = (
        0.30 * s_drugclip
        + 0.30 * (s_vina if not np.isnan(s_vina) else 0.5)
        + 0.15 * s_bbb
        + 0.08 * (s_physchem if not np.isnan(s_physchem) else 0.5)
        + 0.05 * s_pgp
        + 0.07 * s_alert
        + 0.05 * (s_sa if not np.isnan(s_sa) else 0.5)
    )
    return round(100 * raw, 4)


def compute_target_stats(df):
    """
    计算每个靶点内部的归一化统计量（仅基于 DrugClip > 0 的分子）。
    """
    pos = df[df["DrugClip"] > 0]
    stats = {}
    if len(pos) > 0:
        stats["drugclip_positive_p95"] = np.percentile(pos["DrugClip"], 95)
    else:
        stats["drugclip_positive_p95"] = 1.0

    vina_valid = df["Vina"].dropna()
    if len(vina_valid) >= 4:
        stats["vina_p5"] = np.percentile(vina_valid, 5)
        stats["vina_p95"] = np.percentile(vina_valid, 95)
    elif len(vina_valid) > 0:
        stats["vina_p5"] = vina_valid.min()
        stats["vina_p95"] = vina_valid.max()
    else:
        stats["vina_p5"] = -10.0
        stats["vina_p95"] = 0.0

    return stats


# =============================================================================
# 字段名标准化映射（model_test 和 ablation 列名不完全一致）
# =============================================================================

FIELD_ALIASES = {
    # alias -> canonical
    "DrugClip Score": "DrugClip",
    "drugclip_score": "DrugClip",
    "DrugClip": "DrugClip",
    "Vina Dock Score": "Vina",
    "vina_docking_score": "Vina",
    "Vina": "Vina",
    "Consensus Log P": "LogP",
    "Consensus LogP": "LogP",
    "LogP": "LogP",
    "BBB permeant": "BBB",
    "BBB": "BBB",
    "Pgp substrate": "Pgp",
    "Pgp": "Pgp",
    "PAINS Alerts": "PAINS",
    "PAINS #alerts": "PAINS",
    "PAINS": "PAINS",
    "Brenk Alerts": "Brenk",
    "Brenk #alerts": "Brenk",
    "Brenk": "Brenk",
    "Synthetic Accessibility": "SA",
    "SyntheticAccessibility": "SA",
    "SA": "SA",
}


def standardize_columns(df):
    """将列名标准化为 canonical 名称，并选择/重命名所需字段。"""
    df = df.copy()

    # 先做别名映射
    rename = {}
    for col in df.columns:
        col_stripped = col.strip()
        if col_stripped in FIELD_ALIASES:
            rename[col] = FIELD_ALIASES[col_stripped]
        elif col in FIELD_ALIASES:
            rename[col] = FIELD_ALIASES[col]
    df = df.rename(columns=rename)

    # 确保 canonical 列存在，不存在则填充空
    for col in ["DrugClip", "Vina", "MW", "TPSA", "LogP", "BBB", "Pgp", "PAINS", "Brenk", "SA"]:
        if col not in df.columns:
            df[col] = np.nan

    return df


# =============================================================================
# 靶点名标准化
# =============================================================================

def normalize_target(raw_target):
    """将原始靶点名称映射为统一标签。"""
    t = str(raw_target).strip().upper()
    if t in ("EGFR", "EGFR_00000"):
        return "EGFR"
    if "IDH" in t:
        return "IDH1"
    if "VEGF" in t or "VEGFR" in t:
        return "VEGF"
    # fallback
    return t


# =============================================================================
# 主流程
# =============================================================================

def guess_target_from_path(csv_path):
    """从文件路径和内容猜靶点名。"""
    path_str = str(csv_path).upper()
    if "EGFR" in path_str:
        return "EGFR"
    if "IDH" in path_str:
        return "IDH1"
    if "VEGF" in path_str or "VEGFR" in path_str:
        return "VEGF"
    # fallback：从文件内容读取
    return None


def load_and_score(csv_path, experiment_name):
    """
    读取单个 CSV，按靶点分组评分，返回 {target: df} 字典。
    """
    try:
        df_raw = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [WARN] 读取失败 {csv_path}: {e}")
        return {}

    if df_raw.empty:
        return {}

    df = standardize_columns(df_raw)

    # 确保 SMILES 列名统一（VEGF 文件用小写 smiles）
    for col in ["smiles", "SMILES"]:
        if col in df.columns and col != "SMILES":
            df = df.rename(columns={col: "SMILES"})
    if "SMILES" not in df.columns:
        df["SMILES"] = ""

    # 识别靶点列
    target_col = None
    for col in ["靶点", "library_name", "Target", "target"]:
        if col in df.columns:
            target_col = col
            break

    # 识别 SMILES 列
    smiles_col = None
    for col in ["SMILES", "smiles", "Smiles"]:
        if col in df.columns:
            smiles_col = col
            break

    # 如果靶点列缺失，从文件名推断
    results_by_target = {}
    if target_col is not None:
        for raw_t, sub in df.groupby(target_col):
            t = normalize_target(raw_t)
            if t not in results_by_target:
                results_by_target[t] = []
            results_by_target[t].append(sub.copy())
    else:
        # 没有靶点列 → 整个文件视为单靶点
        guessed = guess_target_from_path(csv_path)
        t = guessed if guessed else "UNKNOWN"
        results_by_target[t] = [df.copy()]

    # 确保 SMILES 列名统一（VEGF 文件用小写 smiles）
    smiles_col = None
    for col in ["SMILES", "smiles"]:
        if col in df.columns:
            smiles_col = col
            break
    if smiles_col and smiles_col != "SMILES":
        df = df.rename(columns={smiles_col: "SMILES"})
    if "SMILES" not in df.columns:
        df["SMILES"] = ""

    scored = {}
    for t, frames in results_by_target.items():
        concat_df = pd.concat(frames, ignore_index=True)
        sub = concat_df.drop_duplicates(subset=["SMILES"], keep="first")
        if sub.empty:
            continue

        # 计算该靶点内部的归一化统计量
        stats = compute_target_stats(sub)

        # 评分
        sub["TotalScore"] = sub.apply(lambda row: score_one_molecule(row, stats), axis=1)

        # 排名（降序，高分在前）
        sub = sub.sort_values("TotalScore", ascending=False).reset_index(drop=True)
        sub.insert(0, "Rank", range(1, len(sub) + 1))

        # 子分拆解（方便后续分析）
        sub["S_DrugClip"] = sub.apply(
            lambda r: clip(r["DrugClip"] / stats["drugclip_positive_p95"])
            if (pd.notna(r["DrugClip"]) and r["DrugClip"] > 0)
            else 0.0,
            axis=1,
        )
        vina_vals = []
        for _, r in sub.iterrows():
            if pd.notna(r["Vina"]):
                s_v = clip(
                    (stats["vina_p95"] - r["Vina"])
                    / max(stats["vina_p95"] - stats["vina_p5"], 1e-9)
                )
            else:
                s_v = np.nan
            vina_vals.append(s_v)
        sub["S_Vina"] = vina_vals
        sub["S_BBB"] = sub["BBB"].apply(score_bbb)
        sub["S_PhysChem"] = sub.apply(
            lambda r: np.nanmean([score_mw(r["MW"]), score_tpsa(r["TPSA"]), score_logp(r["LogP"])]),
            axis=1,
        )
        sub["S_Pgp"] = sub["Pgp"].apply(score_pgp)
        sub["S_Alert"] = sub.apply(lambda r: score_alert(r["PAINS"], r["Brenk"]), axis=1)
        sub["S_SA"] = sub["SA"].apply(score_sa)

        scored[t] = sub

    return scored


def run():
    base_dir = Path(__file__).parent
    output_dir = base_dir / "scored_results"
    output_dir.mkdir(exist_ok=True)

    # --- model_test ---
    model_test_dir = base_dir / "final_results" / "model_test"
    if model_test_dir.exists():
        print("\n" + "=" * 70)
        print("处理: model_test")
        print("=" * 70)
        csv_files = sorted(model_test_dir.glob("*.csv"))
        for csv_path in csv_files:
            print(f"\n  >> {csv_path.name}")
            results = load_and_score(csv_path, "model_test")
            for t, df in results.items():
                out_name = f"model_test_{t}_ranked.csv"
                out_path = output_dir / out_name
                cols = [
                    "Rank", "TotalScore",
                    "S_DrugClip", "S_Vina", "S_BBB", "S_PhysChem",
                    "S_Pgp", "S_Alert", "S_SA",
                    "DrugClip", "Vina", "MW", "TPSA", "LogP",
                    "BBB", "Pgp", "PAINS", "Brenk", "SA",
                    "SMILES",
                ]
                out_cols = [c for c in cols if c in df.columns]
                # 保留原始文件中所有其他列
                extra = [c for c in df.columns if c not in out_cols]
                df[out_cols + extra].to_csv(out_path, index=False)
                print(f"      → {out_name}  ({len(df)} molecules)")
                # Top-5 摘要
                top5 = df.head(5)[["Rank", "TotalScore", "DrugClip", "Vina", "MW", "BBB"]].to_string(index=False)
                print(f"      Top-5:\n{top5}")

    # --- ablation ---
    ablation_dir = base_dir / "final_results" / "ablation"
    if ablation_dir.exists():
        print("\n" + "=" * 70)
        print("处理: ablation")
        print("=" * 70)
        # 按实验分组
        exp_dirs = sorted(d for d in ablation_dir.iterdir() if d.is_dir())
        for exp_dir in exp_dirs:
            exp_name = exp_dir.name  # e.g. ablation_1_chain_of_thought
            print(f"\n  实验组: {exp_name}")
            csv_files = sorted(exp_dir.glob("*.csv"))
            for csv_path in csv_files:
                # 从文件名提取靶点（EGFR_RE.csv → EGFR）
                fname_stem = csv_path.stem.replace("_RE", "").replace("_merged_summary", "")
                print(f"    >> {csv_path.name}")
                results = load_and_score(csv_path, exp_name)
                for t, df in results.items():
                    out_name = f"{exp_name}_{t}_ranked.csv"
                    out_path = output_dir / out_name
                    cols = [
                        "Rank", "TotalScore",
                        "S_DrugClip", "S_Vina", "S_BBB", "S_PhysChem",
                        "S_Pgp", "S_Alert", "S_SA",
                        "DrugClip", "Vina", "MW", "TPSA", "LogP",
                        "BBB", "Pgp", "PAINS", "Brenk", "SA",
                        "SMILES",
                    ]
                    out_cols = [c for c in cols if c in df.columns]
                    extra = [c for c in df.columns if c not in out_cols]
                    df[out_cols + extra].to_csv(out_path, index=False)
                    print(f"        → {out_name}  ({len(df)} molecules)")

    print(f"\n所有结果已保存至: {output_dir}")
    print("文件命名规范: {experiment}_{target}_ranked.csv")


if __name__ == "__main__":
    run()
