"""
Score llasmol molecules using the same scoring function as score_experiments.py
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ─── Scoring functions (copied from score_experiments.py) ─────────────────────

def clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def linear_score(x, x0, x1, y0=0.0, y1=1.0):
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


def compute_target_stats(df):
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


def score_one_molecule(row, target_stats):
    drugclip = row["DrugClip"]
    vina = row["Vina"]

    if pd.isna(drugclip) or drugclip < 0:
        return 0.0

    p95_pos = target_stats.get("drugclip_positive_p95", None)
    if p95_pos and p95_pos > 0:
        s_drugclip = clip(drugclip / p95_pos)
    else:
        s_drugclip = 1.0

    v_good = target_stats.get("vina_p5", None)
    v_bad = target_stats.get("vina_p95", None)
    if pd.notna(vina) and v_good is not None and v_bad is not None and (v_bad - v_good) != 0:
        s_vina = clip((v_bad - vina) / (v_bad - v_good))
    else:
        s_vina = np.nan

    s_bbb = score_bbb(row.get("BBB"))
    s_mw = score_mw(row.get("MW"))
    s_tpsa = score_tpsa(row.get("TPSA"))
    s_logp = score_logp(row.get("LogP"))
    physchem_valid = [v for v in [s_mw, s_tpsa, s_logp] if not np.isnan(v)]
    s_physchem = np.mean(physchem_valid) if physchem_valid else np.nan
    s_pgp = score_pgp(row.get("Pgp"))
    s_alert = score_alert(row.get("PAINS"), row.get("Brenk"))
    s_sa = score_sa(row.get("SA"))

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


# ─── Field name mapping ────────────────────────────────────────────────────────

FIELD_ALIASES = {
    "drugclip_score": "DrugClip",
    "vina_docking_score": "Vina",
    "Consensus Log P": "LogP",
    "BBB permeant": "BBB",
    "Pgp substrate": "Pgp",
    "PAINS #alerts": "PAINS",
    "Brenk #alerts": "Brenk",
    "Synthetic Accessibility": "SA",
}


def standardize_columns(df):
    df = df.copy()
    rename = {}
    for col in df.columns:
        col_stripped = col.strip()
        if col_stripped in FIELD_ALIASES:
            rename[col] = FIELD_ALIASES[col_stripped]
    df = df.rename(columns=rename)
    for col in ["DrugClip", "Vina", "MW", "TPSA", "LogP", "BBB", "Pgp", "PAINS", "Brenk", "SA"]:
        if col not in df.columns:
            df[col] = np.nan
    return df


def score_file(csv_path):
    df_raw = pd.read_csv(csv_path)
    df = standardize_columns(df_raw)

    # Ensure SMILES column
    for col in ["SMILES", "smiles", "SMILES"]:
        if col in df.columns and col != "SMILES":
            df = df.rename(columns={col: "SMILES"})
    if "SMILES" not in df.columns:
        df["SMILES"] = ""

    # Deduplicate
    df = df.drop_duplicates(subset=["SMILES"], keep="first")

    if df.empty:
        return None

    # Drop molecules with missing Vina score
    n_before = len(df)
    df = df.dropna(subset=["Vina"])
    n_after = len(df)
    if n_before != n_after:
        print(f"    [Note] Dropped {n_before - n_after} molecules with missing Vina score")

    if df.empty:
        return None

    stats = compute_target_stats(df)

    df["TotalScore"] = df.apply(lambda row: score_one_molecule(row, stats), axis=1)

    df["S_DrugClip"] = df.apply(
        lambda r: clip(r["DrugClip"] / stats["drugclip_positive_p95"])
        if (pd.notna(r["DrugClip"]) and r["DrugClip"] > 0) else 0.0,
        axis=1,
    )

    vina_vals = []
    for _, r in df.iterrows():
        if pd.notna(r["Vina"]):
            s_v = clip(
                (stats["vina_p95"] - r["Vina"])
                / max(stats["vina_p95"] - stats["vina_p5"], 1e-9)
            )
        else:
            s_v = np.nan
        vina_vals.append(s_v)
    df["S_Vina"] = vina_vals
    df["S_BBB"] = df["BBB"].apply(score_bbb)
    df["S_PhysChem"] = df.apply(
        lambda r: np.nanmean([score_mw(r["MW"]), score_tpsa(r["TPSA"]), score_logp(r["LogP"])]),
        axis=1,
    )
    df["S_Pgp"] = df["Pgp"].apply(score_pgp)
    df["S_Alert"] = df.apply(lambda r: score_alert(r["PAINS"], r["Brenk"]), axis=1)
    df["S_SA"] = df["SA"].apply(score_sa)

    df = df.sort_values("TotalScore", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df, stats


# ─── Main ──────────────────────────────────────────────────────────────────────

base_dir = Path(__file__).parent
llasmol_dir = base_dir / "final_results" / "comparative" / "llasmol"
output_dir = base_dir / "scored_results" / "llasmol"
output_dir.mkdir(parents=True, exist_ok=True)

results_all = {}
for csv_path in sorted(llasmol_dir.glob("*.csv")):
    target = csv_path.stem
    print(f"\n{'='*70}")
    print(f"Target: {target}  |  File: {csv_path.name}")
    print('='*70)

    result = score_file(csv_path)
    if result is None:
        print("  [SKIP] No data")
        continue

    df, stats = result

    print(f"\n  Normalization stats (within this target):")
    print(f"    DrugClip positive P95 : {stats['drugclip_positive_p95']:.4f}")
    print(f"    Vina P5              : {stats['vina_p5']:.4f}")
    print(f"    Vina P95             : {stats['vina_p95']:.4f}")

    print(f"\n  {'Rank':>4}  {'Score':>8}  {'S_DC':>6}  {'S_Vina':>7}  {'S_BBB':>5}  "
          f"{'S_PC':>5}  {'S_Pgp':>6}  {'S_Alrt':>6}  {'S_SA':>5}  "
          f"{'DrugClip':>8}  {'Vina':>8}  {'MW':>7}  {'BBB':>4}  {'Pgp':>4}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*5}  "
          f"{'-'*5}  {'-'*6}  {'-'*6}  {'-'*5}  "
          f"{'-'*8}  {'-'*8}  {'-'*7}  {'-'*4}  {'-'*4}")

    for _, row in df.iterrows():
        s_vina_str = f"{row['S_Vina']:.3f}" if pd.notna(row['S_Vina']) else 'N/A'
        s_pc_str = f"{row['S_PhysChem']:.3f}" if pd.notna(row['S_PhysChem']) else 'N/A'
        s_sa_str = f"{row['S_SA']:.3f}" if pd.notna(row['S_SA']) else 'N/A'
        dclip_str = f"{row['DrugClip']:.3f}" if pd.notna(row['DrugClip']) else 'N/A'
        vina_str = f"{row['Vina']:.2f}" if pd.notna(row['Vina']) else 'N/A'
        mw_str = f"{row['MW']:.1f}" if pd.notna(row['MW']) else 'N/A'

        print(
            f"  {int(row['Rank']):>4}  {row['TotalScore']:>8.4f}  "
            f"{row['S_DrugClip']:>6.3f}  {s_vina_str:>7}  "
            f"{row['S_BBB']:>5.2f}  "
            f"{s_pc_str:>5}  "
            f"{row['S_Pgp']:>6.2f}  "
            f"{row['S_Alert']:>6.2f}  "
            f"{s_sa_str:>5}  "
            f"{dclip_str:>8}  "
            f"{vina_str:>8}  "
            f"{mw_str:>7}  "
            f"{str(row.get('BBB','N/A')):>4}  "
            f"{str(row.get('Pgp','N/A')):>4}"
        )

    out_path = output_dir / f"llasmol_{target}_ranked.csv"
    cols = [
        "Rank", "TotalScore",
        "S_DrugClip", "S_Vina", "S_BBB", "S_PhysChem",
        "S_Pgp", "S_Alert", "S_SA",
        "DrugClip", "Vina", "MW", "TPSA", "LogP",
        "BBB", "Pgp", "PAINS", "Brenk", "SA",
        "SMILES", "entity_id", "mol_id", "library_name",
    ]
    out_cols = [c for c in cols if c in df.columns]
    extra = [c for c in df.columns if c not in out_cols]
    df[out_cols + extra].to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path.name}  ({len(df)} molecules)")

    results_all[target] = (df, stats)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n\n" + "="*70)
print("SUMMARY ACROSS ALL TARGETS")
print("="*70)

summary_rows = []
for target, (df, stats) in results_all.items():
    top = df.iloc[0]
    n_positive = (df["DrugClip"] > 0).sum()
    n_with_vina = df["Vina"].notna().sum()
    avg_score = df["TotalScore"].mean()
    top3_avg = df.head(3)["TotalScore"].mean()

    summary_rows.append({
        "Target": target,
        "N_total": len(df),
        "N_DrugClip>0": n_positive,
        "N_with_Vina": n_with_vina,
        "DClip_P95": round(stats["drugclip_positive_p95"], 3),
        "Vina_P5": round(stats["vina_p5"], 3),
        "Vina_P95": round(stats["vina_p95"], 3),
        "Top1_Score": round(top["TotalScore"], 4),
        "Top1_DClip": round(top["DrugClip"], 3) if pd.notna(top["DrugClip"]) else 0,
        "Top1_Vina": round(top["Vina"], 3) if pd.notna(top["Vina"]) else None,
        "Avg_Score": round(avg_score, 4),
        "Top3_Avg": round(top3_avg, 4),
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

summary_path = output_dir / "llasmol_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\nSummary saved -> {summary_path.name}")
