"""
Score all comparative experiment results using the same scoring function.
"""

import numpy as _np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

NAN = float("nan")


def clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def linear_score(x, x0, x1, y0=0.0, y1=1.0):
    if x1 == x0:
        return y1
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def score_mw(mw):
    if pd.isna(mw):
        return NAN
    if 250 <= mw <= 500:
        return 1.0
    if 150 <= mw < 250:
        return clip(linear_score(mw, 150, 250, 0, 1))
    if 500 < mw <= 650:
        return clip(linear_score(mw, 500, 650, 1, 0))
    return 0.0


def score_tpsa(tpsa):
    if pd.isna(tpsa):
        return NAN
    if 20 <= tpsa <= 90:
        return 1.0
    if 0 <= tpsa < 20:
        return clip(linear_score(tpsa, 0, 20, 0.7, 1.0))
    if 90 < tpsa <= 140:
        return clip(linear_score(tpsa, 90, 140, 1.0, 0.0))
    return 0.0


def score_logp(logp):
    if pd.isna(logp):
        return NAN
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
        return NAN
    return clip((6 - float(sa)) / 5)


def compute_target_stats(df):
    pos = df[df["DrugClip"] > 0]
    stats = {}
    if len(pos) > 0:
        stats["drugclip_positive_p95"] = _np.percentile(pos["DrugClip"], 95)
    else:
        stats["drugclip_positive_p95"] = 1.0
    vina_valid = df["Vina"].dropna()
    if len(vina_valid) >= 4:
        stats["vina_p5"] = _np.percentile(vina_valid, 5)
        stats["vina_p95"] = _np.percentile(vina_valid, 95)
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
        s_vina = NAN
    s_bbb = score_bbb(row.get("BBB"))
    s_mw = score_mw(row.get("MW"))
    s_tpsa = score_tpsa(row.get("TPSA"))
    s_logp = score_logp(row.get("LogP"))
    physchem_vals = [v for v in [s_mw, s_tpsa, s_logp] if not pd.isna(v)]
    s_physchem = sum(physchem_vals) / len(physchem_vals) if physchem_vals else NAN
    s_pgp = score_pgp(row.get("Pgp"))
    s_alert = score_alert(row.get("PAINS"), row.get("Brenk"))
    s_sa = score_sa(row.get("SA"))
    raw = (
        0.30 * s_drugclip
        + 0.30 * (s_vina if not pd.isna(s_vina) else 0.5)
        + 0.15 * s_bbb
        + 0.08 * (s_physchem if not pd.isna(s_physchem) else 0.5)
        + 0.05 * s_pgp
        + 0.07 * s_alert
        + 0.05 * (s_sa if not pd.isna(s_sa) else 0.5)
    )
    return round(100 * raw, 4)


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
            df[col] = NAN
    return df


def score_file(csv_path):
    df_raw = pd.read_csv(csv_path)
    df = standardize_columns(df_raw)

    for col in ["smiles", "SMILES"]:
        if col in df.columns and col != "SMILES":
            df = df.rename(columns={col: "SMILES"})
    if "SMILES" not in df.columns:
        df["SMILES"] = ""

    n_raw = len(df)
    df = df.drop_duplicates(subset=["SMILES"], keep="first")
    if df.empty:
        return None, None, n_raw, 0

    n_before = len(df)
    df = df.dropna(subset=["Vina"])
    n_after = len(df)

    stats = compute_target_stats(df)

    df["TotalScore"] = df.apply(lambda row: score_one_molecule(row, stats), axis=1)

    def _s_dc(r):
        if pd.notna(r["DrugClip"]) and r["DrugClip"] > 0:
            return clip(r["DrugClip"] / stats["drugclip_positive_p95"])
        return 0.0
    df["S_DrugClip"] = df.apply(_s_dc, axis=1)

    vina_vals = []
    for _, r in df.iterrows():
        if pd.notna(r["Vina"]):
            s_v = clip(
                (stats["vina_p95"] - r["Vina"])
                / max(stats["vina_p95"] - stats["vina_p5"], 1e-9)
            )
        else:
            s_v = NAN
        vina_vals.append(s_v)
    df["S_Vina"] = vina_vals
    df["S_BBB"] = df["BBB"].apply(score_bbb)

    def _s_physchem(r):
        vals = [score_mw(r["MW"]), score_tpsa(r["TPSA"]), score_logp(r["LogP"])]
        valid = [v for v in vals if not pd.isna(v)]
        return sum(valid) / len(valid) if valid else NAN
    df["S_PhysChem"] = df.apply(_s_physchem, axis=1)
    df["S_Pgp"] = df["Pgp"].apply(score_pgp)
    df["S_Alert"] = df.apply(lambda r: score_alert(r["PAINS"], r["Brenk"]), axis=1)
    df["S_SA"] = df["SA"].apply(score_sa)

    df = df.sort_values("TotalScore", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))

    return df, stats, n_raw, n_after


def infer_target(fname):
    f = fname.upper()
    if "IDH" in f:
        return "IDH1"
    if "VEGF" in f or "VEGFR" in f:
        return "VEGF"
    return "EGFR"


base_dir = Path(__file__).parent
comparative_dir = base_dir / "final_results" / "comparative"
output_dir = base_dir / "scored_results" / "comparative"
output_dir.mkdir(parents=True, exist_ok=True)

methods = {
    "LlaSMol": comparative_dir / "llasmol",
    "PlayMolecule_AI": comparative_dir / "PlayMolecule AI",
    "Llamole_retro": comparative_dir / "llamole",
}

all_rows = []

for method_name, method_dir in methods.items():
    if not method_dir.exists():
        print(f"[SKIP] {method_dir}")
        continue

    print(f"\n{'='*60}")
    print(f"Method: {method_name}")
    print('='*60)

    for csv_path in sorted(method_dir.glob("*.csv")):
        target = infer_target(csv_path.name)
        print(f"\n  [{method_name}] {csv_path.name} -> target={target}")

        result = score_file(csv_path)
        if result[0] is None:
            print(f"    [SKIP] No data")
            continue

        df, stats, n_raw, n_after = result

        print(f"    Norm: DrugClip P95={stats['drugclip_positive_p95']:.3f}, "
              f"Vina P5={stats['vina_p5']:.2f}, P95={stats['vina_p95']:.2f}")
        print(f"    Deduplicated: {n_raw} -> {n_after}  "
              f"(dropped missing Vina: {n_raw - n_after})")

        # Header
        h1 = "  Rank    Score   S_DC   S_Vina  S_BBB  DrugClip    Vina  BBB      MW"
        print(h1)
        print("  " + "-" * (len(h1) - 2))

        for _, row in df.head(5).iterrows():
            dc = f"{row['DrugClip']:.3f}" if pd.notna(row.get('DrugClip')) else "N/A"
            vn = f"{row['Vina']:.2f}" if pd.notna(row.get('Vina')) else "N/A"
            svn = f"{row['S_Vina']:.3f}" if pd.notna(row.get('S_Vina')) else "0.000"
            bbb = str(row.get('BBB', '?'))
            mw = f"{row['MW']:.1f}" if pd.notna(row.get('MW')) else "N/A"
            print(f"  {int(row['Rank']):>4}  {row['TotalScore']:>6.2f}  "
                  f"{row['S_DrugClip']:>5.3f}  {svn:>7}  {row['S_BBB']:>5.2f}  "
                  f"{dc:>8}  {vn:>7}  {bbb:>5}  {mw:>7}")

        # Save
        out_path = output_dir / f"comparative_{method_name}_{target}_ranked.csv"
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
        print(f"    -> {out_path.name}  ({len(df)} molecules)")

        # Summary
        n_pos = int((df["DrugClip"] > 0).sum())
        n_bbb = int((df["BBB"].astype(str).str.strip().str.lower() == "yes").sum())
        top = df.iloc[0]
        all_rows.append({
            "方法": method_name,
            "靶点": target,
            "总分子数": len(df),
            "DrugClip>0数": n_pos,
            "DrugClip>0率": f"{100*n_pos/len(df):.1f}%" if len(df) > 0 else "N/A",
            "BBB通过数": n_bbb,
            "BBB通过率": f"{100*n_bbb/len(df):.1f}%" if len(df) > 0 else "N/A",
            "Top1_Score": f"{top['TotalScore']:.2f}",
            "Top1_DrugClip": f"{top['DrugClip']:.3f}" if pd.notna(top.get('DrugClip')) else "N/A",
            "Top1_Vina": f"{top['Vina']:.2f}" if pd.notna(top.get('Vina')) else "N/A",
            "平均Score": f"{df['TotalScore'].mean():.2f}",
            "DrugClip均值": f"{df['DrugClip'].mean():.3f}" if df['DrugClip'].notna().any() else "N/A",
            "Vina均值": f"{df['Vina'].mean():.2f}" if df['Vina'].notna().any() else "N/A",
        })

# Print and save summary
print(f"\n\n{'='*90}")
print("COMPARATIVE SUMMARY TABLE")
print('='*90)

df_summary = pd.DataFrame(all_rows)
order_map = {"EGFR": 0, "IDH1": 1, "VEGF": 2}
df_summary["_s"] = df_summary["靶点"].map(order_map)
df_summary = df_summary.sort_values(["靶点", "_s"]).drop(columns="_s")

print(df_summary.to_string(index=False))

summary_path = output_dir / "comparative_summary.csv"
df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
print(f"\nSaved: {summary_path}")
