#!/usr/bin/env python3
"""
Merge IDH1 DRUGCLIP and ADMET tables by SMILES (using RDKit canonicalization for fuzzy match).
"""
import pandas as pd
from rdkit import Chem

def canonical_smiles(smiles):
    """Return the RDKit canonical SMILES, or None if invalid."""
    if not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

BASE = "/root/Llamole-main/gbm_project/experiments/tot_batch_20260422_162424/IDH1_IDH2_result"

drugclip = pd.read_csv(f"{BASE}/IDH1_DRUGCLIP.csv")
admet = pd.read_csv(f"{BASE}/IDH1_ADMET.csv")

# Normalize SMILES
drugclip["_smiles_norm"] = drugclip["smiles"].apply(canonical_smiles)
admet["_smiles_norm"] = admet["Canonical SMILES"].apply(canonical_smiles)

# Drop rows where normalization failed
n_dc_fail = drugclip["_smiles_norm"].isna().sum()
n_admet_fail = admet["_smiles_norm"].isna().sum()
drugclip = drugclip.dropna(subset=["_smiles_norm"])
admet = admet.dropna(subset=["_smiles_norm"])

print(f"DRUGCLIP: {len(drugclip)} / {len(drugclip)+n_dc_fail} rows after normalization "
      f"({n_dc_fail} failed)")
print(f"ADMET: {len(admet)} / {len(admet)+n_admet_fail} rows after normalization "
      f"({n_admet_fail} failed)")

# For duplicate normalized SMILES in admet, keep first
admet = admet.drop_duplicates(subset=["_smiles_norm"], keep="first")

# Merge on normalized SMILES
merged = pd.merge(
    drugclip, admet,
    left_on="_smiles_norm", right_on="_smiles_norm",
    how="outer", indicator=True, suffixes=("_dc", "_admet")
)

both = merged[merged["_merge"] == "both"]
left_only = merged[merged["_merge"] == "left_only"]   # only in DRUGCLIP
right_only = merged[merged["_merge"] == "right_only"]  # only in ADMET

print(f"\nTotal unique molecules: {len(merged)}")
print(f"  - In both tables: {len(both)}")
print(f"  - Only in DRUGCLIP: {len(left_only)}")
print(f"  - Only in ADMET: {len(right_only)}")

final_rows = []

# --- molecules in both ---
for _, row in both.iterrows():
    final_rows.append({
        "靶点": "IDH1",
        "SMILES": row["smiles"],
        "DrugClip Score": row["drugclip_score"],
        "Vina Dock Score": row["vina_docking_score"],
        "MW": row["MW"],
        "TPSA": row["TPSA"],
        "Consensus Log P": row["Consensus Log P"],
        "Pgp substrate": row["Pgp substrate"],
        "BBB permeant": row["BBB permeant"],
        "PAINS Alerts": row["PAINS #alerts"],
        "Brenk Alerts": row["Brenk #alerts"],
    })

# --- molecules only in DRUGCLIP ---
for _, row in left_only.iterrows():
    final_rows.append({
        "靶点": "IDH1",
        "SMILES": row["smiles"],
        "DrugClip Score": row["drugclip_score"],
        "Vina Dock Score": row["vina_docking_score"],
        "MW": None,
        "TPSA": None,
        "Consensus Log P": None,
        "Pgp substrate": None,
        "BBB permeant": None,
        "PAINS Alerts": None,
        "Brenk Alerts": None,
    })

# --- molecules only in ADMET ---
for _, row in right_only.iterrows():
    final_rows.append({
        "靶点": "IDH1",
        "SMILES": row["Canonical SMILES"],
        "DrugClip Score": None,
        "Vina Dock Score": None,
        "MW": row["MW"],
        "TPSA": row["TPSA"],
        "Consensus Log P": row["Consensus Log P"],
        "Pgp substrate": row["Pgp substrate"],
        "BBB permeant": row["BBB permeant"],
        "PAINS Alerts": row["PAINS #alerts"],
        "Brenk Alerts": row["Brenk #alerts"],
    })

final = pd.DataFrame(final_rows)

# Sort by DrugClip Score descending (NaN at bottom)
final = final.sort_values(
    by="DrugClip Score", ascending=False, na_position="last"
).reset_index(drop=True)

# Save
out_path = f"{BASE}/IDH1_merged_summary.csv"
final.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\nSaved to: {out_path}")

# Show sample
pd.set_option("display.max_colwidth", 40)
pd.set_option("display.width", 220)
print(f"\n--- Preview (first 20 rows) ---")
print(final.head(20).to_string())
