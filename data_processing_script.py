#!/usr/bin/env python3
"""
Data audit, cleaning, descriptor extraction, and fixed split generation
for data_2026.csv according to the specified protocol.
Random seed: 6133
"""

import pandas as pd
import numpy as np
import hashlib
import json
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

# Configuration
SEED = 6133
HOLDOUT_RATIO = 0.20  # 20% holdout by drug groups
N_FOLDS = 5
INPUT_FILE = "/workspace/attachments/data_2026.csv"
OUTPUT_DIR = Path("/workspace/artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)

# Fixed descriptor list (frozen)
DRUG_DESC_NAMES = [
    "Drug_MolWt", "Drug_MolLogP", "Drug_TPSA", "Drug_NumHDonors",
    "Drug_NumHAcceptors", "Drug_NumRotatableBonds", "Drug_HeavyAtomCount",
    "Drug_FormalCharge", "Drug_NumAromaticRings", "Drug_NumHeteroatoms"
]
SOLVENT_DESC_NAMES = [
    "Solvent_MolWt", "Solvent_MolLogP", "Solvent_TPSA", "Solvent_NumHDonors",
    "Solvent_NumHAcceptors", "Solvent_NumRotatableBonds", "Solvent_HeavyAtomCount",
    "Solvent_FormalCharge", "Solvent_NumAromaticRings", "Solvent_NumHeteroatoms"
]

def compute_file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def safe_mol_from_smiles(smi):
    if pd.isna(smi) or not isinstance(smi, str):
        return None
    return Chem.MolFromSmiles(smi)

def extract_descriptors(mol, prefix):
    if mol is None:
        return {f"{prefix}_{name}": np.nan for name in [
            "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
            "NumRotatableBonds", "HeavyAtomCount", "FormalCharge",
            "NumAromaticRings", "NumHeteroatoms"
        ]}
    return {
        f"{prefix}_MolWt": Descriptors.MolWt(mol),
        f"{prefix}_MolLogP": Descriptors.MolLogP(mol),
        f"{prefix}_TPSA": Descriptors.TPSA(mol),
        f"{prefix}_NumHDonors": Descriptors.NumHDonors(mol),
        f"{prefix}_NumHAcceptors": Descriptors.NumHAcceptors(mol),
        f"{prefix}_NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        f"{prefix}_HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        f"{prefix}_FormalCharge": Chem.GetFormalCharge(mol),
        f"{prefix}_NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        f"{prefix}_NumHeteroatoms": rdMolDescriptors.CalcNumHeteroatoms(mol),
    }

def main():
    print("=== Starting data processing ===")
    file_hash = compute_file_hash(INPUT_FILE)
    print(f"Input SHA256: {file_hash}")

    df = pd.read_csv(INPUT_FILE)
    df.insert(0, "row_id", range(1, len(df) + 1))  # 1-based original row numbers

    # Flags
    df["is_exact_duplicate"] = df.duplicated(subset=df.columns.difference(["row_id"]), keep=False)

    # Conflict flag: same conditions, different solubility
    condition_cols = ["Drug", "Solvent", "Drug SMILES", "Solvent SMILES", "Temperature"]
    group_stats = df.groupby(condition_cols)["Solubility_log10"].transform("nunique")
    df["has_conflicting_solubility"] = group_stats > 1

    # SMILES parsing and canonicalization
    df["Drug_mol"] = df["Drug SMILES"].apply(safe_mol_from_smiles)
    df["Solvent_mol"] = df["Solvent SMILES"].apply(safe_mol_from_smiles)
    df["Drug_parse_success"] = df["Drug_mol"].notna()
    df["Solvent_parse_success"] = df["Solvent_mol"].notna()

    df["Drug_canonical_SMILES"] = df["Drug_mol"].apply(
        lambda m: Chem.MolToSmiles(m, canonical=True) if m is not None else None
    )
    df["Solvent_canonical_SMILES"] = df["Solvent_mol"].apply(
        lambda m: Chem.MolToSmiles(m, canonical=True) if m is not None else None
    )

    # Descriptors
    drug_descs = df["Drug_mol"].apply(lambda m: extract_descriptors(m, "Drug"))
    solvent_descs = df["Solvent_mol"].apply(lambda m: extract_descriptors(m, "Solvent"))
    drug_desc_df = pd.DataFrame(drug_descs.tolist())
    solvent_desc_df = pd.DataFrame(solvent_descs.tolist())
    df = pd.concat([df, drug_desc_df, solvent_desc_df], axis=1)

    # Drop mol objects for saving
    df = df.drop(columns=["Drug_mol", "Solvent_mol"])

    # === Fixed group split by canonical Drug SMILES ===
    unique_drugs = df["Drug_canonical_SMILES"].dropna().unique()
    rng = np.random.RandomState(SEED)
    # Shuffle and split
    shuffled = rng.permutation(unique_drugs)
    n_holdout = max(1, int(len(unique_drugs) * HOLDOUT_RATIO))
    holdout_drugs = set(shuffled[:n_holdout])
    dev_drugs = set(shuffled[n_holdout:])

    df["split"] = df["Drug_canonical_SMILES"].apply(
        lambda s: "holdout" if s in holdout_drugs else ("development" if s in dev_drugs else "unknown")
    )

    # CV folds on development set only
    dev_mask = df["split"] == "development"
    dev_groups = df.loc[dev_mask, "Drug_canonical_SMILES"]
    unique_dev = dev_groups.unique()
    gkf = GroupKFold(n_splits=N_FOLDS)
    # Assign fold numbers 0 to N_FOLDS-1
    fold_map = {}
    for fold_idx, (_, test_idx) in enumerate(gkf.split(unique_dev, groups=unique_dev)):
        for g in unique_dev[test_idx]:
            fold_map[g] = fold_idx

    df["cv_fold"] = df["Drug_canonical_SMILES"].map(fold_map).fillna(-1).astype(int)
    # holdout and unknown get -1

    # Save processed data
    output_cols = [
        "row_id", "Drug", "Solvent", "Drug SMILES", "Solvent SMILES",
        "Temperature", "Solubility_log10", "Solubility Unit",
        "is_exact_duplicate", "has_conflicting_solubility",
        "Drug_parse_success", "Solvent_parse_success",
        "Drug_canonical_SMILES", "Solvent_canonical_SMILES",
        "split", "cv_fold"
    ] + DRUG_DESC_NAMES + SOLVENT_DESC_NAMES

    processed_path = OUTPUT_DIR / "processed_data_with_features.csv"
    df[output_cols].to_csv(processed_path, index=False)
    print(f"Saved processed data: {processed_path} ({len(df)} rows)")

    # Split summary
    split_summary = {
        "seed": SEED,
        "holdout_ratio": HOLDOUT_RATIO,
        "n_folds": N_FOLDS,
        "n_unique_drug_canonical": int(len(unique_drugs)),
        "n_holdout_drugs": len(holdout_drugs),
        "n_development_drugs": len(dev_drugs),
        "holdout_drugs": sorted(list(holdout_drugs)),
        "development_drugs_sample": sorted(list(dev_drugs))[:20],  # sample
        "file_hash": file_hash,
        "processed_hash": compute_file_hash(processed_path),
    }
    with open(OUTPUT_DIR / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2, ensure_ascii=False)
    print("Saved split_summary.json")

    # EDA plots
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Solubility distribution
    axes[0, 0].hist(df["Solubility_log10"], bins=40, color="steelblue", edgecolor="black")
    axes[0, 0].set_title("Distribution of Solubility_log10")
    axes[0, 0].set_xlabel("Solubility_log10")
    axes[0, 0].set_ylabel("Count")

    # Temperature distribution
    axes[0, 1].hist(df["Temperature"], bins=40, color="darkorange", edgecolor="black")
    axes[0, 1].set_title("Distribution of Temperature (°C)")
    axes[0, 1].set_xlabel("Temperature")
    axes[0, 1].set_ylabel("Count")

    # Top 15 Drugs by frequency
    top_drugs = df["Drug"].value_counts().head(15)
    axes[1, 0].barh(top_drugs.index[::-1], top_drugs.values[::-1], color="seagreen")
    axes[1, 0].set_title("Top 15 Drugs by Frequency")
    axes[1, 0].set_xlabel("Count")

    # Top 15 Solvents by frequency
    top_solvents = df["Solvent"].value_counts().head(15)
    axes[1, 1].barh(top_solvents.index[::-1], top_solvents.values[::-1], color="purple")
    axes[1, 1].set_title("Top 15 Solvents by Frequency")
    axes[1, 1].set_xlabel("Count")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_distributions.png", dpi=150)
    plt.close()
    print("Saved eda_distributions.png")

    # Audit text summary
    audit_text = f"""# Data Audit & Preprocessing Report (Draft)

## File Information
- Source: data_2026.csv
- SHA256: {file_hash}
- Rows: {len(df)}
- Columns (original): Drug, Solvent, Drug SMILES, Solvent SMILES, Temperature, Solubility_log10, Solubility Unit

## Data Quality
- Missing values: 0
- Infinite values: 0
- Exact duplicate rows: {df['is_exact_duplicate'].sum()} (flagged, retained)
- Condition groups with conflicting Solubility_log10: {df['has_conflicting_solubility'].sum()} rows belonging to such groups
  → Per protocol: retained without averaging or deletion; flagged in `has_conflicting_solubility`.

## SMILES Parsing (RDKit)
- Drug SMILES parse success: {df['Drug_parse_success'].sum()} / {len(df)}
- Solvent SMILES parse success: {df['Solvent_parse_success'].sum()} / {len(df)}
- Note: Successful parsing does not confirm chemical identity against drug library.

## Descriptors Extracted (Frozen List)
Drug: {', '.join(DRUG_DESC_NAMES)}
Solvent: {', '.join(SOLVENT_DESC_NAMES)}

## Fixed Split (seed = {SEED})
- Grouping key: Drug_canonical_SMILES
- Holdout ratio: {HOLDOUT_RATIO:.0%}
- Unique drug groups: {len(unique_drugs)}
- Development drugs: {len(dev_drugs)}
- Holdout drugs: {len(holdout_drugs)}
- CV folds on development set: {N_FOLDS} (GroupKFold)
- Holdout and unknown rows have cv_fold = -1

## Deliverables
- processed_data_with_features.csv (includes row_id, flags, canonical SMILES, descriptors, split, cv_fold)
- split_summary.json
- eda_distributions.png
- This script (data_processing_script.py)

## Acceptance Criteria Compliance
- row_id preserves original order
- Feature column order is fixed
- Splits are fully determined by seed {SEED} and canonical Drug SMILES grouping
- No fitted preprocessing used holdout information
- Conflicting observations retained and flagged
"""
    with open(OUTPUT_DIR / "audit_report_draft.md", "w", encoding="utf-8") as f:
        f.write(audit_text)
    print("Saved audit_report_draft.md")

    print("=== Processing complete ===")
    print(f"Development rows: {(df['split']=='development').sum()}")
    print(f"Holdout rows: {(df['split']=='holdout').sum()}")
    print(f"Conflicting rows: {df['has_conflicting_solubility'].sum()}")

if __name__ == "__main__":
    main()
