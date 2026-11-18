#!/usr/bin/env python3
"""
Robust data audit, cleaning, descriptor extraction, and fixed split generation
for data_2026.csv according to the specified protocol.
Random seed: 6133

Usage:
    python data_processing_script_robust.py [path_to_data_2026.csv]
"""

import pandas as pd
import numpy as np
import hashlib
import json
import sys
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold

# Configuration
SEED = 6133
HOLDOUT_RATIO = 0.20  # 20% holdout by drug groups
N_FOLDS = 5
OUTPUT_DIR = Path(".")
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

def read_csv_robust(path):
    """Try multiple encodings to avoid UnicodeDecodeError."""
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin1", "cp1252"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"Successfully read with encoding: {enc}")
            return df, enc
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to read {path} with tried encodings. Last error: {last_err}")

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
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
    else:
        input_file = Path("data_2026.csv")
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        print("Usage: python data_processing_script_robust.py [path_to_data_2026.csv]")
        sys.exit(1)

    print("=== Starting data processing ===")
    file_hash = compute_file_hash(input_file)
    print(f"Input SHA256: {file_hash}")

    df, used_encoding = read_csv_robust(input_file)
    df.insert(0, "row_id", range(1, len(df) + 1))  # 1-based original row numbers

    # Flags
    df["is_exact_duplicate"] = df.duplicated(subset=df.columns.difference(["row_id"]), keep=False)

    # Conflict flag: same conditions, different solubility
    condition_cols = ["Drug", "Solvent", "Drug SMILES", "Solvent SMILES", "Temperature"]
    # Only if these columns exist
    if all(c in df.columns for c in condition_cols):
        group_stats = df.groupby(condition_cols)["Solubility_log10"].transform("nunique")
        df["has_conflicting_solubility"] = group_stats > 1
    else:
        df["has_conflicting_solubility"] = False
        print("Warning: Some condition columns missing; conflict flag set to False")

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
    shuffled = rng.permutation(unique_drugs)
    n_holdout = max(1, int(len(unique_drugs) * HOLDOUT_RATIO))
    holdout_drugs = set(shuffled[:n_holdout])
    dev_drugs = set(shuffled[n_holdout:])

    df["split"] = df["Drug_canonical_SMILES"].apply(
        lambda s: "holdout" if s in holdout_drugs else ("development" if s in dev_drugs else "unknown")
    )

    # CV folds on development set only
    dev_mask = df["split"] == "development"
    unique_dev = df.loc[dev_mask, "Drug_canonical_SMILES"].dropna().unique()
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_map = {}
    for fold_idx, (_, test_idx) in enumerate(gkf.split(unique_dev, groups=unique_dev)):
        for g in unique_dev[test_idx]:
            fold_map[g] = fold_idx

    df["cv_fold"] = df["Drug_canonical_SMILES"].map(fold_map).fillna(-1).astype(int)

    # Save processed data
    output_cols = [
        "row_id", "Drug", "Solvent", "Drug SMILES", "Solvent SMILES",
        "Temperature", "Solubility_log10", "Solubility Unit",
        "is_exact_duplicate", "has_conflicting_solubility",
        "Drug_parse_success", "Solvent_parse_success",
        "Drug_canonical_SMILES", "Solvent_canonical_SMILES",
        "split", "cv_fold"
    ] + DRUG_DESC_NAMES + SOLVENT_DESC_NAMES

    # Keep only columns that exist
    output_cols = [c for c in output_cols if c in df.columns]

    processed_path = OUTPUT_DIR / "processed_data_with_features.csv"
    df[output_cols].to_csv(processed_path, index=False, encoding="utf-8-sig")  # utf-8-sig for Excel compatibility
    print(f"Saved processed data: {processed_path} ({len(df)} rows)")

    # Split summary
    split_summary = {
        "seed": SEED,
        "holdout_ratio": HOLDOUT_RATIO,
        "n_folds": N_FOLDS,
        "n_unique_drug_canonical": int(len(unique_drugs)),
        "n_holdout_drugs": len(holdout_drugs),
        "n_development_drugs": len(dev_drugs),
        "input_encoding_used": used_encoding,
        "file_hash": file_hash,
        "processed_hash": compute_file_hash(processed_path),
    }
    with open(OUTPUT_DIR / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2, ensure_ascii=False)
    print("Saved split_summary.json")

    # EDA plots (optional, skip if matplotlib fails)
    try:
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0, 0].hist(df["Solubility_log10"], bins=40, color="steelblue", edgecolor="black")
        axes[0, 0].set_title("Distribution of Solubility_log10")
        axes[0, 0].set_xlabel("Solubility_log10")
        axes[0, 0].set_ylabel("Count")
        axes[0, 1].hist(df["Temperature"], bins=40, color="darkorange", edgecolor="black")
        axes[0, 1].set_title("Distribution of Temperature (°C)")
        axes[0, 1].set_xlabel("Temperature")
        axes[0, 1].set_ylabel("Count")
        top_drugs = df["Drug"].value_counts().head(15)
        axes[1, 0].barh(top_drugs.index[::-1], top_drugs.values[::-1], color="seagreen")
        axes[1, 0].set_title("Top 15 Drugs by Frequency")
        axes[1, 0].set_xlabel("Count")
        top_solvents = df["Solvent"].value_counts().head(15)
        axes[1, 1].barh(top_solvents.index[::-1], top_solvents.values[::-1], color="purple")
        axes[1, 1].set_title("Top 15 Solvents by Frequency")
        axes[1, 1].set_xlabel("Count")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "eda_distributions.png", dpi=150)
        plt.close()
        print("Saved eda_distributions.png")
    except Exception as e:
        print(f"Warning: Could not generate EDA plots: {e}")

    print("=== Processing complete ===")
    print(f"Development rows: {(df['split']=='development').sum()}")
    print(f"Holdout rows: {(df['split']=='holdout').sum()}")

if __name__ == "__main__":
    main()
