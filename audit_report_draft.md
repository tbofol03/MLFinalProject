# Data Audit & Preprocessing Report (Draft)

## File Information
- Source: data_2026.csv
- SHA256: 7a7138354a2a2534cff4540d8366feeba4370cc0a00634fbb85dcc571d2dd84c
- Rows: 1904
- Columns (original): Drug, Solvent, Drug SMILES, Solvent SMILES, Temperature, Solubility_log10, Solubility Unit

## Data Quality
- Missing values: 0
- Infinite values: 0
- Exact duplicate rows: 8 (flagged, retained)
- Condition groups with conflicting Solubility_log10: 148 rows belonging to such groups
  → Per protocol: retained without averaging or deletion; flagged in `has_conflicting_solubility`.

## SMILES Parsing (RDKit)
- Drug SMILES parse success: 1904 / 1904
- Solvent SMILES parse success: 1904 / 1904
- Note: Successful parsing does not confirm chemical identity against drug library.

## Descriptors Extracted (Frozen List)
Drug: Drug_MolWt, Drug_MolLogP, Drug_TPSA, Drug_NumHDonors, Drug_NumHAcceptors, Drug_NumRotatableBonds, Drug_HeavyAtomCount, Drug_FormalCharge, Drug_NumAromaticRings, Drug_NumHeteroatoms
Solvent: Solvent_MolWt, Solvent_MolLogP, Solvent_TPSA, Solvent_NumHDonors, Solvent_NumHAcceptors, Solvent_NumRotatableBonds, Solvent_HeavyAtomCount, Solvent_FormalCharge, Solvent_NumAromaticRings, Solvent_NumHeteroatoms

## Fixed Split (seed = 6133)
- Grouping key: Drug_canonical_SMILES
- Holdout ratio: 20%
- Unique drug groups: 168
- Development drugs: 135
- Holdout drugs: 33
- CV folds on development set: 5 (GroupKFold)
- Holdout and unknown rows have cv_fold = -1

## Deliverables
- processed_data_with_features.csv (includes row_id, flags, canonical SMILES, descriptors, split, cv_fold)
- split_summary.json
- eda_distributions.png
- This script (data_processing_script.py)

## Acceptance Criteria Compliance
- row_id preserves original order
- Feature column order is fixed
- Splits are fully determined by seed 6133 and canonical Drug SMILES grouping
- No fitted preprocessing used holdout information
- Conflicting observations retained and flagged
