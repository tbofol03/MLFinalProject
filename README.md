
| 列名                           | 含义                                                                       |      |
| ---------------------------- | ------------------------------------------------------------------------ | ---- |
| `row_id`                     | 原始数据中的行号（从 1 开始），用于追溯与复现                                                 | 原始數據 |
| `Drug`                       | 药物名称                                                                     | 原始數據 |
| `Solvent`                    | 溶剂名称                                                                     | 原始數據 |
| `Drug SMILES`                | 原始药物 SMILES 字符串                                                          | 原始數據 |
| `Solvent SMILES`             | 原始溶剂 SMILES 字符串                                                          | 原始數據 |
| `Temperature`                | 测定温度（°C）                                                                 | 原始數據 |
| `Solubility_log10`           | 溶解度对数值（log₁₀，单位 mol/L）                                                   | 原始數據 |
| `Solubility Unit`            | 溶解度单位（均为 mol/L）                                                          | 原始數據 |
| `is_exact_duplicate`         | 是否为完全重复行（True 表示与其他行完全相同，已标记但保留）                                         |      |
| `has_conflicting_solubility` | 是否属于“相同条件但溶解度值不同”的冲突组（True 表示存在冲突，按规范保留未删除或平均）                           |      |
| `Drug_parse_success`         | 药物 SMILES 是否被 RDKit 成功解析（True/False）                                     |      |
| `Solvent_parse_success`      | 溶剂 SMILES 是否被 RDKit 成功解析（True/False）                                     |      |
| `Drug_canonical_SMILES`      | 药物的规范 SMILES（RDKit 标准化后）                                                 |      |
| `Solvent_canonical_SMILES`   | 溶剂的规范 SMILES（RDKit 标准化后）                                                 |      |
| `split`                      | 数据集划分标签：`development`（开发集）或 `holdout`（留出集），按规范 Drug SMILES 分组、种子 6133 生成 |      |
| `cv_fold`                    | 开发集内的交叉验证折编号（0–4）；留出集为 −1                                                |      |
| `Drug_MolWt`                 | 药物分子量                                                                    |      |
| `Drug_MolLogP`               | 药物脂水分配系数（MolLogP）                                                        |      |
| `Drug_TPSA`                  | 药物拓扑极性表面积                                                                |      |
| `Drug_NumHDonors`            | 药物氢键供体数                                                                  |      |
| `Drug_NumHAcceptors`         | 药物氢键受体数                                                                  |      |
| `Drug_NumRotatableBonds`     | 药物可旋转键数                                                                  |      |
| `Drug_HeavyAtomCount`        | 药物重原子数                                                                   |      |
| `Drug_FormalCharge`          | 药物形式电荷                                                                   |      |
| `Drug_NumAromaticRings`      | 药物芳香环数                                                                   |      |
| `Drug_NumHeteroatoms`        | 药物杂原子数                                                                   |      |
| `Solvent_MolWt`              | 溶剂分子量                                                                    |      |
| `Solvent_MolLogP`            | 溶剂脂水分配系数（MolLogP）                                                        |      |
| `Solvent_TPSA`               | 溶剂拓扑极性表面积                                                                |      |
| `Solvent_NumHDonors`         | 溶剂氢键供体数                                                                  |      |
| `Solvent_NumHAcceptors`      | 溶剂氢键受体数                                                                  |      |
| `Solvent_NumRotatableBonds`  | 溶剂可旋转键数                                                                  |      |
| `Solvent_HeavyAtomCount`     | 溶剂重原子数                                                                   |      |
| `Solvent_FormalCharge`       | 溶剂形式电荷                                                                   |      |
| `Solvent_NumAromaticRings`   | 溶剂芳香环数                                                                   |      |
| `Solvent_NumHeteroatoms`     | 溶剂杂原子数                                                                   |      |
### 推荐作为特征（Features）的列
| 类别 | 列名 |
|------|------|
| **温度** | `Temperature` |
| **药物分子描述符** | `Drug_MolWt`, `Drug_MolLogP`, `Drug_TPSA`, `Drug_NumHDonors`, `Drug_NumHAcceptors`, `Drug_NumRotatableBonds`, `Drug_HeavyAtomCount`, `Drug_FormalCharge`, `Drug_NumAromaticRings`, `Drug_NumHeteroatoms` |
| **溶剂分子描述符** | `Solvent_MolWt`, `Solvent_MolLogP`, `Solvent_TPSA`, `Solvent_NumHDonors`, `Solvent_NumHAcceptors`, `Solvent_NumRotatableBonds`, `Solvent_HeavyAtomCount`, `Solvent_FormalCharge`, `Solvent_NumAromaticRings`, `Solvent_NumHeteroatoms` |

以上共 21 列数值特征，已按规范冻结，适合直接用于表格型模型（如 Random Forest、XGBoost、LightGBM、MLP 等）。

### 目标列（Target）
- `Solubility_log10`

### 不建议作为特征的列（避免信息泄漏或无意义）
| 列名 | 原因 |
|------|------|
| `row_id` | 仅用于追溯，无预测价值 |
| `Drug`, `Solvent`, `Drug SMILES`, `Solvent SMILES`, `Drug_canonical_SMILES`, `Solvent_canonical_SMILES` | 字符串标识；若使用图神经网络或分子指纹可另行处理，但不应与数值描述符简单拼接 |
| `Solubility Unit` | 常量，无信息量 |
| `is_exact_duplicate`, `has_conflicting_solubility` | 数据质量标记，可用于过滤样本，但不应作为特征 |
| `Drug_parse_success`, `Solvent_parse_success` | 几乎全为 True，信息量极低 |
| `split`, `cv_fold` | 数据集划分与交叉验证标签，**仅用于模型评估与训练控制**，绝不可作为特征 |

### 使用建议
1. **训练集 / 验证集**：使用 `split == "development"` 的样本，并依据 `cv_fold`（0–4）进行交叉验证。
2. **最终测试**：仅在模型选定后，使用 `split == "holdout"` 的样本进行一次评估。
3. 如需处理冲突样本，可先根据 `has_conflicting_solubility` 决定是否保留或单独建模。
4. 若采用图模型或序列模型，可额外使用 `Drug_canonical_SMILES` 与 `Solvent_canonical_SMILES` 生成图或指纹，再与 `Temperature` 结合。

