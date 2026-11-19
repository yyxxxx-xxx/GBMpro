# 图1技术文档：GBMPhysicalEvaluator 流水线 + MPO Reward 公式 + 硬截断

## 1. 组件概述

`GBMPhysicalEvaluator`（位于 `src/gbm_physical_evaluator.py`）是系统的外部物理评估模块，内部整合了四个计算组件：
- `VinaDockingTool` — 分子对接评分
- `ADMETCalculator` — DILI + hERG + BBB 计算
- `MPORewardCalculator` — 多目标优化 Reward + 硬截断
- RDKit 内置描述符计算

## 2. 评估流水线（自上而下 5 层）

### Layer 0 — 输入
- 输入：SMILES 字符串（如 `"CC(=O)Nc1ccc(-n2ccnc2)cc1C(=O)Nc3ccccc3"`）

### Layer 1 — RDKit 描述符计算（本地，毫秒级）
```
TPSA、MW（ExactMolWt）、LogP（Crippen）、HBD、HBA、
Rotatable Bonds、Rings（芳香环数）
```

### Layer 2 — DILI 风险评估
**SMARTS 警示结构（7 种，全部为真实毒性基团）：**

| SMARTS | 中文名 | 修复建议 |
|--------|--------|---------|
| `C(=O)Cl` | 酰氯基团 | 用酰胺 `-C(=O)NH-` 或酯替换 |
| `N=C=S` | 异硫氰酸酯 | 用碳酰胺替代 |
| `C1CO1` | 未取代环氧烷 | 用内酰胺环替代 |
| `[cx]1([N+](=O)[O-])ccc1` | 二硝基芳烃 | 减少硝基或用磺酰胺替代 |
| `C1(=O)ccc(=O)c1` | 醌类 | 替换为氢醌二甲醚或环己酮 |
| `OP(=O)(O)O` | 磷酸酯 | 用膦酸酯替代 |
| `[N-]=[N+]=C` | 重氮盐 | 用稳定氨基 `-NH2` 替代 |

**EGFR Warhead 豁免逻辑：** Acrylamide（`-CH=CH-C(=O)NH-`）、Cyanoacrylamide（`-CC=CH-CN`）被明确豁免，DILI 评分乘以 0.25（即降低 75% 惩罚），因为它们是 EGFR 共价抑制剂的设计基团（如 Afatinib、Osimertinib）。

**DILI 评分规则：**
- 每命中 1 个 SMARTS 警示：+0.25
- 命中 ≥2 个警示结构：额外 +0.25
- EGFR Warhead 豁免：若同时命中 warhead，score × 0.25
- MW > 800 Da：+0.15；MW 700-800：+0.05
- LogP > 5.5：+0.10；LogP 4.5-5.5：+0.05
- 命中 PAINS（如 `c1cc(-n2nc(-c3ccccc3)nc2O)on1`）：+0.20

最终 `dili_prob = min(1.0, score)`

### Layer 3 — hERG 心脏毒性 + BBB 血脑屏障通透性

**hERG SMARTS 警示（10 种）：**

| SMARTS | 描述 |
|--------|------|
| `[NX3;H2,H1;!$(NC=O)]-[CX4H2]-[CX4H2]-[cR1]` | 脂溶性三级胺侧链 |
| `[cR1]1[cR1][cR1]([N+!0;D1])[cR1][cR1]1` | 氮杂芳环上的取代基 |
| `c1ccc2c(c1)cccc2` | 萘环 |
| `c1ccc2ccccc2c1` | 萘骨架（替代写法） |
| `[cR1]1[cR1][cR1][cR1]([N+](=O)[O-])[cR1][cR1]1` | 硝基芳香族 |
| `[cR1]1[cR1][cR1][cR1][cR1][cR1]1` | 未取代苯环（高 LogP 风险） |
| `[CX4H3]-[CX4H2]-[NX3;H2,H1]` | 烷基胺侧链 |
| `[cR1]1[cR1][cR1]([OX2H1])[cR1][cR1][cR1]1` | 苯酚/羟基芳香族 |
| `c1ccc(C(F)(F)F)cc1` | 三氟甲基苯 |
| `c1ccc(Cl)cc1` | 氯苯 |
| `c1ccc(Br)cc1` | 溴苯 |

每命中 1 个 hERG SMARTS：+0.15；`herg_prob = min(1.0, score)`

**BBB 通透性计算（Clark 公式 + 启发式惩罚）：**

第一步 Clark 公式（Clark & Delany, 2000）：
```
logBB_est = 0.152 × LogP − 0.0148 × TPSA + 0.139
```

第二步启发式惩罚：
```
if MW > 450: penalty -= 0.20
if HBD > 2:  penalty -= 0.15
if HBA > 6:  penalty -= 0.10
```

最终：
```
bbb_score = logBB_est + penalty
映射到 [0, 1]: bbb_score = max(0.0, min(1.0, (bbb_score + 1.0) / 1.5))
其中 logBB > 0.5 → 1.0, logBB < -1.0 → 0.0
```

### Layer 4 — Vina 对接 + MPO Reward 聚合 + 硬截断

**Vina 对接：**
- QVina/Wina 命令行对接（真实模式），或基于 MW/LogP/TPSA 的伪 Vina 公式：
  ```
  pseudo_vina = −0.08 × MW + 0.4 × LogP − 0.008 × TPSA − 5.5
  限制在 [-12, -4] kcal/mol
  ```

**归一化（每个维度映射到 [0, 1]，越高越好）：**
```
vina_norm  = (−4.0 − vina_score) / 8.0          # -12 → 1.0，-4 → 0.0
dili_norm  = 1 − dili_prob                       # 越低毒性越好，取补数
herg_norm  = 1 − herg_prob                       # 越低风险越好，取补数
bbb_norm   直接来自 Clark 公式输出
mw_norm    最优区间 300-500 Da（线性惩罚）
```

**TPSA 梯段式惩罚（Stepped Penalty）：**
```
TPSA > 140:  死亡线（hard prune），返回 −1
90 < TPSA ≤ 140:  norm = 1.0 − 2.0 × (TPSA − 90) / 50
40 ≤ TPSA ≤ 90:   理想区间，norm = 1.0
20 ≤ TPSA < 40:   低谷警告，norm = 0.5 + 0.5 × (TPSA − 20) / 20
TPSA < 20:       死亡线（hard prune），返回 −1
```

**硬截断规则（HARD_CUTOFFS，任一命中即剪枝）：**

| 指标 | 阈值 | 含义 |
|------|------|------|
| `vina_min` | -5.0 kcal/mol | 过弱结合 |
| `dili_max` | 0.95 | 极高肝毒性 |
| `herg_max` | 0.90 | 极高心脏风险 |
| `mw_min` | 200 Da | 过小，不类药 |
| `mw_max` | 700 Da | 过大，难穿透 BBB |
| `tpsa_min` | 20 Å² | 水溶性极差 |
| `tpsa_max` | 140 Å² | 肠道吸收困难 |

**Delta Scoring：** 若毒性（DILI/hERG）比父节点升高 ≤ 0.05，或有所下降，即使略微超过硬截断阈值也不触发剪枝，保留探索机会。

**MPO Reward 公式（加权几何平均，参考 REINVENT4）：**

```
PhysChem = √(TPSA_norm × MW_norm)

Reward = exp( w_vina · ln(Vina_norm)
            + w_dili · ln(DILI_norm)
            + w_herg · ln(hERG_norm)
            + w_bbb  · ln(BBB_norm)
            + w_phys · ln(PhysChem) )

其中 DILI_norm = 1 − DILI_prob，hERG_norm = 1 − hERG_prob
等价形式：Reward = (Vina_norm ^ w1) × (DILI_norm ^ w2) × (hERG_norm ^ w3) × (BBB_norm ^ w4) × (PhysChem ^ w5)
```

**权重配置（GBM 场景）：**
| 维度 | 权重 | 理由 |
|------|------|------|
| Vina | 0.30 | 结合亲和力是核心 |
| DILI | 0.20 | 安全性必须保障 |
| hERG | 0.20 | 心脏安全性 |
| BBB | 0.20 | BBB 穿透对 GBM 至关重要 |
| PhysChem | 0.10 | 类药性约束 |

**Verdict 判定：**
- Reward ≥ 0.7 → **SURE**（强烈推荐继续）
- Reward ≥ 0.4 → **LIKELY**（值得探索）
- Reward < 0.4 → **LIKELY**（仍保留探索机会，避免死胡同）

## 3. 输出

`PhysicalEvaluationResult` 数据类，包含：
- 原始物理量（`vina_score`, `dili_prob`, `herg_prob`, `bbb_score`）
- RDKit 描述符（`rd_tpsa`, `rd_mw`, `rd_logp`, `rd_hbd`, `rd_hba`）
- 归一化分数（`vina_norm`, `dili_norm`, `herg_norm`, `bbb_norm`, `tpsa_norm`）
- 最终 Reward、是否剪枝、Verdict、剪枝原因
- 警示结构列表（`dili_alert_matches`, `herg_alert_matches`）
- 自然语言反馈文本（`build_feedback_text()`，供 Prompt 注入）
