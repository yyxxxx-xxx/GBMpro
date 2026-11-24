# PBCNet2.0 借鉴点 —— 实施记录 (Phase 1)
==========================================

> **日期**: 2026-11-24
> **备份**: `backup/pre-pbcnet-improvements` 分支 + `backup/pbcnet-baseline-20261124` 标签
> **依据**: `docs/PBCNET2_0_BENCHMARKS.md` 的优先级表
> **范围**: 全部不依赖 GPU 的 P0 / P1 项

---

## 1. 备份

```
git branch -f backup/pre-pbcnet-improvements HEAD
git tag -f backup/pbcnet-baseline-20261124 HEAD
git push origin backup/pre-pbcnet-improvements
git push origin backup/pbcnet-baseline-20261124
```

回滚方式:
```bash
git reset --hard backup/pbcnet-baseline-20261124
# 或
git checkout -b restore backup/pbcnet-baseline-20261124
```

---

## 2. P1-① 多标签冲突处理

**文件**: `scripts/clean_active_molecules.py`

### 规则

| 条件 | 处理 |
|---|---|
| `max_label <= 3 * min_label` | **几何平均**合并为 1 条 |
| `max_label >  3 * min_label` | 保留**最强测量** (min), 标记 `conflict=True` |
| `value_um <= 0` 或 None | 丢弃该条记录 |

### 与 PBCNet2.0 原文的差异 (重要)

原文:

> "if the maximum label exceeded three times the minimum, we removed the
> molecule to ensure data quality."

**我们改成了"保留最强测量"而非"丢弃整个分子"**, 原因:

原始数据 900 条只含 **583 个唯一分子** (来自 PubChem 多 assay 检索, 同一分子在多个
assay 下有多条测量)。若照搬"丢弃整个分子":

| 靶点 | 原始条数 | 照搬后 | 我们的方案 |
|---|---|---|---|
| EGFR | 150 | 67 | **68** |
| IDH1 | 150 | 82 | **89** |
| MGMT | 150 | 79 | **85** |
| PI3K_AKT_mTOR | 150 | 134 | **135** |
| VEGFR | 150 | 91 | **95** |
| p53_MDM2 | 150 | 109 | **111** |

照搬会额外损失"多次测量但都一致"的好数据。

### 副发现: "每靶点 150 条"是假均衡

清洗后暴露出真实覆盖度:

| 靶点 | 原始条数 | **唯一分子数** |
|---|---|---|
| EGFR | 150 | **68** |
| IDH1 | 150 | **89** |
| MGMT | 150 | **85** |
| PI3K_AKT_mTOR | 150 | **135** |
| VEGFR | 150 | **95** |
| p53_MDM2 | 150 | **111** |

原始的"150/靶点"是**重复计数**造成的假均衡。这在写论文时必须如实说明。

### 产物

- `data/lora_datasets/active_molecules_6targets_clean.json` (583 条)
  - 每条含 `n_labels_merged` (合并了几个测量)
  - 冲突分子含 `conflict` / `label_range_um` / `conflict_ratio`
- 原文件**未被覆盖** (保护 `active_qa_train.json` 的可复现性)

---

## 3. P1-② Modification Bias 审查

**文件**: `scripts/check_modification_bias.py`

### 要回答的问题

> "你的 RL 提升 reward +31%, 会不会只是**复现了 SFT 数据里固有的统计偏置**?"

### 方法

对每一对**共享 Murcko 骨架**的分子 (只改了 R-group), 计算 Δ = A - B,
按性质统计 Δ 的均值与 t 统计量。

### ⚠️ 配对方向必须外生 (踩坑记录)

这是实现时最关键的一个坑, 三种做法结果完全不同:

| 配对方式 | t (value_um) | 问题 |
|---|---|---|
| 按数据原始顺序取 A-B | **-12.33** | 顺序与活性相关 -> **造出假偏置** |
| 对称化取 (+d 和 -d 各一次) | **0.00** | 均值恒为 0 -> **检验失效** |
| **按 SMILES 字典序取 A-B** | **+2.69** | ✅ 顺序外生, 检验有效 |

SMILES 字典序是**外生于活性**的, 所以单样本 t 检验有效。

### 结果 (`experiments/bias_report.json`)

**无显著偏置** (✅ 数据中立):

| 性质 | n | Δ均值 | t |
|---|---|---|---|
| MW | 2228 | +1.01 | 1.158 |
| QED | 2228 | +0.0020 | 0.806 |
| DILI_prob | 2228 | -0.0003 | -0.185 |

**存在显著偏置** (⚠️ 必须在论文中如实报告):

| 性质 | n | Δ均值 | t | 含义 |
|---|---|---|---|---|
| **TPSA** | 2228 | **+3.12** | **+8.27** | 数据系统性偏向**高极性**改造 |
| **BBB_prob** | 2228 | **-0.0303** | **-9.07** | 数据系统性偏向**低血脑屏障**改造 |
| **hERG_prob** | 2228 | **+0.0107** | **+3.71** | 数据偏向**高心脏毒性**改造 |
| **AMES_prob** | 2228 | **-0.0112** | **-3.75** | 数据偏向**低致突变毒性**改造 |
| **value_um** | 2877 | **+0.0117** | **+2.69** | 活性改造存在微弱偏置 |

### 解读与应对

TPSA 的偏置 (t=8.27) 最值得注意: 数据集**系统性偏向高 TPSA 的改造**,
这与 BBB_prob 的负偏置方向一致 (高 TPSA -> 低 BBB 渗透)。

**对 RL 结论的影响**:
- ✅ 我们的 reward 里 BBB 是**硬约束** (`bbb_min`), 硬截断把高 TPSA 的高 reward 分子直接剪掉
- ✅ M3 实测: RL 的 BBB_prob 从 0.6582 -> **0.7393** (+0.081), **方向与数据偏置相反**
- ✅ 这说明 RL 不只是复现数据偏置, 而是在**对抗**它 (因为 reward 函数硬约束了 BBB)

> 📌 **这是本次分析最有价值的发现**: 原本担心"RL 提升只是复现偏置",
> 但 BBB 维度上 RL 的改善方向与数据偏置**相反**, 反而证明 RL 学到了
> reward 函数真正指定的东西, 而不是数据里的统计捷径。

### 后续动作

- [ ] 在论文的 bias/limitations 章节如实报告这张表
- [ ] 在评估时检查 RL 提升幅度是否超过偏置幅度
      (TPSA 偏置 +3.12, 但 reward 里 TPSA 被 normalize + clamp, 影响有限)

---

## 4. P0-① Advantage 模式扩展

**文件**: `src/rl/advantage.py`, `src/rl/trainer.py`, `scripts/train_rl.py`

### 四种模式

| 模式 | 公式 | 特点 |
|---|---|---|
| `zscore` (默认) | `(R - mean) / std` | **原实现, 行为逐位不变** |
| `pairwise` | `(R - mean) / (R_best - R_worst)` | PBCNet2.0 启发, 去偏移, 但所有组同尺度 |
| `pairwise_scaled` | `(R-mean)/rng * w`, `w = std_group/median(std)` | **推荐**, 去偏移 + 保留组间信息量 |
| `rank` | `(rank - (n+1)/2) / 16` | 最稳健 (对离群值免疫) |

### 设计依据

PBCNet2.0 的 readout (`models/readout.py`):

```python
emb = self.norm(emb1 - emb2)      # 只用差, 不拼接
out_put = self.FNN(emb)
```

论文:

> "employing Δ as the pair-level representation allows the model to devote more
> attention to protein-ligand interactions... if a model solely focuses on
> ligand molecular structures, [g_ref] and [g_query] would be similar as they
> are structural analogues, making Δ close to a null vector."

映射: z-score 隐含假设"绝对分数高 = 更好", 但 reward 里混着大量**与组内对比无关**
的常数偏移 (TPSA 归一化、MW 分段、ADMET 端点截断的固定地板), 会稀释组内真实差异。

### 离线对比 (`scripts/compare_advantage_modes.py`, 用 M3 真实评估数据)

```
模式                |adv|均值   advantage跨度均值   组间尺度CV
zscore                0.9130        2.6096            0.055
pairwise              0.3537        1.0000            0.000  ← 抹平组间差异
pairwise_scaled       0.3634        1.0231            0.062  ← 折中
rank                  0.4295        1.6562            0.153
```

**关键发现**:
- 纯 `pairwise` 的跨度在**所有组都恒为 1.0000** —— 把"这个组 reward 差异大 (信息多)"
  和"差异小 (信息少)"强行拉到同一尺度, 抹平了组间信息量差异
- `zscore` 保留了组间差异 (跨度 2.27~2.74 随组变化)
- `pairwise_scaled` 成功折中: 去偏移 (常数偏移不变) + 保留组间差异 (跨度 0.94~1.13, CV=0.062 ≈ zscore 的 0.055)

**离群值敏感度**: 四种模式都表现良好 (注入 999.0 的极端高分后组均 |adv| 变化 ≤ 8%)。

### 向后兼容保证

- 默认 `mode="zscore"`, **与改动前逐位一致** (单元测试 [1] 手算验证)
- `TrainerConfig.advantage_mode` 默认 `"zscore"`
- CLI `--advantage_mode` 默认 `"zscore"`

### 使用

```bash
python3 scripts/train_rl.py --target EGFR --algorithm grpo \
    --advantage_mode pairwise_scaled --steps 30 ... \
    --out_dir ../experiments/m5_egfr_pairwise
```

---

## 5. 单元测试

**文件**: `tests/test_advantage_modes.py` (28 个断言, 全部通过)

覆盖:
1. zscore 与原实现**逐位一致** (可复现性)
2. 组内 advantage 均值恒为 0
3. pairwise 值域 [-1, 1]
4. pairwise 对常数偏移不敏感 (PBCNet2.0 Δ 的核心动机)
5. pairwise 对正比例缩放不敏感
6. rank 对离群值免疫 (对照组: zscore 会被拉偏)
7. 零方差组被跳过 (三种模式一致)
8. 非法 mode 抛 ValueError
9. NaN 样本被排除
10. flatten 顺序一致
11. pairwise_scaled 折中性质

```bash
python3 tests/test_advantage_modes.py
# 通过 28 / 28
```

> ⚠️ `tests/test_reward_parity.py` 目前无法运行, 原因是**环境问题**:
> `pytest` 的 `typeguard` 插件与 `typing_extensions` 版本冲突
> (`cannot import name 'NoExtraItems'`)。已用 `git stash` 验证该失败在本次改动**之前**就存在。

---

## 6. 尚未实施 (需要 GPU 或更大改动)

| 项 | 状态 | 阻塞原因 |
|---|---|---|
| §6 差分对接 + 扰动归因 | 未做 | 需要写 pose 分析脚本 (不阻塞 GPU, 可后续做) |
| §5 多轮迭代 + 少样本评测 | 未做 | 需要 GPU 训练多轮 |
| §3 交互感知多样性过滤 | 未做 | 需要从 Vina pose 提取接触指纹 (中等工作量) |
| §4 锚点条件生成 | 未做 | 需要改 prompt_bank + 动态更新 |
| §8 构象柔性惩罚 | 未做 | 需要改 reward 函数 + 重新标定 |
| §2 对称性正则 | 未做 | 需要改 loss 函数 |
| M5-E 消融实验 | 待做 | 需要 GPU |

---

## 7. 建议的 M5 扩展计划

在原计划 (M5-A/B/C/D) 基础上增加 advantage 消融:

| 实验 | 靶点 | advantage_mode | 目的 |
|---|---|---|---|
| M5-A | EGFR | zscore | 与 M3 对齐的基线 |
| M5-B | EGFR | **pairwise_scaled** | 验证成对差分的收益 |
| M5-C | VEGFR | zscore | 跨靶点验证 |
| M5-D | VEGFR | **pairwise_scaled** | 跨靶点 × 差分 advantage |

**判据**: pairwise_scaled 若在 EGFR 和 VEGFR 上**都**优于 zscore,
说明成对差分 advantage 是**跨靶点通用**的改进, 可升为主力配置。

---

## 8. 产物清单

| 类型 | 路径 |
|---|---|
| 数据清洗脚本 | `gbm_project/scripts/clean_active_molecules.py` |
| 清洗后数据 | `gbm_project/data/lora_datasets/active_molecules_6targets_clean.json` |
| 偏置检查脚本 | `gbm_project/scripts/check_modification_bias.py` |
| 偏置报告 | `experiments/bias_report.json` |
| advantage 实现 | `gbm_project/src/rl/advantage.py` |
| 训练配置接入 | `gbm_project/src/rl/trainer.py`, `gbm_project/scripts/train_rl.py` |
| 离线对比脚本 | `gbm_project/scripts/compare_advantage_modes.py` |
| 对比结果 | `experiments/advantage_mode_comparison.json` |
| 单元测试 | `gbm_project/tests/test_advantage_modes.py` |
