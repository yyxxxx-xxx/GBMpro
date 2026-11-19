# GBM分子生成消融实验设计

## 1. 实验概述

### 1.1 消融实验目的

本消融实验（Ablation Study）旨在系统性地评估 GBM 分子生成项目中各核心模块对最终生成分子质量的贡献程度。通过逐个关闭或替换关键组件，定量分析每个模块的独立价值和组合效果，为方法论优化提供数据支撑。

### 1.2 项目核心架构回顾

```
┌─────────────────────────────────────────────────────────────┐
│                      Llamole 基座模型                       │
│              (Qwen2-7B-Instruct + Graph LM)                  │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌─────────────────┐   ┌─────────────────┐
│ GBM知识库注入  │    │ CoT/ToT推理引擎  │   │ LoRA微调Adapter │
│ (Domain KB)    │    │ (Reasoning)     │   │ (Fine-tuned)    │
└───────────────┘    └─────────────────┘   └─────────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │    分子约束过滤器      │
                  │ (MW/LogP/TPSA/HBD/HBA)│
                  └───────────────────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │   多维度评估体系       │
                  │ BBB/活性/毒性/合成    │
                  └───────────────────────┘
```

---

## 2. 消融维度设计

### 2.1 消融维度总览

本实验设计了 **6 大消融维度**，涵盖从输入端到输出端的全链路分析：

| 维度编号 | 消融维度 | 组件描述 | 消融类型 |
|---------|---------|---------|---------|
| Ablation-1 | GBM领域知识注入 | 靶点信息、临床数据、分子知识 | 组件开关 |
| Ablation-2 | CoT/ToT推理策略 | Chain-of-Thought 与 Tree-of-Thought | 策略替换 |
| Ablation-3 | 靶点特异性优化 | EGFR/VEGF/IDH 等 8 种靶点 | 变量控制 |
| Ablation-4 | 分子约束条件 | MW/LogP/TPSA 等物理化学约束 | 参数梯度 |
| Ablation-5 | LoRA微调适配器 | 有/无 LoRA adapter 的差异 | 预训练 vs 微调 |
| Ablation-6 | Prompt模板变体 | 不同 prompt 策略的效果对比 | 模板替换 |

---

## 3. 详细实验设计

### 3.1 Ablation-1：GBM领域知识注入的贡献

#### 实验动机

GBM领域知识库（`gbm_knowledge_base.py`）是项目的核心创新点之一，提供了靶点生物学信息、临床治疗经验和分子设计原则。需要验证该知识注入是否真正提升了生成分子的质量。

#### 实验配置

```
┌────────────────────────────────────────────────────────────────┐
│                    Ablation-1: 领域知识注入                     │
├──────────┬──────────────┬─────────────┬───────────────────────┤
│ 实验组    │ GBM知识库     │ 临床数据    │ 分子参考库              │
├──────────┼──────────────┼─────────────┼───────────────────────┤
│ Exp-0    │     ✗         │     ✗       │     ✗ (基线)            │
│ Exp-1    │     ✓         │     ✗       │     ✗                  │
│ Exp-2    │     ✗         │     ✓       │     ✗                  │
│ Exp-3    │     ✗         │     ✗       │     ✓                  │
│ Exp-4    │     ✓         │     ✓       │     ✗                  │
│ Exp-5    │     ✓         │     ✗       │     ✓                  │
│ Exp-6    │     ✗         │     ✓       │     ✓                  │
│ Exp-7    │     ✓         │     ✓       │     ✓ (完整知识)        │
└──────────┴──────────────┴─────────────┴───────────────────────┘
```

#### 配置修改方式

修改 `gbm_generation.yaml` 中的知识库路径和开关：

```yaml
# Exp-0: 完全关闭知识注入
gbm_targets_path: null
gbm_clinical_path: null
gbm_molecules_path: null

# Exp-7: 完整知识注入
gbm_targets_path: gbm_project/data/gbm_targets/gbm_targets.json
gbm_clinical_path: gbm_project/data/gbm_clinical/gbm_clinical_data.json
gbm_molecules_path: gbm_project/data/gbm_molecules/gbm_molecules.json
```

#### 评估指标

- **BBB穿透性提升率**：`ΔBBB = BBB_score(Exp-7) - BBB_score(Exp-0)`
- **GBM活性提升率**：`ΔActivity = Activity(Exp-7) - Activity(Exp-0)`
- **综合得分提升率**：基于加权综合得分的对比
- **知识贡献度分解**：使用 Sobol' 指数分解各子知识库的贡献

#### 预期假设

- 完整知识注入（Exp-7）相比基线（Exp-0）应显著提升 BBB 穿透性和 GBM 活性
- 靶点知识（Exp-1）主要影响靶点选择性
- 临床数据（Exp-2）主要影响安全性和毒性
- 分子参考库（Exp-3）主要影响合成可行性和结构新颖性

---

### 3.2 Ablation-2：CoT/ToT推理策略的效果分析

#### 实验动机

项目支持两种推理策略：Chain-of-Thought（CoT）和 Tree-of-Thoughts（ToT）。需要对比无推理、CoT、ToT 以及 CoT+ToT 组合四种配置下的分子生成质量。

#### 推理策略说明

**Chain-of-Thought (CoT)**：
- 逐步推理，引导模型按 7 个步骤进行分子设计
- 模板：`cot_reasoning_templates.step_by_step_design`
- 包含参考药物分析 → 靶点分析 → 骨架选择 → 片段组装 → 性质预测 → 选择性检查 → 候选生成

**Tree-of-Thoughts (ToT)**：
- 树状搜索，支持广度优先策略
- 三阶段推理：骨架提议（propose_scaffold）→ 片段组装策略（propose_assembly）→ SMILES 生成（generate_smiles）
- 每阶段生成 3 个候选节点，总搜索空间为 3³ = 27

#### 实验配置

```
┌──────────────────────────────────────────────────────────────────┐
│                    Ablation-2: 推理策略对比                        │
├──────────┬─────────────────┬──────────────┬───────────────────────┤
│ 实验组    │ 推理策略         │ 启用模板      │ 搜索策略               │
├──────────┼─────────────────┼──────────────┼───────────────────────┤
│ Exp-0    │ 无推理           │     —        │ 直接生成               │
│ Exp-1    │ CoT only        │ step_by_step │ 单路径顺序推理          │
│ Exp-2    │ ToT only        │ tot_*        │ BFS树搜索 (宽度=3)      │
│ Exp-3    │ CoT + ToT       │ step_by_step │ 先CoT后ToT两阶段推理    │
│ Exp-4    │ ToT (宽度=5)    │ tot_*        │ 扩展树搜索 (宽度=5)      │
│ Exp-5    │ ToT (深度=2)    │ tot_*        │ 深度优先变体            │
└──────────┴─────────────────┴──────────────┴───────────────────────┘
```

#### 配置修改方式

```yaml
# Exp-0: 无推理
use_cot_reasoning: false
use_tot_reasoning: false

# Exp-1: CoT only
use_cot_reasoning: true
use_tot_reasoning: false
cot_template: step_by_step_design

# Exp-2: ToT only
use_cot_reasoning: false
use_tot_reasoning: true
tot_breadth: 3
tot_max_depth: 3

# Exp-3: CoT + ToT
use_cot_reasoning: true
use_tot_reasoning: true
cot_template: step_by_step_design
tot_breadth: 3
```

#### 评估指标

- **生成成功率**：`Valid_SMILES / Total_Generated`
- **分子多样性**：Tanimoto 相似度分布（使用 Morgan fingerprint, radius=2）
- **综合得分分布**：各得分区间的分子数量对比
- **推理深度有效性**：ToT 中 "sure" 评价节点的比例
- **计算效率比**：`Quality_Score / Generation_Time`

---

### 3.3 Ablation-3：靶点特异性优化分析

#### 实验动机

项目覆盖 8 种 GBM 相关靶点（EGFR、VEGF/VEGFR、IDH1/2、MGMT、GBM干性细胞、PD-1/PD-L1、PI3K/AKT/mTOR、p53/MDM2）。不同靶点对分子设计的要求差异显著，需要分析靶点特异性知识对各靶点生成效果的影响。

#### 实验配置

```
┌────────────────────────────────────────────────────────────────────┐
│                    Ablation-3: 靶点特异性优化                        │
├──────────┬──────────────────┬──────────────┬────────────────────────┤
│ 实验组    │ 靶点              │ 权重配置     │ 靶点知识                │
├──────────┼──────────────────┼──────────────┼────────────────────────┤
│ Exp-0    │ 无指定靶点        │   均匀权重    │    ✗                   │
│ Exp-1    │ EGFR              │   EGFR=1.0   │    ✓                   │
│ Exp-2    │ VEGF/VEGFR        │   VEGF=1.0  │    ✓                   │
│ Exp-3    │ IDH1/IDH2         │   IDH=1.0    │    ✓                   │
│ Exp-4    │ MGMT              │   MGMT=1.0   │    ✓                   │
│ Exp-5    │ GBM_Stem_Cells    │   GSC=1.0   │    ✓                   │
│ Exp-6    │ PD-1/PD-L1        │  PD1=1.0    │    ✓                   │
│ Exp-7    │ PI3K/AKT/mTOR     │  PI3K=1.0   │    ✓                   │
│ Exp-8    │ p53/MDM2          │  p53=1.0    │    ✓                   │
│ Exp-9    │ 全部靶点          │   按配置权重  │    ✓                   │
└──────────┴──────────────────┴──────────────┴────────────────────────┘
```

#### 靶点知识差异

每个靶点的专业知识模板定义了不同的设计要求：

| 靶点 | 关键药效团 | MW范围 | LogP范围 | 特殊要求 |
|------|-----------|--------|----------|---------|
| EGFR | 喹唑啉母核 + 丙烯酰胺弹头 | 400-550 | 2-4 | EGFRvIII突变选择性 |
| VEGF/VEGFR | VEGFR抑制剂药效团 | 450-600 | 1-3 | 抗血管生成 |
| IDH1/IDH2 | IDH抑制剂结构 | 400-550 | 2-4 | R132H/R172K选择性 |
| MGMT | DNA损伤增强剂 | 350-500 | 1-3 | 绕过MGMT修复 |
| GBM_Stem_Cells | 干性分化诱导剂 | 400-550 | 2-5 | CD133/SOX2靶向 |
| PD-1/PD-L1 | 免疫检查点抑制剂 | 500-700 | 2-5 | 免疫激活 |
| PI3K/AKT/mTOR | PI3K抑制剂骨架 | 400-600 | 2-4 | 多通路抑制 |
| p53/MDM2 | MDM2拮抗剂 | 400-600 | 2-5 | p53激活 |

#### 评估指标

- **靶点命中率**：生成的分子中符合靶点药效团要求的比例
- **靶点特异性得分**：针对特定靶点的预测结合亲和力
- **骨架多样性**：各靶点生成的骨架类型分布
- **跨靶点泛化能力**：无靶点指定时各靶点的平均命中率

---

### 3.4 Ablation-4：分子约束条件的梯度实验

#### 实验动机

分子约束条件（MW、LogP、TPSA、HBD、HBA、可旋转键数等）是控制药物类药性的关键。需要分析不同约束强度对生成质量和多样性的影响。

#### 约束梯度设计

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Ablation-4: 约束条件梯度                          │
├──────────┬──────────────┬──────────────┬──────────────┬──────────────┤
│ 实验组    │  MW范围       │  LogP范围    │  TPSA上限    │  HBD/HBA    │
├──────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Exp-0    │   无限制      │   无限制     │   无限制     │   无限制     │
│ Exp-1    │  200-800      │  -2 ~ 6      │   200        │   ≤8/≤12    │
│ Exp-2    │  300-600      │  0 ~ 5       │   150        │   ≤6/≤10    │
│ Exp-3    │  300-500      │  1 ~ 4       │   120        │   ≤5/≤8     │
│ Exp-4    │  350-500      │  2 ~ 4       │   100        │   ≤3/≤6     │  ← BBB优化
│ Exp-5    │  300-500      │  1.5~3.5     │   90         │   ≤2/≤5     │  ← 最严格BBB
│ Exp-6    │  自适应       │  自适应      │   自适应     │   自适应     │  ← ToT引导
└──────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

#### 配置修改方式

```yaml
# Exp-4: BBB优化约束
molecular_constraints:
  molecular_weight_max: 500
  molecular_weight_min: 350
  logp_min: 2.0
  logp_max: 4.0
  tpsa_max: 100
  tpsa_min: 60
  hbd_max: 3
  hbd_min: 1
  hba_max: 6
  hba_min: 3
  rotatable_bonds_max: 6
  bbb_permeability_threshold: -0.5

# Exp-5: 最严格BBB约束
molecular_constraints:
  molecular_weight_max: 500
  molecular_weight_min: 300
  logp_min: 1.5
  logp_max: 3.5
  tpsa_max: 90
  hbd_max: 2
  hba_max: 5
  bbb_permeability_threshold: 0.0
```

#### 评估指标

- **约束满足率**：`Molecules_Meeting_All_Constraints / Total_Valid_Molecules`
- **BBB高分分子比例**：logBB > -0.5 的分子占比
- **类药性通过率**：满足 Lipinski rule of 5 的比例
- **约束-多样性权衡曲线**：约束严格程度 vs 分子多样性散点图
- **最优约束区间发现**：通过 Exp-6 自适应约束发现的最佳参数区间

---

### 3.5 Ablation-5：LoRA微调适配器的贡献

#### 实验动机

项目训练了专门的 LoRA adapter（`Llamole-Qwen2-7B-Instruct-Adapter-gbm-with-graph-models`）。需要对比使用和不使用 adapter 的生成效果，量化微调对领域适应的贡献。

#### 实验配置

```
┌────────────────────────────────────────────────────────────────────┐
│                    Ablation-5: LoRA微调效果                        │
├──────────┬──────────────────────┬──────────────┬───────────────────┤
│ 实验组    │ 模型配置             │  Adapter     │ Graph LM          │
├──────────┼──────────────────────┼──────────────┼───────────────────┤
│ Exp-0    │ Qwen2-7B-Instruct    │     ✗        │        ✗          │
│ Exp-1    │ Qwen2-7B-Instruct    │     ✓        │        ✗          │
│ Exp-2    │ Qwen2-7B-Instruct    │     ✗        │        ✓          │
│ Exp-3    │ Qwen2-7B-Instruct    │     ✓        │        ✓ (完整)    │
└──────────┴──────────────────────┴──────────────┴───────────────────┘
```

#### 配置修改方式

```yaml
# Exp-0: 基础模型（无任何适配）
adapter_name_or_path: null
graph_lm_connector_path: null
graph_encoder_path: null
graph_predictor_path: null

# Exp-3: 完整适配（项目默认配置）
adapter_name_or_path: saves/Llamole-Qwen2-7B-Instruct-Adapter-gbm-with-graph-models
graph_lm_connector_path: saves/Llamole-Qwen2-7B-Instruct-Adapter/connector
graph_encoder_path: saves/graph_encoder
graph_predictor_path: saves/graph_predictor
```

#### 评估指标

- **任务适应度提升**：`Adaptation_Score = f(Exp-3) - f(Exp-0)`
- **分子语法正确率**：生成有效 SMILES 的比例
- **领域术语使用率**：生成结果中 GBM 领域专业词汇的出现频率
- **图结构解码质量**：Graph Decoder 输出的分子图与文本描述的一致性
- **分子性质预测准确率**：Graph Predictor 对生成分子的活性预测分布

---

### 3.6 Ablation-6：Prompt模板变体效果对比

#### 实验动机

项目支持多种 prompt 模板（基础模板、靶点特异性模板、CoT 模板、ToT 模板、评估模板等）。需要对比不同模板组合对生成质量的影响，找出最优模板配置。

#### 实验配置

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Ablation-6: Prompt模板变体                        │
├──────────┬──────────────────────────┬──────────────┬─────────────────┤
│ 实验组    │ Base Prompt              │ CoT模板      │ 约束模板         │
├──────────┼──────────────────────────┼──────────────┼─────────────────┤
│ Exp-0    │ 通用分子生成prompt        │     —        │      —          │
│ Exp-1    │ base_gbm_prompt          │     —        │      —          │
│ Exp-2    │ base_gbm_prompt          │ step_by_step │      —          │
│ Exp-3    │ base_gbm_prompt          │ step_by_step │ bbb_optimization │
│ Exp-4    │ base_gbm_prompt          │ step_by_step │ selectivity_design│
│ Exp-5    │ base_gbm_prompt          │ step_by_step │ resistance_evasion│
│ Exp-6    │ target_specific_prompts  │ step_by_step │      —          │
│ Exp-7    │ base_gbm_prompt          │ clinical_relevance |      —    │
│ Exp-8    │ base_gbm_prompt          │ multi-objective | multi-constraint │
└──────────┴──────────────────────────┴──────────────┴─────────────────┘
```

#### 各模板说明

| 模板名称 | 文件位置 | 主要内容 | 适用场景 |
|---------|---------|---------|---------|
| `base_gbm_prompt` | gbm_prompts.yaml:3 | GBM生物学特性、治疗挑战、设计要求 | 通用GBM分子生成 |
| `target_specific_prompts` | gbm_prompts.yaml:23 | EGFR/VEGF/IDH等8个靶点的专门设计要求 | 靶点导向设计 |
| `step_by_step_design` | gbm_prompts.yaml:120 | 7步CoT推理流程，含参考药物分析 | 精确控制生成过程 |
| `clinical_relevance_assessment` | gbm_prompts.yaml:170 | 临床转化潜力评估 | 关注临床可行性 |
| `bbb_optimization` | gbm_prompts.yaml:192 | BBB穿透性优化约束 | BBB优先设计 |
| `selectivity_design` | gbm_prompts.yaml:200 | GBM选择性设计原则 | 选择性优先设计 |
| `resistance_evasion` | gbm_prompts.yaml:207 | 耐药性规避策略 | 耐药肿瘤设计 |

#### 配置修改方式

```python
# 在 gbm_prompt_generator.py 中切换模板
prompt_config = {
    # Exp-6: 靶点特异性prompt
    'base_prompt': 'target_specific_prompts',
    'cot_template': 'step_by_step_design',
    'constraint_template': None,
    'target_override': 'EGFR'  # 指定靶点
}

# Exp-8: 多目标优化prompt
prompt_config = {
    'base_prompt': 'base_gbm_prompt',
    'cot_template': 'multi_objective_optimization',
    'constraint_template': 'multi_constraint_optimization'
}
```

#### 评估指标

- **Prompt有效性评分**：PE（Prompt Effectiveness）= 生成质量 / Prompt长度
- **模板组合增益**：特定模板组合相比基础模板的提升百分比
- **模板-靶点匹配度**：特定模板对特定靶点的适用性评分
- **Token消耗效率比**：`Quality_Score / Total_Tokens_Used`

---

## 4. 实验执行计划

### 4.1 实验执行顺序

```
Phase 1: 基础消融 (快速验证)
├── Ablation-5 (LoRA Adapter) → 确定是否使用微调模型
├── Ablation-1 (领域知识) → 确定知识注入的价值
└── Ablation-2 (推理策略) → 确定推理策略

Phase 2: 精细消融 (参数优化)
├── Ablation-4 (约束梯度) → 找到最优约束区间
├── Ablation-6 (Prompt变体) → 找到最优模板组合
└── Ablation-3 (靶点特异性) → 各靶点的最优配置

Phase 3: 组合验证 (最终验证)
├── 最优配置组合验证
├── 统计显著性检验 (n=3, 每次生成≥20个分子)
└── 生成最终消融报告
```

### 4.2 实验参数标准

```
通用实验参数:
├── 每组生成候选分子数: 20
├── 每组重复次数: 3次
├── 随机种子: 42, 43, 44
├── 生成温度: 0.7
├── Top-p: 0.9
└── 最大token数: 256

统计参数:
├── 显著性水平: α = 0.05
├── 多重比较校正: Bonferroni校正
└── 效应量报告: Cohen's d
```

### 4.3 实验执行脚本

```bash
# 执行 Ablation-1: 领域知识注入
cd gbm_project/experiments
python benchmark_experiments.py \
    --base_config ../configs/gbm_generation.yaml \
    --output_dir ./ablation_results/ablation-1_domain_knowledge \
    --test_scenarios baseline gbm_knowledge cot_reasoning combined \
    --num_candidates_per_test 20 \
    --repeat_times 3

# 执行 Ablation-2: 推理策略
python benchmark_experiments.py \
    --base_config ../configs/gbm_generation.yaml \
    --output_dir ./ablation_results/ablation-2_reasoning \
    --test_scenarios baseline cot_reasoning tot_reasoning cot_tot_combined \
    --num_candidates_per_test 20 \
    --repeat_times 3

# 执行 Ablation-3: 靶点特异性
cd ../scripts
python batch_generate_all_targets.py \
    --config ../configs/gbm_generation.yaml \
    --output_dir ./ablation_results/ablation-3_targets \
    --targets EGFR VEGF IDH1_IDH2 MGMT GBM_Stem_Cells \
    --num_candidates_per_target 20

# 执行 Ablation-4: 约束梯度
python -c "
from gbm_generator import GBMGenerator
import yaml

constraints_variants = [
    {'name': 'exp0_unconstrained', 'constraints': {}},
    {'name': 'exp3_standard', 'mw_range': (300,500), 'logp_range': (1,4)},
    {'name': 'exp4_bbb_optimized', 'mw_range': (350,500), 'logp_range': (2,4), 'tpsa_max': 100},
    {'name': 'exp5_strict_bbb', 'mw_range': (300,500), 'logp_range': (1.5,3.5), 'tpsa_max': 90}
]

for variant in constraints_variants:
    gen = GBMGenerator('../configs/gbm_generation.yaml')
    gen.update_constraints(variant.get('constraints', {}))
    results = gen.generate_gbm_molecules(num_candidates=20)
    gen.save_results(results, f'./ablation_results/ablation-4_constraints/{variant[\"name\"]}.json')
"
```

---

## 5. 评估与分析框架

### 5.1 核心评估指标体系

```
┌─────────────────────────────────────────────────────────────────┐
│                    多维度评估指标体系                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  维度1: 分子有效性                                               │
│  ├── 语法有效率 = Valid_SMILES / Total_Generated                │
│  ├── 结构有效性 = Cananonical_SMILES / Valid_SMILES             │
│  └── 约束满足率 = Meeting_All_Constraints / Valid_SMILES         │
│                                                                 │
│  维度2: 药代动力学潜力                                           │
│  ├── BBB穿透性 = logBB预测值                                     │
│  ├── BBB分类准确率 = (high + medium_correct) / Total            │
│  ├── 口服吸收潜力 = TPSA + LogP 组合评分                         │
│  └── 血脑屏障选择性 = BBB_score / Peripheral_Toxicity            │
│                                                                 │
│  维度3: GBM治疗效果                                             │
│  ├── GBM细胞活性 = pIC50预测值                                   │
│  ├── 靶点结合亲和力 = Ki/Kd预测值                                 │
│  ├── 选择性指数 = GBM_activity / Normal_cell_toxicity           │
│  └── 干细胞杀伤效率 = GSC_inhibition_rate                        │
│                                                                 │
│  维度4: 安全性                                                  │
│  ├── 正常细胞毒性 = CC50预测值                                   │
│  ├── 选择性窗口 = CC50 / IC50                                    │
│  ├── 预期不良反应风险 = 毒性预测评分                              │
│  └── 代谢稳定性 = metabolic_stability_score                     │
│                                                                 │
│  维度5: 可开发性                                                │
│  ├── 合成可行性 = SA_score (1-10)                                │
│  ├── 专利风险 = novelty_score                                   │
│  ├── 与已知药物的相似度 = Tanimoto_to_drugs                     │
│  └── 骨架新颖性 = Scaffold_novelty                              │
│                                                                 │
│  综合得分 = 0.25×BBB + 0.25×GBM活性 + 0.20×安全性 +             │
│            0.15×合成可行性 + 0.15×临床相似度                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 统计分析方法

| 分析方法 | 应用场景 | 具体指标 |
|---------|---------|---------|
| 描述性统计 | 所有实验组 | 均值±标准差、中位数、四分位数 |
| t检验 | 两组比较 | p值、Cohen's d效应量 |
| ANOVA | 多组比较 | F统计量、η²效应量 |
| Tukey HSD | 多组两两比较 | 调整后p值 |
| Bonferroni校正 | 多重比较 | 校正后α |
| 相关性分析 | 约束与质量关系 | Pearson/Spearman相关系数 |
| 主成分分析 | 多维度可视化 | PC1/PC2解释方差 |
| 聚类分析 | 分子分组 | K-means silhouette score |

### 5.3 结果可视化设计

```
实验结果可视化矩阵:

         ┌─────────┬─────────┬─────────┬─────────┬─────────┐
         │ 雷达图   │ 箱线图   │ 热力图   │ 散点图   │ 柱状图   │
─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
Ablation-1│ 8维度雷达 │ 各子知识库 │ 知识×靶点 │ 约束满足 │ 知识贡献度 │
Ablation-2│ 推理策略  │ 策略得分  │ ToT树可视化│ 质量vs效率│ CoT vs ToT │
Ablation-3│ 靶点性能  │ 8靶点分布 │ 靶点×骨架 │ MW vs BBB│ 靶点排名  │
Ablation-4│ 约束效果  │ 约束梯度  │ 约束相关性 │ 多样性曲线│ 最优区间  │
Ablation-5│ Adapter  │ Adapter  │ 参数热力图│ 适应度曲线│ 贡献度排名 │
Ablation-6│ Prompt  │ Prompt   │ 模板匹配  │ Token效率│ 模板排名  │
         └─────────┴─────────┴─────────┴─────────┴─────────┘
```

---

## 6. 预期结论与假设

### 6.1 各维度预期结论

#### Ablation-1：领域知识注入

| 假设 | 预期结论 | 置信度 |
|------|---------|-------|
| H1.1 | 完整领域知识（Exp-7）相比无知识（Exp-0）综合得分提升 **30-50%** | 高 |
| H1.2 | 靶点知识主要提升 **靶点命中率**（↑40%） | 高 |
| H1.3 | 临床数据主要降低 **正常细胞毒性**（↓25%） | 中 |
| H1.4 | 分子参考库主要提升 **合成可行性**（↑20%） | 中 |
| H1.5 | 各子知识库的贡献度排序：靶点知识 > 临床数据 > 分子参考 | 中 |

#### Ablation-2：推理策略

| 假设 | 预期结论 | 置信度 |
|------|---------|-------|
| H2.1 | CoT+ToT（Exp-3）综合得分最高，相比无推理（Exp-0）提升 **25-40%** | 高 |
| H2.2 | ToT（Exp-2）生成 **更高多样性** 的分子骨架 | 高 |
| H2.3 | CoT（Exp-1）生成 **更高BBB穿透性** 的分子（步骤引导更精确） | 中 |
| H2.4 | ToT宽度增加（Exp-4）提升质量但 **增加计算成本 1.7倍** | 高 |
| H2.5 | CoT+ToT组合在 **GBM活性** 指标上提升最显著 | 高 |

#### Ablation-3：靶点特异性

| 假设 | 预期结论 | 置信度 |
|------|---------|-------|
| H3.1 | EGFR靶点生成质量最高（知识库最完善） | 高 |
| H3.2 | IDH1/2靶点生成质量最低（突变特异性要求高） | 中 |
| H3.3 | 靶点特异性prompt相比通用prompt提升 **35%** 的靶点命中率 | 高 |
| H3.4 | 多靶点组合（Exp-9）生成更均衡的分子 | 中 |

#### Ablation-4：约束条件

| 假设 | 预期结论 | 置信度 |
|------|---------|-------|
| H4.1 | BBB优化约束（Exp-4）使BBB-high分子比例从 30% 提升到 **55%** | 高 |
| H4.2 | 最严格约束（Exp-5）降低 **有效生成率** 到约 50% | 高 |
| H4.3 | 约束满足率与综合得分呈 **倒U型关系**（过严约束损害质量） | 中 |
| H4.4 | 自适应约束（Exp-6）找到最优区间：MW 350-500，LogP 2-3.5 | 中 |

#### Ablation-5：LoRA微调

| 假设 | 预期结论 | 置信度 |
|------|---------|-------|
| H5.1 | LoRA+Graph LM（Exp-3）相比基础模型（Exp-0）语法有效率提升 **20%** | 高 |
| H5.2 | Graph LM 主要提升 **分子性质预测** 的准确性 | 高 |
| H5.3 | LoRA adapter 使领域术语使用率从 15% 提升到 **60%** | 高 |
| H5.4 | 无adapter时（Exp-0）生成 **更多新颖但无效** 的分子 | 中 |

#### Ablation-6：Prompt模板

| 假设 | 预期结论 | 置信度 |
|------|---------|-------|
| H6.1 | base_gbm_prompt（Exp-1）相比通用prompt（Exp-0）综合得分提升 **20%** | 高 |
| H6.2 | CoT模板（Exp-2）相比无CoT（Exp-1）BBB得分提升 **15%** | 中 |
| H6.3 | bbb_optimization约束（Exp-3）使BBB-high分子比例额外提升 **10%** | 中 |
| H6.4 | target_specific_prompts（Exp-6）在靶点指定时效果最佳 | 高 |
| H6.5 | resistance_evasion（Exp-5）主要提升 **耐药肿瘤** 的分子设计 | 低（待验证） |

### 6.2 关键交互效应假设

```
消融维度交互效应分析:

领域知识 × 推理策略:
  - GBM知识 + CoT+ToT: 协同效应最强 (预期 > 各自提升之和)
  - GBM知识 × ToT: ToT在有领域知识时多样性提升更显著

约束条件 × 靶点:
  - EGFR靶点 + BBB优化约束: 最优组合（EGFR抑制剂天然BBB通透）
  - PD-1/PD-L1 + 严格BBB约束: 冲突最大（抗体类药物分子量大）

LoRA × 领域知识:
  - 有adapter时领域知识提升更显著（adapter强化了知识利用能力）
  - 无adapter时领域知识提升有限（基础模型无法充分利用知识）

Prompt模板 × 靶点:
  - EGFR + EGFR特异性prompt: 最优匹配
  - IDH1/2 + base_gbm_prompt: 比特异性prompt效果更好（突变知识更关键）
```

---

## 7. 实验记录模板

### 7.1 单次实验记录

```markdown
# 消融实验记录

## 实验基本信息
- 实验编号: Ablation-[维度]-[编号]
- 实验日期: YYYY-MM-DD
- 实验人员: [姓名]
- 实验环境: Python 3.11, PyTorch, RDKit

## 实验配置
- 基线配置: gbm_generation.yaml
- 修改配置:
  - gbm_generation_mode: [true/false]
  - use_gbm_domain_knowledge: [true/false]
  - use_cot_reasoning: [true/false]
  - target: [靶点名或null]
  - constraints: {...}

## 生成的分子
- 分子数量: N
- 有效分子数: M
- 有效率: M/N (%)

## 评估结果
| 指标 | 均值 | 标准差 | 中位数 |
|------|------|--------|--------|
| 综合得分 | 0.xxx | ±0.xx | 0.xxx |
| BBB得分 | 0.xxx | ±0.xx | 0.xxx |
| GBM活性 | 0.xxx | ±0.xx | 0.xxx |
| 毒性评分 | 0.xxx | ±0.xx | 0.xxx |
| 合成可行性 | 0.xxx | ±0.xx | 0.xxx |
| 临床相似度 | 0.xxx | ±0.xx | 0.xxx |

## 观察与发现
[记录实验中的异常、意外发现等]

## 结论
[该实验组的主要结论]
```

---

## 8. 总结

### 8.1 消融实验价值

本消融实验设计覆盖了 GBM 分子生成项目的 **6 大核心维度**，通过系统性、层次化的实验设计，可以：

1. **量化每个模块的贡献**：明确各组件对最终生成质量的贡献度
2. **发现最优配置组合**：通过交互效应分析找到最佳模块组合
3. **指导工程优化方向**：优先优化贡献度高的模块
4. **验证方法论假设**：用实验数据验证项目设计中的核心假设
5. **发表高质量消融论文**：完整的消融实验是方法论论文的必要组成部分

### 8.2 实验优先级建议

| 优先级 | 消融维度 | 建议理由 |
|-------|---------|---------|
| P0 | Ablation-5 (LoRA) | 基础模型选择，影响所有后续实验 |
| P0 | Ablation-1 (领域知识) | 核心创新点，决定项目价值 |
| P1 | Ablation-2 (推理策略) | 高性价比优化，效果显著 |
| P1 | Ablation-4 (约束条件) | 直接影响BBB优化目标 |
| P2 | Ablation-6 (Prompt模板) | 锦上添花式优化 |
| P2 | Ablation-3 (靶点特异性) | 特定靶点优化，价值相对局限 |

### 8.3 预期实验周期

| 阶段 | 实验内容 | 预估时间 |
|------|---------|---------|
| Phase 1 | 基础消融（Ablation-5, 1, 2） | 2-3天 |
| Phase 2 | 精细消融（Ablation-4, 6, 3） | 3-4天 |
| Phase 3 | 组合验证与统计分析 | 1-2天 |
| **总计** | **完整消融实验** | **6-9天** |

---

*文档版本: v1.0*
*创建日期: 2026-03-23*
*维护者: GBM药物设计研究组*
