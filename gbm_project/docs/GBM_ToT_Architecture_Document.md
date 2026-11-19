# GBM ToT 分子生成系统架构文档

## 1. 系统概述

本系统基于 Tree-of-Thoughts（ToT）搜索策略，结合领域知识注入和物理评估反馈，为胶质母细胞瘤（GBM）靶点自动设计候选药物分子。

### 1.1 核心流程

```
用户指定靶点（如 EGFR）
        ↓
TreeOfThoughtsGenerator.generate_molecules(target_name)
        ↓
┌──────────────────────────────────────────────────────────────┐
│  Step 1: GBMPromptGenerator.generate_domain_prompt()          │
│  → 从 GBMKnowledgeBase 检索靶点知识 + 临床数据 + 分子库      │
│  → 生成领域上下文 prompt                                    │
└──────────────────────────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────────────────────────┐
│  Step 2: BFS ToT Search (TreeOfThoughtsGenerator)           │
│                                                              │
│  Level 0: Scaffold Proposal（骨架提议）                      │
│      → ToT 调用 LLM 生成 k 个候选骨架                       │
│      → ToT 调用 GBMPhysicalEvaluator 评估，保留 b 个最优分支  │
│                                                              │
│  Level 1: Assembly Strategy（组装策略）                       │
│      → ToT 调用 LLM 生成 warhead + BBB enhancers            │
│      → ToT 调用 GBMPhysicalEvaluator 评估，保留 b 个最优分支  │
│                                                              │
│  Level 2: SMILES Generation（SMILES 生成）                    │
│      → ToT 调用 LLM 生成完整 SMILES 字符串                   │
│      → 多轮物理反馈优化：                                    │
│        SMILES → ToT 调用物理评估 → 反馈 → 重调用 LLM         │
│      → ToT 调用 RDKit 化学合法性校验                        │
│      → ToT 调用 LLM Self-Correction（无效 SMILES 自动修正）  │
│      → ToT 硬剪枝（is_pruned=True 的分支直接丢弃）          │
│                                                              │
│  输出: final_smiles_nodes（多个通过物理评估的 SMILES）        │
└──────────────────────────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────────────────────────┐
│  Step 3: 结果输出                                            │
│  → 过滤无效 SMILES，计算 RDKit 描述符（TPSA/MW/LogP）      │
│  → 保存完整 ToT 路径信息（含每层的评估结论）                │
│  → checkpoint 实时保存，防止中途丢数据                      │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. 核心组件

### 2.0 微调模型（LLM）—— 被调用的生成引擎

模型加载分两阶段，最终以 PeftModel 形式存在，供 ToT 模块随时调用：

```
Qwen2-7B-Instruct（基座模型，FP16）
        ↓
┌─────────────────────────────────────────────┐
│ Stage 1: Llamole Adapter                    │
│ 路径: saves/Llamole-Qwen2-7B-Instruct-Adapter │
│ 类型: LoRA → merge_and_unload()             │
│ 作用: 通用指令遵循能力                       │
└─────────────────────────────────────────────┘
        ↓ 合并后的基座模型
┌─────────────────────────────────────────────┐
│ Stage 2: GBM Domain Adapter                 │
│ 路径: saves/Llamole-Qwen2-7B-Instruct-     │
│        Adapter-gbm-with-graph-models         │
│ 类型: LoRA → PeftModel（保留，未合并）       │
│ 作用: GBM 领域专业知识 + 分子生成能力         │
└─────────────────────────────────────────────┘
        ↓
最终模型（PeftModel，供 TreeOfThoughtsGenerator 调用）
        ↓
特殊场景: SMILES 生成 fallback 时，可临时切换回 Stage 1 合并模型 + few-shot
```

**重要说明：** LLM 在整个系统中是 **被动角色** ——它不主动运行，而是被 ToT 模块在需要时调用。ToT 模块在每个层级（Scaffold/Assembly/SMILES）都决定"何时调用 LLM"、"传什么 prompt"、"收到输出后如何处理"。

### 2.1 GBMKnowledgeBase（知识库）

负责存储和检索 GBM 相关领域知识，包含三类数据：

| 数据类型 | 来源文件 | 内容 |
|---------|---------|------|
| **靶点信息** | `gbm_targets.json` | 靶点名称、作用机制、突变类型、现有药物、设计挑战、结构要求 |
| **临床数据** | `gbm_clinical_data.json` | 标准治疗方案、治疗挑战、失败试验洞察、成功模式、分子亚型、生物标志物、耐药机制 |
| **分子库** | `gbm_molecules.json` | 已批准药物（gbm_drugs）、临床候选物（clinical_candidates）、失败案例（failed_molecules） |

**使用方式：** `GBMPromptGenerator` 在生成领域 prompt 时调用 `kb.get_target_info()` 检索靶点详情，作为上下文注入 LLM。

### 2.2 GBMPromptGenerator（提示词生成器）

核心职责：从知识库提取信息，构建结构化 prompt。

**关键方法：**

- `generate_domain_prompt(target_name, constraints)` — 生成领域上下文，包含靶点信息、临床洞察、约束条件、结构知识注入
- `build_tot_propose_prompt(domain_prompt, state, step_type)` — 为 ToT 三个层级分别构建 prompt：
  - `step_type="scaffold"` → Level 0 骨架提议 prompt
  - `step_type="assembly"` → Level 1 组装策略 prompt
  - `step_type="smiles"` → Level 2 SMILES 生成 prompt
- `build_tot_propose_prompt_with_feedback(...)` — 在 SMILES 多轮优化时，注入物理评估反馈（Vina/DILI/BBB 分数）

**注意：** GBMPromptGenerator **只负责构建 prompt**，不参与 ToT 搜索逻辑。

### 2.3 GBMPhysicalEvaluator（物理评估器）

采用"外科手术式"设计——纯计算模块，不涉及任何 LLM 调用。评估指标如下：

| 评估维度 | 方法 | 范围 |
|---------|------|------|
| **Vina 对接分数** | QVina/Wina 命令行对接 | kcal/mol（越负越好，< -7.0 为强结合） |
| **DILI 风险** | 结构警报（SA）打分 | 概率 [0, 1]（越低越好） |
| **hERG 阻断风险** | 结构警报打分 | 概率 [0, 1]（越低越好） |
| **BBB 通透性** | Clark logBB + 启发式惩罚 | 评分 [0, 1]（越高越好，GBM 要求 > 0.7） |
| **RDKit 描述符** | TPSA、MW、LogP、HBD、HBA 等 | 本地计算 |

**输出：** `PhysicalEvaluationResult`，包含原始数值、归一化分数 [0, 1]、加权几何平均 Reward、以及 verdict（SURE/LIKELY/IMPOSSIBLE）。

**关键特性：**
- 硬剪枝：命中硬截断规则（如 TPSA 不在 40-120 范围）的分支直接标记 `is_pruned=True`，不会进入下一轮
- 自然语言反馈：`build_feedback_text()` 将物理分数转为文本，注入 LLM prompt 指导下一轮生成

### 2.4 TreeOfThoughtsGenerator（ToT 搜索引擎）

核心引擎，执行 BFS 搜索和物理反馈循环。**ToT 模块是控制流，主动调用 LLM；LLM 是被动生成器，只负责根据 prompt 返回文本。**

**LLM 调用关系（每个层级都遵循此模式）：**

```
ToT 模块                           LLM（微调模型）
   │                                    │
   ├── 调用 GBMPromptGenerator ──────────→│
   │   生成领域 + 结构化 prompt          │
   │←───────────────────────────────────┤
   │                                    │
   ├── 调用 LLM.generate_with_model() ──→│  ← 传入 prompt，LLM 返回文本
   │←───────────────────────────────────┤
   │                                    │
   ├── 解析 LLM 输出                    │
   ├── 调用 GBMPhysicalEvaluator 评估    │
   ├── 剪枝 / 排序 / 保留 b 个分支      │
   └── 决定是否继续下一层 / 下一轮       │
```

**初始化参数：**

| 参数 | 默认值 | 含义 |
|-----|--------|------|
| `k` | 3 | 每层生成的候选数 |
| `b` | 2 | BFS 保留的最优分支数 |
| `depth` | 3 | ToT 搜索深度（固定为 3：Scaffold → Assembly → SMILES） |
| `tot_refinement_rounds` | 1 | SMILES 多轮优化轮数 |
| `tot_good_reward_threshold` | 0.65 | 早停 Reward 阈值 |
| `tot_max_llm_calls_per_search` | 50 | 单次搜索最大 LLM 调用次数 |
| `tot_max_search_seconds` | 300 | 单次搜索最大运行时间（秒） |

**三层 BFS 搜索详解：**

```
Level 0: Scaffold Proposal
├── 输入: 领域 prompt + 约束条件（MW 300-500、BBB 高、LogP 2.0-4.0、TPSA 40-120）
├── ToT 调用 LLM 生成 k 个骨架候选（quinazoline、pyridine、pyrimidine 等）
├── ToT 调用 GBMPhysicalEvaluator 评估（启发式 + RDKit TPSA）
├── 剪枝 impossible 分支，按 sure > likely 排序，保留 b 个
└── fallback: 若全为 impossible，使用 3 个硬编码骨架兜底

Level 1: Assembly Strategy
├── 输入: 选中的骨架 + 剩余 MW 预算
├── ToT 调用 LLM 生成 k 个组装策略（warhead 类型 + BBB 增强基团）
├── 示例: Warhead: acrylamide, BBB Enhancers: fluorine, methoxy
├── ToT 调用 GBMPhysicalEvaluator 评估并保留 b 个最优策略
└── fallback: 若全为 impossible，使用硬编码 EGFR 风格策略

Level 2: SMILES Generation（多轮优化循环）
├── 输入: 骨架 + 组装策略
├── for round in range(tot_refinement_rounds):
│   ├── ToT 调用 LLM 生成 SMILES 字符串
│   ├── ToT 解析 + 调用 RDKit 化学合法性校验
│   ├── 失败 → ToT 调用 LLM Self-Correction（自动修正 SMILES 语法错误）
│   ├── ToT 调用 GBMPhysicalEvaluator 全量评估（Vina/DILI/BBB）
│   ├── 硬剪枝: is_pruned=True → 丢弃
│   ├── 若 round > 0 → 注入上一轮最佳分子的物理反馈到 prompt
│   └── 早停: 若 best_reward >= tot_good_reward_threshold → 退出
└── 输出: 通过所有物理评估的 SMILES 列表
```

**Guardrail 机制：**

| 机制 | 触发条件 | 处理方式 |
|-----|---------|---------|
| Self-Correction | 解析 scaffold/assembly/smiles 格式失败 | 降低 temperature + 前车之鉴，重新调用 LLM |
| SMILES Fallback | 微调模型无有效输出 | 切换 base_merged 模型 + few-shot prompt |
| Hardcoded Fallback | 所有分支 evaluation 为 impossible | 使用 3 个硬编码骨架/策略保底 |
| Early Stopping | 连续 N 次无新增 SMILES | 停止当前靶点搜索 |

---

## 3. ToT 模块与 LLM 的调用关系

**核心原则：ToT 是控制流（主动），LLM 是生成引擎（被动）。** ToT 模块在每个层级决定"何时调用 LLM"、"传什么 prompt"、"收到输出后如何处理"。

### 3.1 典型调用模式

```
generate_molecules(target_name)
        │
        ├── ToT 调用 GBMPromptGenerator.generate_domain_prompt()
        │   → 获取领域上下文 prompt
        │
        └── ToT 执行 bfs_tot_search():
                │
                ├── Level 0: Scaffold Proposal
                │       ├── ToT 调用 GBMPromptGenerator.build_tot_propose_prompt(..., "scaffold")
                │       ├── ToT 调用 LLM.generate_with_model(scaffold_prompt)  ← LLM 返回文本
                │       ├── ToT 解析 scaffold_proposals
                │       ├── ToT 调用 GBMPhysicalEvaluator 评估
                │       ├── ToT 剪枝、排序、保留 b 个
                │       └── 若全 impossible → ToT 决定 fallback
                │
                ├── Level 1: Assembly Strategy
                │       ├── ToT 调用 GBMPromptGenerator.build_tot_propose_prompt(..., "assembly")
                │       ├── ToT 调用 LLM.generate_with_model(assembly_prompt)  ← LLM 返回文本
                │       ├── ToT 解析 assembly_strategies
                │       ├── ToT 评估、剪枝、保留 b 个
                │       └── 若全 impossible → ToT 决定 fallback
                │
                └── Level 2: SMILES Generation（多轮循环）
                        for round in range(tot_refinement_rounds):
                                ├── ToT 调用 GBMPromptGenerator.build_tot_propose_prompt(..., "smiles")
                                │   若 round > 0 → 注入上一轮物理反馈
                                ├── ToT 调用 LLM.generate_with_model(smiles_prompt)  ← LLM 返回文本
                                ├── ToT 解析 SMILES 字符串
                                ├── ToT 调用 RDKit 化学合法性校验
                                ├── 若无效 → ToT 调用 _correct_smiles_via_llm() 自修正
                                ├── ToT 调用 GBMPhysicalEvaluator 全量评估
                                ├── ToT 硬剪枝（is_pruned=True → 丢弃）
                                ├── ToT 记录 best_reward
                                └── 若 reward >= threshold → ToT 决定早停
```

### 3.2 Self-Correction 与 Guardrail

ToT 模块不仅调用 LLM，还包含多种自保护机制：

| ToT 内置机制 | 触发时机 | ToT 的决策 |
|-------------|---------|-----------|
| Self-Correction 重试 | 解析 scaffold/assembly/smiles 格式失败 | 降低 temperature + 前车之鉴，重新调用 LLM |
| SMILES Fallback | 微调模型无有效 SMILES 输出 | 切换 base_merged 模型，重新调用 LLM |
| Hardcoded Fallback | 所有分支 evaluation 为 impossible | ToT 强制注入 3 个硬编码候选，跳过 LLM 调用 |
| 早停 | 连续 N 次无新增 / reward 达标 | ToT 主动终止当前靶点搜索 |

### 3.3 物理评估反馈

```
SMILES Generation (Round N)
        ↓
ToT 调用 GBMPhysicalEvaluator.evaluate(smiles)
        ↓
┌──────────────────────────────────────────────┐
│ PhysicalEvaluationResult:                    │
│ - vina_score: -8.5 kcal/mol (强结合)        │
│ - dili_prob: 0.15 (低风险)                  │
│ - bbb_score: 0.75 (高通透性)               │
│ - reward: 0.72                              │
│ - is_pruned: False                          │
│ - verdict: SURE                              │
└──────────────────────────────────────────────┘
        ↓
若 reward < threshold 且 round < max_rounds:
        ↓
ToT 调用 _build_physical_feedback_text() → 自然语言反馈
        ↓
注入下一轮 ToT 调用 LLM.generate_with_model() 的 prompt:
"Vina = -8.5 kcal/mol (STRONG) | DILI = 0.15 (low) | BBB = 0.75 (HIGH) | Reward = 0.72"
        ↓
SMILES Generation (Round N+1) ← LLM 基于反馈优化分子
```

### 3.4 批量生成循环

```
generate_tot_batch_candidates.py
        ↓
for each target in targets:
        ↓
while len(generated) < num_molecules:
        ↓
┌──────────────────────────────────────────────┐
│ ToT 执行 bfs_tot_search():                   │
│ → ToT 控制三层 BFS + LLM 调用 + 物理评估    │
│ → 返回多个通过物理评估的 SMILES              │
└──────────────────────────────────────────────┘
        ↓
去重（基于 SMILES exact match）
        ↓
checkpoint 实时保存（每轮尝试后写入 all_results_checkpoint.json 和 CSV）
        ↓
早停检测: 连续 max_no_new_streak 次无新增 → ToT 决定退出
```

---

## 4. 约束条件与化学规则

| 约束项 | 范围 | 说明 |
|-------|------|------|
| 分子量 (MW) | 300-500 Da | 确保 BBB 通透性和代谢稳定性 |
| TPSA | 40-120 A² | 过低无 BBB 通透性，过高无法穿透细胞膜 |
| LogP | 2.0-4.0 | 亲脂性平衡：过高难以溶解，过低难以穿透脂质双分子层 |
| BBB 需求 | High | GBM 药物必须能够穿透血脑屏障 |
| hERG 风险 | 低 | 避免心脏毒性 |
| DILI 风险 | 低 | 避免药物性肝损伤 |

---

## 5. 输出结构

每个生成的分子包含以下信息：

```json
{
  "smiles": "CC(=O)Nc1ccc(-n2ccnc2)cc1C(=O)Nc3ccccc3",
  "tpsa": 89.32,
  "mw": 415.47,
  "logp": 3.21,
  "target": "EGFR",
  "generation_method": "tot_bfs_feedback",
  "tot_path": [
    {"level": 0, "content": "quinazoline", "evaluation": "sure", ...},
    {"level": 1, "content": "quinazoline + acrylamide", "evaluation": "likely", ...},
    {"level": 2, "content": "CC(=O)Nc1ccc(-n2ccnc2)cc1C(=O)Nc3ccccc3", "evaluation": "sure", ...}
  ],
  "physical_evaluation": {
    "vina_score": -8.52,
    "dili_prob": 0.15,
    "bbb_score": 0.78,
    "reward": 0.72,
    "verdict": "sure",
    "is_pruned": false
  },
  "physical_feedback": "Vina = -8.52 kcal/mol (STRONG) | DILI = 0.15 (low) | BBB = 0.78 (HIGH) | Reward = 0.72"
}
```

---

## 6. 架构图组件对应关系

| 架构图组件 | 代码实现 |
|-----------|---------|
| 知识库（靶点/临床/分子） | `GBMKnowledgeBase` + JSON 数据文件 |
| 提示词生成器 | `GBMPromptGenerator`（src 层） |
| ToT 三层搜索 | `TreeOfThoughtsGenerator.bfs_tot_search()` |
| 物理评估反馈 | `GBMPhysicalEvaluator`（外部计算模块） |
| LLM（微调） | Qwen2-7B-Instruct + Llamole Adapter（merge）+ GBM Adapter（PeftModel） |
| RDKit 化学校验 | `_basic_validate_smiles()` / `_check_rdkit_smiles()` |
| Self-Correction | `_generate_structured_with_retry()` / `_correct_smiles_via_llm()` |
