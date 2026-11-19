# 图2技术文档：ToT BFS 搜索树 (k=3, b=2)

## 1. 组件概述

`TreeOfThoughtsGenerator`（位于 `scripts/generate_tot_molecules.py`）实现 BFS（广度优先搜索）策略的 Tree-of-Thoughts 搜索，核心参数：
- `k = 3`：每层生成 3 个候选节点
- `b = 2`：每层保留 2 个最优分支
- `depth = 3`：固定 3 层（Scaffold → Assembly → SMILES）
- `tot_refinement_rounds = 1`：SMILES 多轮优化轮数（默认 1 轮）

## 2. 三层 BFS 搜索详解

### Level 0 — Scaffold Proposal（骨架提议）

**输入：** 领域 prompt（靶点知识 + 临床数据 + 约束条件：MW 300-500、BBB High、LogP 2.0-4.0、TPSA 40-120）

**处理流程：**
1. ToT 调用 `prompt_generator.build_tot_propose_prompt(..., "scaffold")` 生成骨架提议 prompt
2. ToT 调用 LLM `generate_with_model(scaffold_prompt)` 返回 k=3 个骨架候选
3. 解析输出，获取骨架名称、Base MW、BBB Potential 等元数据
4. ToT 调用 LLM 评估（SURE / LIKELY / IMPOSSIBLE），按 `"sure": 2, "likely": 1` 排序
5. 保留 b=2 个非 impossible 骨架；若全部 impossible，则使用 3 个硬编码骨架兜底

**硬编码骨架兜底列表（全部标记为 `LIKELY`）：**
- `quinazoline` (Base MW: 144 Da, BBB: high)
- `pyridine` (Base MW: 79 Da, BBB: medium-high)
- `pyrimidine` (Base MW: 80 Da, BBB: medium-high)

**输出：** b=2 个骨架节点（Level 0）

### Level 1 — Assembly Strategy（组装策略）

**输入：** 选中的骨架 + 剩余 MW 预算（target MW − scaffold Base MW）

**处理流程：**
1. ToT 调用 `prompt_generator.build_tot_propose_prompt(..., "assembly")` 生成组装策略 prompt
2. ToT 调用 LLM 返回 k=3 个组装策略，每个策略包含：
   - `warhead`：共价弹头类型（如 acrylamide、cyanoacrylamide）
   - `bbb_enhancers`：BBB 增强基团（如 fluorine、methoxy、trifluoromethyl）
   - `estimated_mw`：预估分子量
   - `expected_logp`：预期 LogP 范围
   - `expected_tpsa`：预期 TPSA 范围
   - `rationale`：设计理由
3. LLM 评估并排序，保留 b=2 个最优策略
4. 若全 impossible，使用 2 个硬编码策略兜底：
   - `acrylamide + fluorine, methoxy`（EGFR 共价弹头 + BBB 友好基团）
   - `cyanoacrylamide + fluorine, trifluoromethyl`（替代亲电基团 + 亲脂性增强）

**输出：** b=2 个组装策略节点（Level 1），每个节点持有 warhead + BBB enhancers

### Level 2 — SMILES Generation（SMILES 生成 + 多轮优化）

**输入：** 骨架 + 组装策略

**处理流程（多轮循环）：**
```
for round in range(tot_refinement_rounds):  # 默认 1 轮
    │
    ├── ToT 调用 LLM 生成完整 SMILES 字符串
    │
    ├── ToT 解析 SMILES → RDKit 化学合法性校验
    │       失败 → ToT 调用 LLM Self-Correction（自动修正语法错误）
    │
    ├── ToT 调用 GBMPhysicalEvaluator 全量评估
    │       → Vina/DILI/hERG/BBB/RDKit 描述符
    │       → MPO Reward 计算 + 硬截断
    │       → Verdict: SURE / LIKELY / IMPOSSIBLE
    │
    ├── ToT 硬剪枝：is_pruned=True → 丢弃该分支
    │
    ├── 若 round > 0：注入上一轮最佳分子的物理反馈到 prompt
    │   → build_feedback_text() 生成自然语言反馈
    │   → "Vina=-9.2 kcal/mol (强结合) | DILI=0.12 (低风险) |
    │       BBB=0.75 (高CNS渗透) | MW=432.5 | LogP=3.21 | Reward=0.72"
    │
    └── 早停：若 best_reward >= tot_good_reward_threshold (0.65) → 退出
```

**输出：** 通过物理评估的 SMILES 列表（`final_smiles_nodes`）

## 3. 完整搜索树结构（k=3, b=2）

```
                    ┌──────────────────────────────────────────────────────┐
                    │  Domain Prompt: EGFR + BBB High + LogP 2.0-4.0      │
                    │              TPSA 40-120 + MW 300-500               │
                    └──────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
              Scaffold 1          Scaffold 2          Scaffold 3
           (quinazoline)         (pyridine)         (pyrimidine)
             SURE ⭐               LIKELY             IMPOSSIBLE ✗
                │                    │                    │
        ┌───────┴───────┐     ┌────┴────┐
        ▼               ▼       ▼
  Strategy 1.1    Strategy 1.2  Strategy 2.1
  acrylamide     cyanoacryl    piperazine
  LIKELY         SURE ⭐         IMPOSSIBLE ✗
       │              │
 ┌─────┴─────┐  ┌───┴────┐
 ▼           ▼  ▼        ▼
SMILES     SMILES SMILES SMILES
1.1.1      1.1.2  1.2.1  1.2.2
SURE ⭐    prune ✗ SURE ⭐ prune ✗
   │                   │
   └───────────────────┘
         ↓
  [SMILES_1.1.1, SMILES_1.2.1]
  ✓ 通过物理评估 → 纳入 final_smiles_nodes
```

**说明：**
- 骨架层共 3 个候选（Scaffold 3 为 IMPOSSIBLE，直接丢弃）
- 保留 b=2 个骨架（Scaffold 1 + Scaffold 2）
- 每个骨架下生成 b=2 个组装策略
- 每个策略生成多个 SMILES，物理评估后仅保留通过评估的
- 最终输出：通过物理评估的 SMILES 列表（不含任何 impossible 或 pruned 节点）

## 4. Guardrail 安全机制

| 机制 | 触发条件 | ToT 决策 |
|------|---------|---------|
| Self-Correction | 解析 scaffold/assembly/smiles 格式失败 | 降低 temperature + 前车之鉴，重新调用 LLM（最多 3 次） |
| SMILES Fallback | 微调模型无有效 SMILES 输出 | 切换 base_merged 模型 + few-shot prompt |
| Hardcoded Fallback | 所有分支 evaluation 为 impossible | ToT 强制注入硬编码候选（骨架/策略），跳过 LLM 调用 |
| Early Stopping | 连续 N 次无新增 SMILES 或 reward ≥ 0.65 | ToT 主动终止当前靶点搜索 |
| Budget Check | 达到最大 LLM 调用次数（50）或最大时间（300s） | 立即退出搜索 |

## 5. 评估字段

每个 ToTNode 包含：
- `level`: 0 (Scaffold) / 1 (Assembly) / 2 (SMILES)
- `content`: 节点内容（骨架名、策略描述、SMILES 字符串）
- `evaluation`: 'sure' / 'likely' / 'impossible'
- `physical_result`: `PhysicalEvaluationResult`（Level 2 独有）
- `metadata`: 元数据（warhead 类型、BBB enhancers、预估 MW 等）
- `parent` / `children`: 树结构指针
