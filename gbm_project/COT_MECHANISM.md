## Llamole GBM 项目中的 Chain-of-Thought (CoT) 机制说明

本文件说明 Llamole 在 GBM（胶质母细胞瘤）药物设计任务中如何集成 Chain-of-Thought（CoT）推理，包括配置开关、Prompt 模板、知识库集成和生成管线。

---

## 1. 整体设计概览

- **目标**：在 GBM 药物设计中，引入结构化的 CoT 推理，让大模型按“分析 → 设计策略 → 安全性 → 候选分子生成”的流程思考，而不是直接给出 SMILES。
- **核心模块**：
  - `configs/gbm_prompts.yaml`：CoT 模板与相关 prompt 片段
  - `src/gbm_prompt_generator.py`：封装“GBM 领域知识 + CoT”的 Prompt 生成器
  - `scripts/generate_with_full_llamole.py`：完整 Llamole + CoT + 图模型生成流程
  - `scripts/generate_target_specific_molecules.py`：面向单靶点的 CoT 生成管线
  - `fixed_gbm_method_with_cot.py`：演示“正确使用 CoT + 知识库”的示例方法
  - `configs/gbm_generation.yaml`、`experiments/benchmark_experiments.py`：是否启用 CoT 的配置开关

---

## 2. 配置层：如何启用 / 关闭 CoT

### 2.1 主生成配置

- 文件：`configs/gbm_generation.yaml`
- 关键选项：
  - **`gbm_generation_mode: true`**：启用 GBM 专用生成模式
  - **`use_gbm_domain_knowledge: true`**：注入 GBM 领域知识（靶点、临床、结构等）
  - **`use_cot_reasoning: true`**：开启 Chain-of-Thought 推理

组合效果：

- 只开 `gbm_generation_mode`：普通生成，但走 GBM 代码路径
- 再开 `use_gbm_domain_knowledge`：会注入靶点信息、临床数据等
- 再开 `use_cot_reasoning`：使用 CoT 模板生成结构化推理流程（推荐）

### 2.2 基准实验中的 CoT 场景

- 文件：`experiments/benchmark_experiments.py`
- 预设了多种对比场景（示例概念）：
  - **baseline**：`use_cot_reasoning: False`，不使用 CoT
  - **gbm_knowledge**：只用 GBM 知识，不用 CoT
  - **cot_reasoning**：只用 CoT，不用领域知识
  - **combined**：`use_gbm_domain_knowledge: True` + `use_cot_reasoning: True`，即完整方案

同时在 `generation_params` 中还会设置 `use_cot: True/False`，脚本会根据此选择是否构造 CoT Prompt。

### 2.3 单靶点脚本中的 CoT 开关

- 文件：`scripts/generate_target_specific_molecules.py`
- CLI 参数：
  - **`--use_cot`**：是否使用 CoT 推理（默认 True）
- 内部配置：
  - `config['cot']['enabled']`：
    - `True` → 使用 `_build_cot_prompt` 构造带 CoT 推理的 Prompt
    - `False` → 使用 `_build_simple_prompt` 仅给出简单设计要求

---

## 3. Prompt 模板层：CoT 的“话术骨架”

### 3.1 主 CoT 模板：`step_by_step_design`

- 文件：`configs/gbm_prompts.yaml`
- 键：`cot_reasoning_templates.step_by_step_design`
- 功能：给出一个“市售 GBM 药物参考 + 分步设计”的标准 CoT 流程，指导模型按步骤推理。

**主要结构（概念）：**

- **STEP 1：参考市售 GBM 药物**
  - 分析 Erlotinib、Gefitinib、Afatinib、Osimertinib 等的 BBB 穿透、分子量、骨架特征。
- **STEP 2：靶点与大小约束**
  - 以 EGFR 为例，收紧分子量（例如 300–500 Da）、LogP、TPSA、HBD/HBA 等范围。
- **STEP 3：骨架选择**
  - 优先 quinazoline，其次 pyrimidine/indole。
- **STEP 4：片段组装策略**
  - 核心骨架 + warhead（如 acrylamide）+ BBB 增强取代基，控制整体 MW。
- **STEP 5：性质预测与对标**
  - 与 Erlotinib/Afatinib 的 ADMET 特征对比，微调取代以贴近成功药物。
- **STEP 6：选择性与安全性检查**
  - 考虑 EGFRvIII 选择性、正常细胞毒性、合成可行性等。
- **STEP 7：生成候选**
  - 生成给定数量的 SMILES，并确保满足所有结构和性质约束。

该模板本质上是一个“药物设计 SOP”，以自然语言约束模型的推理路径。

### 3.2 约束与评估模板

同文件中还包含辅助模板：

- **`constraint_templates`**：
  - 如 `bbb_optimization`：把 `mw_threshold, logp_min, logp_max, tpsa_threshold, hbd_max` 等结构化参数转成自然语言约束。
  - 还有 `selectivity_design`、`resistance_evasion` 等，分别描述选择性与耐药规避策略。
- **`evaluation_prompts`**：
  - 如 `bbb_assessment`、`gbm_activity_prediction`、`toxicity_evaluation`，用于后续对生成分子的文本评估。

这些模板由 Prompt 生成器按需拼接，使 CoT 分析部分具备参数化与可复用性。

---

## 4. Prompt 生成层：如何把“知识 + CoT”拼成最终 Prompt

核心类：`GBMPromptGenerator`（在 `src/gbm_prompt_generator.py` 中）

### 4.1 领域 Prompt：`generate_domain_prompt`

- 输入：`target_name`、`constraints`（如分子量、LogP、BBB 要求等）
- 主要步骤：
  - 从 `GBMKnowledgeBase` 中通过 `get_target_info(target_name)` 获取靶点资料。
  - 从 `prompt_templates['target_specific_prompts'][target_name]` 读取预定义靶点描述；若无则调用 `_generate_target_specific_prompt` 生成。
  - 引入 `base_gbm_prompt`：包含“GBM 专家身份 + 生物学特征 + 治疗挑战”等背景。
  - 调用 `_format_target_requirements` 将靶点挑战、结构要求、约束整合为条目式“设计要求”。
  - 通过 `_add_clinical_insights` 注入临床挑战、成功模式等信息。
  - 调用 `StructuralKnowledgeInjector.inject_structural_context(...)` 注入结构/序列/图模型相关的上下文。

**结果**：得到一个长文本 `domain_prompt`，包含：

- GBM 背景与生物学特征
- 特定靶点信息与结构要求
- 明确的分子设计约束
- 临床经验和结构知识

### 4.2 CoT Prompt：`generate_cot_prompt`

- 函数签名：`generate_cot_prompt(domain_prompt, target_name, reasoning_type="step_by_step_design")`
- 逻辑：
  - 从 `cot_reasoning_templates` 中选取对应模板（默认 `step_by_step_design`）。
  - 调 `_generate_target_reasoning(target_name)`：
    - 使用 `get_similar_molecules` 等接口，从知识库中找到若干相似分子。
    - 注入这些分子的名称、研发状态、作用机制等。
    - 补充靶点的耐药机制、临床安全性与选择性要求。
  - 将目标靶点推理附加到通用 CoT 模板后，得到 `cot_prompt`。
  - 最终 CoT Prompt 一般为：
    - **领域 Prompt（domain_prompt）**  
    - + 双空行  
    - + **CoT 模板 + 靶点特定推理（cot_prompt）**

### 4.3 完整 Prompt：`generate_full_prompt`

- 函数签名：`generate_full_prompt(target_name=None, constraints=None, use_cot=True, reasoning_type="step_by_step_design")`
- 步骤：
  - 若未指定靶点，则从知识库中随机选择一个。
  - 调用 `generate_domain_prompt` 得到 `domain_prompt`。
  - 若 `use_cot=True`，调用 `generate_cot_prompt` 得到 `full_cot_prompt`；否则直接使用 `domain_prompt`。
  - 在末尾追加一段生成指令，要求：
    - 输出一个新的 GBM 候选分子。
    - 给出 SMILES 字符串。
    - 说明设计理由。

**总结**：`generate_full_prompt` 是“规范式 CoT 集成方式”，`fixed_gbm_method_with_cot.py` 用的就是这条路径。

---

## 5. 生成与后处理层：如何利用 & 记录 CoT

### 5.1 完整 Llamole + CoT + 评估：`generate_with_full_llamole.py`

主要流程：

- **模型加载**：
  - 基础模型：`Qwen2-7B-Instruct`
  - LoRA 适配器：通用 Llamole 适配器 + GBM 专用适配器
  - 图模型：GraphDecoder、GraphEncoder、GraphPredictor、LM–Graph connector

- **Prompt 构造**：
  - 从 `gbm_prompts.yaml` 读取 `cot_reasoning_templates.step_by_step_design` 作为 `cot_template`。
  - 调用 `GBMPromptGenerator.generate_domain_prompt("EGFR", constraints=...)` 获得 `domain_prompt`。
  - 组合得到：
    - `final_prompt = cot_template + "\n\n" + domain_prompt + "\n\n" + simple_instruction + output_format`
    - 其中：
      - `simple_instruction`：简单英文指令（如“Generate 50 GBM novel drug candidates...”）。
      - `output_format`：强制使用编号输出：
        - `1. SMILES_string`
        - `2. SMILES_string`
        - …

- **生成与 CoT 提取**：
  - 使用微调后模型生成 `raw_response`。
  - 将 `raw_response` 原样保存到 JSON 中（便于后续分析）。
  - 通过正则与编号列表切分出“思维链”部分：
    - 优先以“编号列表起始位置”（`\n\s*1[\.\)\:]\s`）为界，将上半部分视为 CoT。
    - 若没有编号，则尝试以 `"Chain-of-Thought"` 标记截断。
    - 得到的 CoT 文本保存为 `cot_chain` 字段。

- **SMILES 提取与评估**：
  - 按编号模式提取 SMILES，若不足则回退到通用提取器 `extract_multiple_smiles`。
  - 使用 `validate_smiles`（基于 RDKit）过滤无效 SMILES。
  - 通过 `GBMEvaluator` 对每个有效 SMILES 打分，生成统一结构 `unified_results`：
    - 包含 `composite_score_unified`、各项打分、属性与评估文字。
  - 将：
    - `experiment_config`
    - `instruction`
    - `full_prompt`
    - `raw_response`
    - `cot_chain`
    - `results`
    一并写入实验结果 JSON。

**要点**：在该脚本中，CoT 既体现在 Prompt 输入结构上（CoT 模板 + 领域知识 + 格式约束），也通过 `cot_chain` 字段显式记录与分析模型的推理过程。

### 5.2 单靶点 CoT 管线：`TargetSpecificMoleculeGenerator`

- 文件：`scripts/generate_target_specific_molecules.py`
- 类：`TargetSpecificMoleculeGenerator`

主要流程：

- **加载靶点结构信息**：
  - `_load_targets_info` 从 `data/processed_targets` 中读取 `<target>_summary.json`：
    - 包含序列长度、预测结合位点、关键残基等。

- **CoT Prompt 构造**（`cot.enabled=True` 时）：
  - 调 `generate_prompt(target_name, target_info)`，内部走 `_build_cot_prompt`：
    - 标题：`Chain-of-Thought Reasoning for {target_name} Inhibitor Design`
    - 包含：
      - Target Analysis：靶点描述 + 序列长度 + 预测 binding sites。
      - Market Reference Analysis：该靶点已知抑制剂的一般性质（MW、LogP、核心骨架、关键相互作用）。
      - Molecular Design Strategy：选择骨架、添加功能团、确保 H-bond、满足 Lipinski 等。
      - Safety Considerations：避免某些毒性结构（如 aniline）、控制芳环数与 LogP。
      - Generation Phase：明确要求先输出“Chain-of-Thought reasoning”，再按固定格式输出 `SMILES1`–`SMILES5`。

- **生成与 SMILES 抽取**：
  - `_generate_single_molecule` 用该 Prompt 直接驱动模型生成。
  - `_extract_smiles` 通过正则匹配 `SMILES\d*:` 行，并用 `_basic_validate_smiles` 做轻量的有效性检查（不依赖 RDKit）。
  - 在最多若干次尝试内凑够目标数量的有效 SMILES，去重后返回。

**对比**：

- `generate_with_full_llamole.py`：CoT 模板来自 YAML，注重多分子 + 统一评估。
- `generate_target_specific_molecules.py`：CoT 模板在 Python 中手写，强调单靶点 + 结构信息驱动。

---

## 6. “修复版”示例：正确集成 CoT 与知识库

- 文件：`fixed_gbm_method_with_cot.py`
- 目的：对比“错误实现”和“正确实现”的 GBM 方法。

### 6.1 错误实现（概念）

- 仅使用一个非常短的 Prompt：
  - 类似“设计 GBM 分子，靶点 EGFR，给 SMILES”。
- 特点：
  - 没有调用 `GBMKnowledgeBase`。
  - 没有使用 `GBMPromptGenerator`。
  - 没有 CoT 推理步骤。
  - 不包含临床/结构/市售药物等知识。

### 6.2 正确实现流程

在 `demonstrate_correct_gbm_method()` 中：

- **步骤 1：加载知识库与 Prompt 生成器**
  - `GBMKnowledgeBase` 读取：
    - `data/gbm_targets/gbm_targets.json`
    - `data/gbm_clinical/gbm_clinical_data.json`
    - `data/gbm_molecules/gbm_molecules.json`
  - `GBMPromptGenerator` 加载 `configs/gbm_prompts.yaml`。

- **步骤 2：生成完整 CoT Prompt**
  - 调用：
    - `generate_full_prompt(target_name="EGFR", constraints=..., use_cot=True, reasoning_type="step_by_step_design")`
  - 得到包含：
    - GBM 背景、生物学特征、治疗挑战。
    - EGFR 靶点信息、约束条件。
    - 市售药物与类似分子参考。
    - 明确的 CoT 推理步骤。

- **步骤 3：验证 CoT 与知识集成**
  - 检查 Prompt 是否包含若干 CoT 关键短语（如“让我们逐步思考”等）。
  - 从 `GBMKnowledgeBase` 中确认：
    - 已集成靶点描述、挑战、现有药物数。
    - 已集成临床挑战与成功模式。

- **步骤 4：包装为 Llamole 格式**
  - 构造：
    - `<design_start>{full_gbm_prompt}<design_end>`
  - 仅作 tokenization 测试，不强制实际生成（出于资源考虑）。

### 6.3 Prompt 长度对比

- 函数：`compare_prompt_lengths()`
- 对比：
  - 原始“一句话” Prompt：几十字符。
  - CoT + 知识库 Prompt：约 2000 字符（估算）。
- 说明：
  - CoT + 知识库极大增加信息量与结构化程度。
  - 对模型输出质量与可控性有明显提升。

---

## 7. 关键要点总结

- **双层设计**：
  - **模板层**：在 YAML / Python 字符串中定义 CoT 流程、语气和输出格式。
  - **生成器层**：`GBMPromptGenerator`、`TargetSpecificMoleculeGenerator` 将“靶点 + 约束 + 知识库”实例化为最终 Prompt。

- **知识驱动 CoT**：
  - 通过靶点知识、市售药物参考和临床经验将 CoT 从“抽象推理”变成“领域专家 SOP”。

- **端到端闭环**：
  - 配置/CLI 开关决定是否启用 CoT。
  - Prompt 侧用 CoT 模板约束生成路径。
  - 后处理层显式分离 CoT 文本与 SMILES，并对 SMILES 自动评估与持久化。

这套机制使得 Llamole 在 GBM 场景下不仅能“给出分子”，还能给出有医学与化学依据的推理过程，便于研究人员审查与迭代设计。


