## GBM 英文 ToT 提示词问题与修复说明

### 一、问题背景

- **现象**：在英文模式下运行 Tree-of-Thoughts (ToT) 分子设计流程时，Level 0（Scaffold 提议阶段）无法解析出任何合法骨架，最终 `num_generated = 0`，整个 ToT 设计流水线在最初阶段即终止。
- **根因**：
  - `TreeOfThoughtsGenerator` 在英文模式下会调用 `GBMPromptGenerator(language="english")`；
  - 英文配置文件 `english_gbm_prompts.yaml` 中最初**缺少**与中文版对齐的 `tot_design_nodes` 配置；
  - 代码因此回退到一个不带严格格式约束的“泛化提示词”；
  - 但解析逻辑 `parse_scaffold_proposals` 严格依赖诸如 `Scaffold 1: / Name: / Base MW: / BBB Potential:` 等标签；
  - 两者不匹配导致解析不到任何 scaffold，从而 ToT 搜索图为空。

### 二、配置层修改：补齐英文 ToT 模板

#### 2.1 修改文件

- **文件**：`gbm_project/configs/english_gbm_prompts.yaml`
- **新增配置段落**：`tot_design_nodes`，包含 4 个子模板：
  - `propose_scaffold`
  - `propose_assembly`
  - `generate_smiles`
  - `evaluate_state`

#### 2.2 `propose_scaffold` 模板

- **占位符对齐**：与 `GBMPromptGenerator.build_tot_propose_prompt(..., step_type="scaffold")` 一致，使用：
  - `{target_name}`, `{mw_range}`, `{bbb_requirement}`, `{logp_range}`
- **关键设计**：
  - 明确要求模型输出 **3 个** scaffold 提议；
  - 使用与解析逻辑严格对齐的标签格式：
    - `Scaffold 1:`
    - `Name: ...`
    - `Rationale: ...`
    - `Base MW: ... Da`
    - `BBB Potential: ...`
  - 提供示例骨架（如 quinazoline、pyrimidine、indole）以强化模型对结构化格式的遵循。
- **效果**：保证 `parse_scaffold_proposals` 的正则 / 行级解析可以稳定命中，英文模式下也能解析出 3 个有效 scaffold。

#### 2.3 `propose_assembly` 模板

- **占位符对齐**：与 `step_type="assembly"` 的构造逻辑一致，使用：
  - `{domain_prompt}`, `{selected_scaffold}`, `{scaffold_mw}`, `{remaining_mw}`, `{target_mw}`
- **任务说明**：
  - 要求基于已选 scaffold 给出 **3 种不同的片段组装策略**。
- **每个策略需包含**：
  - Warhead 类型（如 acrylamide、cyanoacrylamide 等）；
  - BBB 增强取代基（F、OMe、Me、CF₃ 等）；
  - 估算总分子量；
  - 预期 LogP 范围；
  - 简短设计 rationale。
- **输出格式**：
  - 使用 `Strategy 1/2/3` 与固定字段：`Warhead:`, `BBB Enhancers:`, `Estimated Total MW:`, `Expected LogP:`, `Rationale:`
  - 便于后续（若需要）对策略进行结构化解析或打分。

#### 2.4 `generate_smiles` 模板

- **占位符对齐**：
  - `{domain_prompt}`, `{selected_scaffold}`, `{assembly_strategy}`, `{warhead_type}`, `{bbb_enhancers}`, `{target_mw}`, `{target_logp_range}`
- **任务说明**：
  - 基于 scaffold + 组装策略生成 **3 条具体候选分子的 SMILES**。
- **硬性约束**：
  - 每条 SMILES 必须：
    - 包含指定 scaffold；
    - 引入指定 warhead；
    - 含有给定 BBB 增强取代基；
    - 尽量满足目标 MW 与 LogP 区间；
    - 在化学与合成上具备合理性；
    - 为有效、语法正确的 SMILES。
- **输出格式**：
  - 严格要求：
    - `SMILES 1: <...>`
    - `SMILES 2: <...>`
    - `SMILES 3: <...>`
  - 为 SMILES 解析与后续性质预测提供稳定接口。

#### 2.5 `evaluate_state` 模板

- **占位符对齐**：
  - `{domain_prompt}`, `{partial_solution}`
- **评估维度**（只用于指导模型思考，不做结构化解析）：
  - BBB 穿透潜力（MW、LogP、TPSA、HBD/HBA 等）；
  - 结合位点兼容性；
  - 合成可行性；
  - 安全性 / 毒性风险；
  - 潜在临床转化性。
- **输出约束（关键）**：
  - 输出必须为 **单个词**，且只能是三选一：
    - `"sure"`：高度符合设计目标；
    - `"likely"`：整体可行但存在一定不确定性；
    - `"impossible"`：明显违背关键约束或不可行。
  - 模板中以醒目说明：**仅输出一个单词，不要附带解释**，以严格匹配现有评估函数的解析逻辑。

### 三、代码层修改建议（可选增强）

> 以下为推荐的进一步增强措施，本次报告不假定已全部实现。

#### 3.1 在 `GBMPromptGenerator` 初始化时检查 `tot_design_nodes`

- **文件**：`gbm_prompt_generator.py`
- **建议逻辑**：
  - 在加载英文配置后，读取 `self.prompt_templates.get("tot_design_nodes", {})`；
  - 若该字段缺失或为空：
    - 打印显式告警（例如使用 `print` 或日志）：
      - 提示 ToT 将回退到泛化 prompt；
      - 说明这可能会导致 scaffold 解析失败；
      - 明确指出应在 `english_gbm_prompts.yaml` 中补齐 `tot_design_nodes`。
- **优点**：
  - 避免静默退化，仅在最终结果 `num_generated = 0` 时才暴露问题；
  - 与现有代码兼容：不强制抛异常，只是提供清晰的诊断信息。

### 四、行为预期与影响评估

- **Level 0（Scaffold 提议）**：
  - 由于英文 ToT 提示模板现在与解析逻辑严格对齐，预期能稳定获得并解析出 3 个候选 scaffold；
  - BFS/DFS 的 ToT 搜索可以正常展开，不再在初始层就终止。
- **Level 1（Assembly 策略）与 Level 2（SMILES 生成）**：
  - 输出均采用统一、结构化的 3 候选格式，便于未来对策略与 SMILES 进行自动化筛选与打分扩展。
- **评估阶段**：
  - `evaluate_state` 的单词输出与当前 `sure/likely/impossible` 三态决策逻辑完全匹配；
  - 减少模型输出长段解释导致解析失败的风险。
- **可观测性**：
  - 一旦未来有人误删或破坏 `tot_design_nodes` 配置，通过初始化时的告警以及结构化模板的存在，可以更快速定位问题根因。

### 五、后续优化建议

- **1. 配置解耦**  
  建议将 ToT 专用模板单独拆分到独立文件（如 `gbm_tot_prompts.yaml`），并通过语言无关的方式加载，以减少中英文配置漂移带来的维护成本。

- **2. 解析鲁棒性提升**  
  可以在 `parse_scaffold_proposals` 等函数中增加对标签轻微变体的容错（大小写、冒号后空格、换行差异等），并在解析失败时输出原始模型响应的头几行，方便诊断。

- **3. 日志与监控**  
  对 ToT 每一层的有效分支数量、评估分布（`sure/likely/impossible` 比例）进行简单统计与日志记录，便于在实际项目中监控模型行为与调优提示词。


