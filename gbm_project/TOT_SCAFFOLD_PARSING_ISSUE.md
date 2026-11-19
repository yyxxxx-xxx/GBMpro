### Tree-of-Thoughts 骨架提议解析格式问题说明

本文档梳理当前 GBM ToT 生成流程中「骨架提议（Scaffold Proposal）」解析相关的问题来源、具体表现和建议修复方案，便于后续重构或调参时参考。

---

### 1. 相关模块与调用链

- **核心类**：`TreeOfThoughtsGenerator`  
  - 文件：`gbm_project/scripts/generate_tot_molecules.py`
  - 关键方法：
    - `bfs_tot_search(self, target_name: str)`：执行三层 ToT BFS 搜索
    - `parse_scaffold_proposals(self, response: str)`：解析第一层「骨架提议」
- **Prompt 生成器**：`GBMPromptGenerator`  
  - 文件：`gbm_project/src/gbm_prompt_generator.py`
  - 关键方法：
    - `build_tot_propose_prompt(self, domain_prompt, current_state, step_type)`：为 `scaffold / assembly / smiles` 三个阶段构建 ToT prompt
    - `build_tot_evaluate_prompt(self, domain_prompt, partial_solution)`：构建 ToT 评估阶段 prompt
- **配置文件**：
  - `gbm_project/configs/gbm_prompts.yaml`：包含完整的 **ToT 模板**（`tot_design_nodes`）
  - `gbm_project/configs/english_gbm_prompts.yaml`：**当前运行时实际加载的英文配置，不包含 ToT 模板**

调用流程（简化）：

1. `TreeOfThoughtsGenerator.generate_molecules(target)`  
2. → `self.load_prompt_generator(target_name)`  
   - 构造 `GBMPromptGenerator(language="english")`
   - 加载 `english_gbm_prompts.yaml`，覆盖原始 `gbm_prompts.yaml` 内容
3. → `self.bfs_tot_search(target_name)`  
4. → Level 0 骨架提议：
   - 使用 `GBMPromptGenerator.build_tot_propose_prompt(..., step_type='scaffold')` 构建 prompt
   - 使用 `generate_with_model` 调用大模型得到 `scaffold_response`
   - 使用 `parse_scaffold_proposals(scaffold_response)` 解析骨架提议
5. 若 `parse_scaffold_proposals` 结果为空，则后续 ToT 搜索直接返回空：
   - `logger.warning("No valid scaffolds found, returning empty results")`

---

### 2. 期望的「骨架提议」输出格式（设计意图）

在 `gbm_prompts.yaml` 中，`tot_design_nodes.propose_scaffold` 明确给出了 **期望的输出示例和严格格式要求**，大致如下：

```text
Output exactly 3 scaffold proposals in this exact format:

Scaffold 1:
Name: quinazoline
Rationale: Suitable for EGFR targeting with good BBB penetration
Base MW: 160 Da
BBB Potential: high

Scaffold 2:
...

Scaffold 3:
...
```

`TreeOfThoughtsGenerator.parse_scaffold_proposals` 针对这一格式写了强依赖的解析逻辑：

- **方法 1：严格正则匹配**
  - 期待模式（简化版）：
    - `Scaffold <编号>:`
    - `Name: <单行名称>`
    - `Rationale: <单行理由>`
    - `Base MW: <整数> Da`
    - `BBB Potential: <单行文本>`
- **方法 2：逐行 Heuristic 提取**
  - 按行查找包含 `name: / rationale: / base mw / bbb potential` 等关键词的行
  - 推断出 `name / rationale / base_mw / bbb_potential`
  - 没有给出时使用默认值补全
- **方法 3：兜底关键字搜索**
  - 在全文中查找常见骨架名称，如 `quinazoline / pyrimidine / indole` 等
  - 构建若干默认骨架提议

设计意图是：**只要模型基本遵守了模板中给出的示例格式，就可以稳定解析出 3 个骨架提议**，并用于后续 ToT 层级搜索。

---

### 3. 实际运行时的 Prompt 与配置不匹配

#### 3.1 英文配置中缺失 ToT 模板

`GBMPromptGenerator.__init__` 的逻辑是：

1. 先加载 `gbm_prompts.yaml` 到 `self.prompt_templates`
2. 若 `language == "english"`，则尝试加载：
   - `english_config_path = prompts_config_path.replace("gbm_prompts.yaml", "english_gbm_prompts.yaml")`
   - 如果存在，则 **用英文配置整体覆盖 `self.prompt_templates`**

当前 ToT 生成器调用时使用的是：

```python
GBMPromptGenerator(
    knowledge_base=kb,
    prompts_config_path=self.prompts_config_path,
    language="english"
)
```

而 `english_gbm_prompts.yaml` 中 **只有 base_gbm_prompt / target_specific_prompts / cot_reasoning_templates / constraint_templates / evaluation_prompts**，并 **没有**：

- `tot_design_nodes.propose_scaffold`
- `tot_design_nodes.propose_assembly`
- `tot_design_nodes.generate_smiles`
- `tot_design_nodes.evaluate_state`

结果是：

- `self.prompt_templates.get('tot_design_nodes', {})` 返回空字典
- `build_tot_propose_prompt(..., step_type='scaffold')` 找不到模板，走到默认分支：

```text
{domain_prompt}

Propose scaffold options for GBM drug design.
```

也就是说，**实际发送给模型的「骨架提议」prompt 完全没有示例格式、没有 `Scaffold 1:` / `Name:` / `Base MW:` 等字段要求，只是一段泛化的英文说明**。

#### 3.2 解析逻辑仍然假设有严格格式

`parse_scaffold_proposals` 在解析时仍然假定模型会按照「Scaffold 1: ... Name: ... Base MW: ... BBB Potential: ...」这一结构来输出：

- 第一阶段的正则匹配高度依赖上述精确字段名和顺序
- 第二阶段逐行解析同样依赖 `name: / rationale: / base mw / bbb potential` 这些标记
- 如果模型按自然语言自由描述（例如「First, we consider a quinazoline scaffold with good BBB penetration ...」），则：
  - 正则匹配失败（不存在 `Scaffold 1:` / `Name:` / `Base MW:` 等行）
  - 逐行解析也难以找到成对的字段
  - 兜底的关键字搜索可能只能在极少数情况下勉强识别到一两个骨架名称

在这种情况下，`scaffold_proposals` 大概率为空，后续逻辑会打印：

```text
No valid scaffolds found, returning empty results
```

并直接返回空结果，导致本轮 ToT 搜索 **完全生成不了 SMILES**。

---

### 4. 现象与影响

#### 4.1 ToT 结果文件中 `num_generated = 0`

例如 `gbm_project/experiments/tot_batch_20260301_080810/EGFR/tot_generation_results.json`：

```json
{
  "config": {
    "target": "EGFR",
    "tot_depth": 3,
    "tot_k": 3,
    "tot_b": 2,
    "constraints": {
      "mw_range": "300-500",
      "bbb_requirement": "high",
      "logp_range": "2.0-4.0"
    }
  },
  "num_generated": 0,
  "molecules": [],
  "timestamp": "2026-03-01T08:08:22.054825"
}
```

这说明：**整条 ToT 流水线运行成功，但在骨架层就失败了，后续没有生成任何 SMILES**。

#### 4.2 日志提示但不直观暴露根因

在 `TreeOfThoughtsGenerator.bfs_tot_search` 中存在如下日志输出：

- 打印完整的骨架响应与长度：
  - `Generated scaffold proposals (full response): ...`
  - `Response length: xxx characters`
- 打印解析出的骨架数量：
  - `Parsed {len(scaffold_proposals)} scaffold proposals`
- 若最终 `selected_scaffolds` 为空：
  - `No valid scaffolds found, returning empty results`

但在高层实验脚本（例如 `generate_tot_batch_candidates.py`）的日志中，通常只看到「某靶点生成 0 个分子」，**并不会直接把「骨架解析失败」暴露给最终用户**，因此容易误以为是模型质量问题或 SMILES 解析问题，而忽略了 prompt/解析协议的不一致。

---

### 5. 问题归类

综合来看，「骨架提议解析存在格式问题」主要包含以下几个维度：

1. **Prompt 模板缺失 / 配置不一致**
   - 中文版 `gbm_prompts.yaml` 中定义了完整的 `tot_design_nodes`，包含严格示例格式；
   - 英文版 `english_gbm_prompts.yaml` 中完全没有对应字段；
   - 实际 ToT 流程使用的是英文配置，导致骨架 prompt 没有结构化约束。

2. **解析逻辑过度依赖硬编码格式**
   - `parse_scaffold_proposals` 使用强正则 + 关键字行匹配，假设模型输出精确遵守「Scaffold / Name / Base MW / BBB Potential」标签；
   - 对自然语言、缩写或轻微格式偏差不够鲁棒。

3. **语言配置与 ToT 配置耦合**
   - 当前逻辑是「语言 == english → 整体切换到 english_gbm_prompts.yaml」；
   - 这会把 ToT 专用模板也一并替换掉，除非手动在英文配置中维护同样的 `tot_design_nodes`。

4. **错误反馈不够显式**
   - 从上层实验脚本的视角，只能看到「某靶点生成了 0 个分子」；
   - 没有显式提示「骨架提议解析失败」或「未找到任何符合格式的 Scaffold」这一根因。

---

### 6. 建议的修复方向

#### 6.1 配置层：补全英文 ToT 模板 / 拆分 ToT 配置

1. **在 `english_gbm_prompts.yaml` 中补充 `tot_design_nodes` 段落**  
   - 可以直接参考 `gbm_prompts.yaml` 中的 `tot_design_nodes.propose_scaffold / propose_assembly / generate_smiles / evaluate_state`，使用英文描述但保持：
     - 相同的字段名（`Scaffold 1:`, `Name:`, `Base MW:`, `BBB Potential:` 等）
     - 相同的行结构与标签顺序
   - 这样，英文配置与解析逻辑完全对齐，ToT 在英文模式下也能正常工作。

2. **或者：将 ToT 模板从语言配置中独立出来**  
   - 将 `tot_design_nodes` 拆分到一个单独的配置文件（例如 `gbm_tot_prompts.yaml`），不随 `language` 切换；
   - `GBMPromptGenerator` 在加载英文配置时：
     - 只用英文文件覆盖 CoT/领域相关模板；
     - ToT 模板始终来自一个统一的、与解析逻辑严格对齐的文件。

#### 6.2 解析层：放宽格式要求、增强鲁棒性

在保留现有严格解析逻辑的基础上，可以增加若干兜底策略：

1. **增加「解析失败可视化」日志**  
   - 在 `parse_scaffold_proposals` 返回空列表之前，打印一条更明确的 warning：
     - 提示「未在响应中检测到任何形如 `Scaffold n:` 的结构化段落」
     - 打印响应的前几行以便快速人工检查。

2. **更宽松地接受变体标签**  
   - 例如容忍：
     - `Base MW` 写成 `Base molecular weight`
     - `BBB Potential` 写成 `BBB potential` / `BBB penetration potential`
   - 可通过多种正则模式或关键字组合提升兼容性。

3. **针对无标签的自然语言增加关键字解析**  
   - 若没有任何 `Name:` 行，但文本中多次出现如 `quinazoline`, `pyrimidine`, `indole` 等常见 scaffold 名称：
     - 可以直接构造若干骨架提议（类似当前的「方法3」），并在元数据中标记为「inferred_from_free_text」。

#### 6.3 语言层：避免 ToT 行为依赖语言开关

1. **允许为 ToT 单独指定语言或模板路径**  
   - 在 `TreeOfThoughtsGenerator` 的配置中加入：
     - `tot_prompts_config_path`
     - `tot_prompt_language`
   - 避免因为全局 `language="english"` 导致 ToT 模板被错误替换。

2. **在 `GBMPromptGenerator` 中对缺失 `tot_design_nodes` 做显式检查**  
   - 若 `language="english"` 且 `tot_design_nodes` 不存在或为空：
     - 在初始化阶段直接抛出异常或打印明显 warning；
     - 提示需要在英文配置中补充 ToT 模板。

---

### 7. 总结

当前「骨架提议解析存在格式问题」的核心根因是：

- **解析代码假定模型输出遵循严格的 `Scaffold / Name / Base MW / BBB Potential` 结构；**
- **而实际运行时加载的英文 prompt 配置并未提供对应的 ToT 模板，导致模型只收到一个「泛化描述型」的 scaffold prompt，输出无法满足解析预期。**

直接后果是 ToT 流水线在 Level 0 就失效，`tot_generation_results.json` 中 `num_generated` 始终为 0。  
优先级最高的修复建议是：**确保英文配置中补齐与解析逻辑一一对应的 `tot_design_nodes` 模板，或者将 ToT 模板独立于语言配置单独维护**，再结合适度放宽的解析逻辑和更清晰的日志提示，从源头上消除格式不匹配带来的解析失败。





