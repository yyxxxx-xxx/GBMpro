# 图3技术文档：两阶段 LoRA 微调架构

## 1. 整体架构

模型加载分为两个串联的 LoRA 适配阶段，最终以 PeftModel 形式供 ToT 模块调用：

```
Qwen2-7B-Instruct (7B 参数, FP16, HuggingFace 格式)
        │
        ▼ [Stage 1: PEFT LoRA 适配]
┌─────────────────────────────────────────────────┐
│  Llamole Adapter                                 │
│  路径: saves/Llamole-Qwen2-7B-Instruct-Adapter  │
│  类型: LoRA (r=16, lora_alpha=32)               │
│  目标模块: q_proj, k_proj, v_proj, o_proj,      │
│           gate_proj, up_proj, down_proj          │
│  dropout: 0.05                                   │
│  作用: 通用指令遵循能力 + Llamole 图模型支持      │
└─────────────────┬───────────────────────────────┘
                  │ merge_and_unload()
                  │ LoRA 权重合并回基座，合并后
                  │ Llamole Adapter 不再单独存在
                  ▼
┌─────────────────────────────────────────────────┐
│  合并后基座模型 (Qwen2-7B-Instruct + Llamole)   │
│  作为 Stage 2 的起点                            │
└─────────────────┬───────────────────────────────┘
                  ▼ [Stage 2: PEFT LoRA 适配]
┌─────────────────────────────────────────────────┐
│  GBM Domain Adapter                              │
│  路径: saves/Llamole-Qwen2-7B-Instruct-Adapter-│
│        gbm-with-graph-models                    │
│  类型: LoRA (r=16, lora_alpha=32)               │
│  目标模块: 同 Stage 1                           │
│  dropout: 0.05                                   │
│  状态: PeftModel（保留，未合并）                 │
│  作用: GBM 领域专业知识 + 分子生成能力           │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
        最终推理模型 (PeftModel)
        供 TreeOfThoughtsGenerator 调用
        self.model.generate(prompt, ...)
```

## 2. Stage 1 — Llamole Adapter

**来源：** `saves/Llamole-Qwen2-7B-Instruct-Adapter`

**功能：** 通用指令遵循能力 + Llamole 图模型支持

**LoRA 配置：**
```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

**合并操作：** 调用 `PeftModel.merge_and_unload()`，将 LoRA 权重合并回基座，合并后 Llamole Adapter 不再以独立 PeftModel 形式存在。

**Llamole 图模型组件（可选加载）：**
- `GraphLLMForCausalMLM` — 整合语言模型与图模型的主模型类
- `GraphDecoder` — 图解码器（SMILES → Graph）
- `GraphEncoder` — 图编码器（Graph → Embedding）
- `GraphPredictor` — 图预测器（连接性预测）
- 连接器：`graph_to_lm_connector.pt`、`lm_to_graph_decoder.pt`、`lm_to_graph_predictor.pt`

## 3. Stage 2 — GBM Domain Adapter

**来源：** `saves/Llamole-Qwen2-7B-Instruct-Adapter-gbm-with-graph-models`

**功能：** GBM 领域专业知识 + 分子生成能力

**LoRA 配置：** 同 Stage 1（r=16, lora_alpha=32）

**状态：** 保持为 `PeftModel`，**未合并**（与 Stage 1 不同）

**原因：** Stage 2 Adapter 需要在推理时动态切换：
- 正常推理：使用 Stage 2 GBM Adapter（PeftModel）
- SMILES Fallback：临时切换回 Stage 1 合并模型 + few-shot prompt（当微调模型无有效 SMILES 输出时）

## 4. 训练数据格式

```python
# 格式1（新格式）：instruction + response（评估指标导向）
f"Instruction: {instruction}\nResponse: {response}"

# 格式2（旧格式）：instruction + input + output
f"Instruction: {instruction}\nInput: {input}\nResponse: {output}"
```

训练参数（来自 `TrainingArguments`）：
- `output_dir`: 输出目录
- `num_train_epochs`: 训练轮数
- `per_device_train_batch_size`: 批次大小
- `learning_rate`: 学习率
- `warmup_ratio`: 预热比例
- `logging_steps`: 日志记录步数
- `save_steps`: 保存步数
- `fp16`: 启用 FP16 混合精度

## 5. 特殊设计

### 5.1 推理时的模型切换

```python
# 正常推理：使用 GBM Domain Adapter（PeftModel）
self.model = PeftModel.from_pretrained(
    merged_base_model,
    self.gbm_adapter_path  # Stage 2 PeftModel
)

# SMILES Fallback：切换回 Stage 1 合并模型 + few-shot
if fallback_triggered:
    self.model = merged_base_model  # Stage 1 合并后模型
    prompt = add_few_shot_examples(original_prompt)
```

### 5.2 K-bit 训练支持

可选通过 `prepare_model_for_kbit_training()` 准备模型用于量化训练（节省显存），可通过环境变量 `SKIP_KBIT_PREP=1` 跳过以减少显存峰值。

### 5.3 Llamole 图模型集成

当 `use_llamole_graph_models=True` 时，主模型从 `AutoModelForCausalLM` 升级为 `GraphLLMForCausalMLM`，将图模型组件通过连接器与语言模型深度融合，增强药物-细胞连接性预测和分子图生成能力。

## 6. 在 ToT 系统中的角色

```
TreeOfThoughtsGenerator
        │
        ├── self.model = Stage 1 合并模型
        │       ↓ PeftModel.from_pretrained()
        │   Stage 2 GBM Adapter（PeftModel，未合并）
        │       ↓
        │   self.model.generate(prompt, ...)
        │
        └── ToT 控制流
            │
            ├── Level 0: LLM 生成 Scaffold（调用微调模型）
            ├── Level 1: LLM 生成 Assembly（调用微调模型）
            └── Level 2: LLM 生成 SMILES（调用微调模型）
                          ↑
                    fallback: 切换 Stage 1 + few-shot
```
