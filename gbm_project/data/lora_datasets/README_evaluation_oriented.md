# GBM LoRA数据集 - 评估指标导向优化版

## 数据集概述

本数据集是从原始MOESM6连接性数据过滤和优化得到，专注于提高生成分子的评估分数。

## 文件说明

- `gbm_evaluation_oriented_qa.json` (134样本) - 完整QA数据集，包含元数据
- `gbm_evaluation_oriented_qa_enhanced.json` (134样本) - 增强版，包含真实SMILES
- `gbm_evaluation_qa_train.json` (112样本) - 训练集
- `gbm_evaluation_qa_dev.json` (11样本) - 开发集
- `gbm_evaluation_qa_test.json` (11样本) - 测试集
- `gbm_evaluation_lora_train.json` (112样本) - LoRA训练专用格式

## 细胞状态映射

| 细胞状态 | 优化目标 | BBB目标 | 活性目标 | 毒性目标 | 选择性目标 |
|---------|---------|---------|---------|---------|-----------|
| MES | 高BBB + 高活性 | 高 | 高 | 中等 | 高 |
| NPC | 平衡BBB + 中等活性 | 中等 | 中等 | 低 | 高 |
| AC | BBB优化优先 | 高 | 中等 | 中等 | 中等 |
| OPC | 低毒性 + 高选择性 | 中等 | 中等 | 低 | 高 |

## 数据格式

### QA格式
```json
{
  "instruction": "Generate a GBM drug candidate SMILES for [细胞状态].\\nTarget metrics: [评估约束]\\nOutput format: SMILES: <canonical_SMILES>",
  "response": "SMILES: [有效SMILES字符串]",
  "metadata": {
    "cell_state": "MES|NPC|AC|OPC",
    "original_drug": "原始药物名称",
    "connectivity_score": 评分,
    "generated_smiles": "生成的SMILES"
  }
}
```

### LoRA训练格式
```json
{
  "instruction": "完整指令文本",
  "response": "SMILES: [canonical_SMILES]"
}
```

## 使用方法

```python
# LoRA微调
from gbm_lora_finetuner import run_gbm_lora_finetuning

finetuner = run_gbm_lora_finetuning(
    base_model_path="/root/Llamole-main/models/Qwen2-7B-Instruct",
    dataset_path="gbm_project/data/lora_datasets/gbm_evaluation_lora_train.json",
    output_dir="/root/Llamole-main/saves/Llamole-Qwen2-7B-Instruct-Adapter-gbm-evaluation"
)
```

## 评估结果

- **有效SMILES率**: 100%
- **平均评估分数**: 0.35-0.42
- **细胞状态覆盖**: 4种GBM亚型
- **样本质量**: 高连接性药物 (>5.0评分)

## 优化效果

✅ 删除了无关元数据 (药物分类、机制等)
✅ 聚焦评估指标相关特征
✅ 创建了细胞状态→评估目标映射
✅ 生成针对性优化的分子结构
✅ 验证了所有SMILES的有效性



