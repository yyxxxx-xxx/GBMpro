# 🧬 GBM分子生成项目

基于多模态大语言模型的胶质母细胞瘤（GBM）新药候选分子生成与优化研究

## 📋 项目概述

本项目利用Llamole多模态大语言模型，结合GBM领域专业知识和Chain-of-Thought推理，生成针对胶质母细胞瘤的新药候选分子。项目通过prompt注入GBM治疗领域的专业知识，提高模型在GBM药物设计任务上的性能和质量。

## 🎯 核心特性

- **GBM领域知识注入**：集成GBM靶点、病理机制、临床治疗经验
- **Chain-of-Thought推理**：引导模型进行GBM相关的分子设计推理
- **多维度评估体系**：BBB穿透性、GBM活性、毒性、安全性综合评估
- **基准测试框架**：系统比较不同方法的性能表现
- **完整实验流程**：从数据准备到结果分析的端到端解决方案

## 📁 项目结构

```
gbm_project/
├── configs/                    # 配置文件
│   ├── gbm_generation.yaml    # GBM生成配置
│   └── gbm_prompts.yaml       # GBM专业prompt模板
├── data/                      # GBM领域数据
│   ├── gbm_targets/           # GBM靶点信息
│   ├── gbm_clinical/          # GBM临床数据
│   └── gbm_molecules/         # GBM相关分子
├── src/                       # 核心模块
│   ├── gbm_knowledge_base.py  # GBM知识库
│   ├── gbm_prompt_generator.py # GBM prompt生成器
│   ├── gbm_evaluator.py       # GBM专项评估器
│   └── gbm_generator.py       # GBM分子生成器
└── experiments/               # 实验脚本
    ├── generate_gbm_candidates.py    # 候选分子生成
    ├── evaluate_gbm_candidates.py    # 结果评估分析
    └── benchmark_experiments.py      # 基准测试
```

## 🚀 快速开始

### 环境要求

- Python 3.11+
- PyTorch (支持CUDA)
- RDKit
- Transformers
- 其他依赖见 `requirements.txt`

### 安装依赖

```bash
# 在Llamole-main目录下
pip install -r requirements.txt
```

### 下载预训练模型

确保已下载Llamole所需的预训练模型：
- Qwen2-7B-Instruct
- Graph Diffusion Transformer
- Graph Encoder/Predictor

### 运行基础实验

1. **生成GBM候选分子**
```bash
cd gbm_project/experiments
python generate_gbm_candidates.py \
    --config ../configs/gbm_generation.yaml \
    --num_candidates 20 \
    --target EGFR \
    --output_dir ./results
```

2. **评估生成结果**
```bash
python evaluate_gbm_candidates.py \
    --results_file ./results/generated_molecules.json \
    --output_dir ./analysis \
    --generate_plots
```

3. **运行基准测试**
```bash
python benchmark_experiments.py \
    --base_config ../configs/gbm_generation.yaml \
    --test_scenarios baseline gbm_knowledge combined \
    --num_candidates_per_test 10 \
    --repeat_times 3
```

## 📊 评估指标

### 综合评估体系

- **BBB穿透性**：预测血脑屏障通透性 (logBB)
- **GBM细胞活性**：预测对GBM细胞的生长抑制活性
- **正常细胞毒性**：评估对正常细胞的毒性
- **选择性指数**：GBM活性 vs 正常细胞毒性比值
- **合成可行性**：基于分子复杂度的合成难度评分
- **临床相似度**：与已知GBM药物的结构相似性

### 综合得分计算

```
综合得分 = 0.25×BBB评分 + 0.25×GBM活性 + 0.20×毒性评分 +
          0.15×合成可行性 + 0.15×临床相似度
```

## 🔬 实验场景

### 基准测试场景

1. **Baseline**: 基础Llamole生成（无GBM优化）
2. **GBM Knowledge**: 仅使用GBM领域知识注入
3. **CoT Reasoning**: 仅使用Chain-of-Thought推理
4. **Combined**: GBM知识 + CoT推理（完整方法）
5. **Target Specific**: 特定靶点优化（EGFR等）
6. **Constraints Optimization**: 约束条件优化

### 靶点覆盖

- EGFR (表皮生长因子受体)
- VEGF/VEGFR (血管内皮生长因子)
- IDH1/IDH2 (异柠檬酸脱氢酶)
- MGMT (O6-甲基鸟嘌呤-DNA甲基转移酶)
- GBM Stem Cells (GBM干细胞)
- PD-1/PD-L1 (免疫检查点)
- PI3K/AKT/mTOR (PI3K信号通路)
- p53/MDM2 (p53通路)

## 📈 预期结果

### 性能提升

- **生成质量**: GBM候选分子综合得分提升30-50%
- **BBB穿透性**: 预测准确率提升25%
- **GBM活性**: 细胞活性预测准确率提升35%
- **临床相关性**: 生成分子与已知药物的相似度提升40%

### 新药发现潜力

- 生成1000+个GBM候选分子
- 其中50+个分子具有临床开发潜力
- 3-5个lead compound进入 preclinical evaluation

## 🛠️ 高级用法

### 自定义GBM知识库

```python
from src.gbm_knowledge_base import GBMKnowledgeBase

# 加载自定义知识库
kb = GBMKnowledgeBase(targets_path, clinical_path, molecules_path)

# 添加新靶点
new_target = {
    "name": "Custom_Target",
    "description": "自定义靶点描述",
    "structural_requirements": {
        "molecular_weight": "<500",
        "logP": "2-4"
    }
}
```

### 自定义评估器

```python
from src.gbm_evaluator import GBMEvaluator

# 创建自定义评估器
evaluator = GBMEvaluator(reference_molecules=custom_references)

# 评估分子
result = evaluator.evaluate_molecule(smiles_string)
print(f"综合得分: {result['evaluation']['scores']['composite_score']:.3f}")
```

### 集成到现有工作流

```python
from src.gbm_generator import GBMGenerator

# 初始化生成器
generator = GBMGenerator(config_path)

# 生成靶点特异性分子
results = generator.generate_gbm_molecules(
    num_candidates=50,
    target_name="EGFR",
    constraints={"molecular_weight_max": 500},
    use_cot=True
)

# 保存结果
generator.save_results(results, "gbm_candidates.json")
```

## 📋 配置说明

### GBM生成配置 (`gbm_generation.yaml`)

```yaml
# 模型配置
model_name_or_path: Qwen/Qwen2-7B-Instruct
adapter_name_or_path: saves/Llamole-Qwen2-7B-Instruct-Adapter

# GBM专项配置
gbm_generation_mode: true
use_gbm_domain_knowledge: true
use_cot_reasoning: true

# 靶点权重
gbm_target_weights:
  EGFR: 0.25
  VEGF_VEGFR: 0.20
  IDH1_IDH2: 0.15

# 分子约束
molecular_constraints:
  molecular_weight_max: 600
  logp_min: 1.0
  logp_max: 4.0
```

### Prompt模板配置 (`gbm_prompts.yaml`)

支持自定义：
- 基础GBM prompt
- 靶点特定prompt
- CoT推理模板
- 约束条件模板
- 评估prompt

## 🔍 故障排除

### 常见问题

1. **模型加载失败**
   - 检查HuggingFace token权限
   - 确认模型文件路径正确
   - 验证CUDA环境可用

2. **分子解析错误**
   - 检查SMILES字符串格式
   - 验证RDKit安装完整
   - 查看分子有效性

3. **评估结果异常**
   - 检查分子性质计算
   - 验证评估器参数设置
   - 查看日志输出

4. **内存不足**
   - 减少batch_size
   - 使用CPU模式
   - 分批处理分子

### 调试模式

```bash
# 启用详细日志
export PYTHONPATH=/path/to/llamole-main:$PYTHONPATH
python experiments/generate_gbm_candidates.py --verbose --seed 42
```

## 📚 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@misc{gbm_llamole_2024,
  title={基于多模态大语言模型的胶质母细胞瘤新药候选分子生成与优化研究},
  author={Your Name},
  year={2024},
  note={基于Llamole框架的GBM专项应用}
}
```

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！

### 开发指南

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📄 许可证

本项目采用Apache 2.0许可证 - 查看 [LICENSE](../LICENSE) 文件了解详情。

---

## 📞 联系方式

项目维护者：GBM药物设计研究组

如有问题，请通过GitHub Issues联系。
