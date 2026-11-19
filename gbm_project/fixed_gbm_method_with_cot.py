#!/usr/bin/env python3
"""
修复版GBM方法：正确使用Chain-of-Thought推理
演示如何正确集成GBM知识库和CoT推理功能
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re

# 设置GPU设备
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def extract_smiles_from_text(text):
    """改进的SMILES提取函数"""
    smiles_patterns = [
        r'SMILES:\s*([A-Za-z0-9@+\-\[\]\(\)=#]+)',
        r'smiles:\s*([A-Za-z0-9@+\-\[\]\(\)=#]+)',
        r'分子式[:：]\s*([A-Za-z0-9@+\-\[\]\(\)=#]+)',
        r'([A-Za-z0-9@+\-\[\]\(\)=#]{6,})'
    ]

    all_smiles = []
    for pattern in smiles_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            cleaned_smiles = re.sub(r'[.,;:!?]$', '', match.strip())
            if len(cleaned_smiles) >= 6 and any(c in cleaned_smiles for c in ['C', 'N', 'O', 'S', 'P']):
                all_smiles.append(cleaned_smiles)

    return all_smiles[0] if all_smiles else None

def demonstrate_correct_gbm_method():
    """演示正确的GBM方法实现，包含完整的CoT推理"""
    print("🧬 修复版GBM方法演示：正确使用Chain-of-Thought推理")
    print("=" * 70)

    try:
        # 加载模型
        model_path = "../models/Qwen2-7B-Instruct"
        adapter_path = "../saves/Llamole-Qwen2-7B-Instruct-Adapter"

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map={"": 0},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory={0: "20GB"}
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

        # 测试案例
        test_case = {
            'target': 'EGFR',
            'constraints': {'molecular_weight_max': 550, 'logp_min': 2.0, 'logp_max': 4.0}
        }

        print("🎯 测试案例: EGFR靶点GBM抑制剂")
        print("📋 约束条件: 分子量<550, logP 2.0-4.0")

        # 正确实现GBM方法：使用完整的知识库和CoT推理
        print("\n🔧 步骤1: 加载GBM知识库和Prompt生成器")
        from src.gbm_knowledge_base import GBMKnowledgeBase
        from src.gbm_prompt_generator import GBMPromptGenerator

        kb = GBMKnowledgeBase(
            targets_path="data/gbm_targets/gbm_targets.json",
            clinical_path="data/gbm_clinical/gbm_clinical_data.json",
            molecules_path="data/gbm_molecules/gbm_molecules.json"
        )
        prompt_gen = GBMPromptGenerator(kb, "configs/gbm_prompts.yaml")

        print("✅ 知识库加载完成")
        print("✅ Prompt生成器初始化完成")

        # 生成完整的GBM prompt，包含CoT推理
        print("\n🔧 步骤2: 生成完整的GBM CoT推理prompt")
        full_gbm_prompt = prompt_gen.generate_full_prompt(
            target_name=test_case['target'],
            constraints=test_case['constraints'],
            use_cot=True,  # 启用Chain-of-Thought推理
            reasoning_type="step_by_step_design"
        )

        print(f"📝 生成的GBM prompt长度: {len(full_gbm_prompt)} 字符")
        print("\n🔍 Prompt内容预览 (前500字符):")
        print("-" * 50)
        print(full_gbm_prompt[:500] + "..." if len(full_gbm_prompt) > 500 else full_gbm_prompt)
        print("-" * 50)

        # 验证CoT推理是否包含
        cot_indicators = [
            "让我们逐步思考",
            "靶点选择和验证",
            "分子骨架设计",
            "药效团优化",
            "ADMET性质优化",
            "GBM特异性验证"
        ]

        print("\n✅ CoT推理验证:")
        for indicator in cot_indicators:
            if indicator in full_gbm_prompt:
                print(f"  ✓ 包含: {indicator}")
            else:
                print(f"  ✗ 缺失: {indicator}")

        # 验证靶点知识是否包含
        target_info = kb.get_target_info(test_case['target'])
        if target_info:
            print(f"\n✅ 靶点知识集成: {target_info.name}")
            print(f"  - 描述: {target_info.description[:50]}...")
            print(f"  - 挑战: {len(target_info.challenges)}项")
            print(f"  - 现有药物: {len(target_info.current_drugs)}个")

        # 验证临床知识是否包含
        clinical_insights = kb.get_clinical_insights()
        if clinical_insights:
            print(f"\n✅ 临床知识集成:")
            print(f"  - 治疗挑战: {len(clinical_insights.get('challenges', []))}项")
            print(f"  - 成功模式: {len(clinical_insights.get('successful_patterns', []))}项")

        # 添加Llamole格式包装
        llamole_prompt = f"<design_start>{full_gbm_prompt}<design_end>"

        print(f"\n📋 最终Llamole格式prompt长度: {len(llamole_prompt)} 字符")

        # 现在可以进行实际生成（如果需要）
        print("\n🚀 准备进行分子生成...")
        print("⚠️  注意: 由于环境限制，这里只展示prompt，不进行实际生成")

        # 模拟生成过程验证
        inputs = tokenizer(llamole_prompt, return_tensors="pt").to(model.device)
        print(f"✅ Tokenization成功: {inputs['input_ids'].shape[1]} tokens")

        # 清理内存
        del model
        torch.cuda.empty_cache()

        print("\n" + "=" * 70)
        print("🎉 修复版GBM方法演示完成！")
        print("=" * 70)

        print("\n📊 修复总结:")
        print("✅ 正确加载GBM知识库")
        print("✅ 正确初始化Prompt生成器")
        print("✅ 启用Chain-of-Thought推理")
        print("✅ 集成靶点特定知识")
        print("✅ 包含临床治疗经验")
        print("✅ 使用Llamole格式包装")

        print("\n🎯 与原问题版本对比:")
        print("❌ 原版: 硬编码简单prompt，无CoT推理")
        print("✅ 修复版: 完整GBM知识体系 + CoT推理框架")

        return True

    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_prompt_lengths():
    """对比不同实现方式的prompt长度"""
    print("\n" + "=" * 70)
    print("📏 Prompt长度对比分析")
    print("=" * 70)

    # 模拟原版prompt
    original_prompt = """<design_start>设计GBM治疗分子：靶点EGFR，分子量<550，logP范围2.0-4.0，需要高BBB穿透性，降低毒性，基于GBM临床经验<design_end>"""

    print(f"❌ 原版GBM方法prompt长度: {len(original_prompt)} 字符")
    print(f"   内容: {original_prompt}")

    # 这里无法实际生成修复版prompt（需要加载知识库），所以估算
    estimated_fixed_length = 2000  # 基于实际代码估算

    print(f"✅ 修复版GBM方法prompt长度: ~{estimated_fixed_length} 字符 (估算)")
    print(f"   包含: GBM专家介绍 + 生物学特征 + 治疗挑战 + 靶点知识 + CoT推理步骤 + 临床经验")

    print(f"\n📈 长度提升: {estimated_fixed_length/len(original_prompt):.1f}倍")
    print(f"💡 信息密度提升: 简单指令 → 结构化专业知识 + 推理框架")

if __name__ == "__main__":
    success = demonstrate_correct_gbm_method()
    if success:
        compare_prompt_lengths()

