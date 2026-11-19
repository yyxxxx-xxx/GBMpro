#!/usr/bin/env python3
"""
调试GBM模型生成内容
检查模型实际生成的文本内容
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# 设置GPU设备为GPU 2
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def debug_generation():
    """调试模型生成内容"""
    print("🔍 开始调试GBM模型生成内容")

    # 加载模型和tokenizer
    model_path = "../models/Qwen2-7B-Instruct"
    adapter_path = "../saves/Llamole-Qwen2-7B-Instruct-Adapter"

    print("🤖 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    print("✓ 模型加载完成")

    # 测试不同的prompt
    test_prompts = [
        "请设计一个药物分子，给出SMILES字符串。",
        """请设计一个针对EGFR的药物分子。

要求：
- 分子量在500 Da以内
- logP在2-4范围内

请给出SMILES字符串。

SMILES:""",
        "Design a drug molecule for GBM treatment. Provide SMILES string."
    ]

    for i, prompt in enumerate(test_prompts):
        print(f"\n{'='*50}")
        print(f"测试Prompt {i+1}:")
        print(prompt)
        print(f"{'='*50}")

        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,  # 减少长度以便调试
                    temperature=0.1,     # 降低随机性
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("生成的完整文本:")
            print(repr(generated_text))
            print("\n生成的文本内容:")
            print(generated_text)

            # 检查是否包含SMILES
            import re
            smiles_patterns = [
                r'SMILES:\s*([^\s\n]+)',
                r'smiles:\s*([^\s\n]+)',
                r'([A-Za-z0-9@+\-\[\]\(\)=#]{5,})',  # 通用SMILES模式
            ]

            for pattern_name, pattern in [("SMILES标签", smiles_patterns[0]),
                                        ("小写smiles", smiles_patterns[1]),
                                        ("通用SMILES", smiles_patterns[2])]:
                matches = re.findall(pattern, generated_text)
                if matches:
                    print(f"✓ 找到{patter_name}: {matches}")

        except Exception as e:
            print(f"❌ 生成失败: {e}")

if __name__ == "__main__":
    debug_generation()
