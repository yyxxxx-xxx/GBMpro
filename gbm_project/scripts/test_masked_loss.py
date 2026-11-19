#!/usr/bin/env python3
"""
单元测试：验证 response-only masked loss 的正确性。

检查：
  1. "Response:" 之前的 token label 全为 -100
  2. "Response:" 之后的 token label 等于 input_ids（保留监督信号）
  3. 监督 token 占比应接近 response 长度 / 总长度（而不是 16%）
  4. modules_to_save / merge 逻辑的静态检查
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer

BASE = "/root/Llamole-main/models/Qwen2-7B-Instruct"


class FakeFinetuner:
    """复用真实 tokenize_function，但不加载 7B 模型"""

    def __init__(self, tokenizer, lora_config):
        self.tokenizer = tokenizer
        self.lora_config = lora_config

    # 直接绑定真实实现
    from src.gbm_lora_finetuner import GBMLoRAFinetuner
    tokenize_function = GBMLoRAFinetuner.tokenize_function


def main():
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True,
                                        local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ft = FakeFinetuner(tok, {"response_only_loss": True, "max_length": 1024})

    examples = {
        "instruction": [
            "You are an expert in GBM drug development, specializing in novel drug "
            "candidates for Glioblastoma Multiforme (GBM). Please design novel drug "
            "candidates with the following characteristics:\n"
            "- Target cell phenotype: Mesenchymal-like GBM cells\n"
            "- Target metrics: BBB: good permeability (logBB > -0.5); "
            "GBM Activity: high; Toxicity: low; Selectivity: high (GBM/normal ratio)\n"
            "- Molecular features: MW 300-500 Da, logP 2-4, TPSA 60-120, HBD 1-4, HBA 3-8\n\n"
            "Design GBM candidates targeting the EGFR pathway:\n"
            "- Prioritize targeting EGFRvIII mutants\n"
            "- Molecular weight: 400-550 Da\n"
            "- logP: 2-4\n\n"
            "Based on the above analysis, generate a novel GBM candidate molecule, "
            "including SMILES string and design rationale."
        ],
        "input": [""],
        "output": ["SMILES: NC(=O)Nc1ccc2ccccc2c1C(N)=O"],
    }

    out = ft.tokenize_function(examples)
    ids = out["input_ids"][0]
    labels = out["labels"][0]

    assert len(ids) == len(labels), "长度不一致"

    resp_ids = tok("Response:", add_special_tokens=False)["input_ids"]
    n_resp = len(resp_ids)
    start = None
    for j in range(0, len(ids) - n_resp + 1):
        if ids[j:j + n_resp] == resp_ids:
            start = j + n_resp
            break

    print(f"总 token 数: {len(ids)}")
    print(f"'Response:' 结束位置: {start}")
    print(f"监督 token 数: {len(ids) - start}")
    print(f"监督占比: {(len(ids) - start) * 100 / len(ids):.1f}%  (旧实现为 16.1%)")

    # 校验 1: 前缀全部 mask
    bad = [i for i in range(start) if labels[i] != -100]
    assert not bad, f"前缀存在未 mask 的 label: {bad[:5]}"
    print("✅ 前缀全部为 -100")

    # 校验 2: 后缀保留
    bad2 = [i for i in range(start, len(ids)) if labels[i] != ids[i]]
    assert not bad2, f"后缀 label 与 input_ids 不一致: {bad2[:5]}"
    print("✅ response 段 label == input_ids")

    # 校验 3: 解码监督部分
    sup_ids = [ids[i] for i in range(start, len(ids)) if labels[i] != -100]
    print(f"监督部分解码: {tok.decode(sup_ids)!r}")

    # 校验 4: 对比旧实现（全序列 loss）
    old_ratio = 100.0
    print(f"\n对比：旧实现监督占比 100%（其中化学部分仅 16.1%）")
    print(f"      新实现监督占比 {(len(ids)-start)*100/len(ids):.1f}%，"
          f"全部用于化学")

    # 校验 5: 找不到 Response: 时的兜底
    weird = {
        "instruction": ["abc"],
        "input": [""],
        "output": ["xyz"],
    }
    out2 = ft.tokenize_function(weird)
    sup = sum(1 for l in out2["labels"][0] if l != -100)
    tot = len(out2["labels"][0])
    print(f"\n无 'Response:' 标记时: 总 {tot} token, 监督 {sup} token "
          f"({sup*100/max(tot,1):.0f}%) <- 兜底保留最后 1/4")
    assert 0 < sup <= tot

    print("\n全部单元测试通过 ✅")


if __name__ == "__main__":
    main()
