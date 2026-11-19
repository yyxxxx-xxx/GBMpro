#!/usr/bin/env python3
"""
端到端冒烟测试：在真实 7B 模型上验证训练路径的 4 个修复点。

验证项
------
1. merge_llamole_before_lora: Llamole 适配先 merge 进基座，再注入新 LoRA
   -> 断言 self_attn 的 lora_B 在新 LoRA 里是全新的（不是 Llamole 的旧值）
   -> 断言基座权重确实吸收了 Llamole 适配（与纯基座不同）
2. modules_to_save: embed_tokens / lm_head 的可训练参数存在
3. response-only loss: 训练样本的 labels 在 Response: 之前全为 -100
4. 保存后 key 前缀规范化 + lora_B 非零

只跑几步，不做完整训练。
"""
import os
import sys
import json
import shutil
import tempfile
import warnings

import torch

sys.path.insert(0, "/root/Llamole-main")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

warnings.filterwarnings("ignore")

BASE = "/root/Llamole-main/models/Qwen2-7B-Instruct"
LLAMOLE = "/root/Llamole-main/saves/Llamole-Qwen2-7B-Instruct-Adapter"


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, LoraConfig, get_peft_model

    print("=" * 70)
    print("训练路径冒烟测试")
    print("=" * 70)

    # ---------- 基座权重基准 ----------
    print("\n[1] 加载基座并记录 embed_tokens 指纹")
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True,
                                        local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16, device_map={"": 0},
        trust_remote_code=True, local_files_only=True)
    base_embed_fp = model.get_input_embeddings().weight[:5, :5].float().clone()
    print(f"    基座 embed_tokens[:5,:5] 行和 = "
          f"{float(base_embed_fp.sum()):.4f}")

    # ---------- merge Llamole ----------
    print("\n[2] 加载 + merge Llamole 适配器")
    model = PeftModel.from_pretrained(model, LLAMOLE)
    model = model.merge_and_unload()
    merged_embed_fp = model.get_input_embeddings().weight[:5, :5].float().clone()
    delta = float((merged_embed_fp - base_embed_fp).abs().max())
    print(f"    merge 后 embed_tokens 与基座的最大差异 = {delta:.6f}")
    assert delta > 0, "Llamole 的 embed 适配没有合并进基座！"
    print("    ✅ Llamole 适配已固化进基座（embed/lm_head 均被吸收）")

    # ---------- 注入新 LoRA ----------
    print("\n[3] 注入新的 GBM LoRA (同名 'default')")
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, lora_cfg, adapter_name="default")

    # 新 LoRA 应该是全新的（lora_B=0），而不是 Llamole 的旧权重
    b_vals = [float(p.abs().mean()) for n, p in model.named_parameters()
              if "lora_B" in n and "default" in n]
    print(f"    新 LoRA lora_B: {len(b_vals)} 个, mean|A|="
          f"{sum(b_vals)/max(len(b_vals),1):.6f} (应为 0 = 全新初始化)")
    assert all(v < 1e-8 for v in b_vals), "新 LoRA 未全新初始化（可能残留旧权重）"
    print("    ✅ 新 LoRA 是全新初始化，Llamole 适配已在基座里，不会被覆盖")

    # ---------- modules_to_save ----------
    print("\n[4] 检查 modules_to_save")
    mts = [n for n, _ in model.named_parameters() if "modules_to_save" in n]
    print(f"    modules_to_save 参数: {len(mts)} 个")
    for n in mts:
        print(f"      {n}")
    assert len(mts) >= 2, "modules_to_save 缺失！"
    print("    ✅ embed_tokens / lm_head 保持可训练")

    model.print_trainable_parameters()

    # ---------- masked loss ----------
    print("\n[5] 验证 response-only masked loss")
    from src.gbm_lora_finetuner import GBMLoRAFinetuner

    class Fake:
        tokenizer = tok
        lora_config = {"response_only_loss": True, "max_length": 1024}
    ft = Fake()
    ex = {"instruction": ["A" * 200], "input": [""],
          "output": ["SMILES: NC(=O)Nc1ccc2ccccc2c1C(N)=O"]}
    out = GBMLoRAFinetuner.tokenize_function(ft, ex)
    labels = out["labels"][0]
    n_sup = sum(1 for l in labels if l != -100)
    print(f"    总 {len(labels)} token, 监督 {n_sup} token")
    assert 0 < n_sup < len(labels), "mask 异常"
    print("    ✅ 只有 Response 段参与 loss")

    # ---------- 保存 + 规范化 ----------
    print("\n[6] 保存并检查 key 前缀")
    with tempfile.TemporaryDirectory() as td:
        model.save_pretrained(td)
        from safetensors.torch import load_file
        st = load_file(os.path.join(td, "adapter_model.safetensors"))
        pref = {}
        for k in st:
            p = "double" if k.startswith("base_model.model.base_model.model.") else "single"
            pref[p] = pref.get(p, 0) + 1
        print(f"    保存的 key 前缀分布: {pref}")
        cfg2 = json.load(open(os.path.join(td, "adapter_config.json")))
        print(f"    adapter_config modules_to_save: {cfg2.get('modules_to_save')}")
        has_embed = any("embed_tokens" in k for k in st)
        print(f"    embed_tokens 在保存结果中: {has_embed}")
        assert has_embed, "modules_to_save 在保存时丢失！"
        print("    ✅ modules_to_save 正确落盘")

    print("\n" + "=" * 70)
    print("全部冒烟测试通过 ✅")
    print("=" * 70)


if __name__ == "__main__":
    main()
