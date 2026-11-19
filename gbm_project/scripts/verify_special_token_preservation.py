#!/usr/bin/env python3
"""
验证 merge_and_unload 是否完整保留了 Llamole 的特殊 token embedding。

背景
----
Llamole 适配器里含 modules_to_save(['embed_tokens','lm_head'])，其中
embed_tokens 的第 151646-151654 行是新增的化学特殊 token
(<molecule> / <design_body> / <design_start> / ...)。

如果不带 modules_to_save 训练新的 GBM LoRA，这些特殊 token 的 embedding
是否会丢？答案取决于 merge_and_unload 是否把它们吸收进基座。

这个问题决定了 GBM LoRA 要不要开 modules_to_save —— 因为开了之后
embed_tokens(152064x3584) + lm_head(152064x3584) = 1.09B 参数可训练，
在 ~1700 条样本上极易过拟合。

本脚本直接比对 merge 前后的特殊 token 行。
"""
import sys
import warnings

import torch

sys.path.insert(0, "/root/Llamole-main")
warnings.filterwarnings("ignore")

BASE = "/root/Llamole-main/models/Qwen2-7B-Instruct"
LLAMOLE = "/root/Llamole-main/saves/Llamole-Qwen2-7B-Instruct-Adapter"
SPECIAL_ROWS = slice(151646, 151655)
NORMAL_ROWS = slice(0, 5)


def main():
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from safetensors.torch import load_file

    print("=" * 70)
    print("验证 merge 是否保留 Llamole 特殊 token embedding")
    print("=" * 70)

    st = load_file(f"{LLAMOLE}/adapter_model.safetensors")
    adapter_emb = st["base_model.model.model.embed_tokens.weight"].float()
    adapter_head = st["base_model.model.lm_head.weight"].float()
    print(f"\n适配器 embed_tokens shape: {tuple(adapter_emb.shape)}")
    print(f"  特殊 token 行 {SPECIAL_ROWS} 绝对值均值: "
          f"{float(adapter_emb[SPECIAL_ROWS].abs().mean()):.6f}")
    print(f"  普通 token 行 {NORMAL_ROWS} 绝对值均值: "
          f"{float(adapter_emb[NORMAL_ROWS].abs().mean()):.6f}")

    print("\n加载基座 ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float32, device_map="cpu",
        trust_remote_code=True, local_files_only=True)

    base_emb = model.get_input_embeddings().weight.float().clone()
    base_head = model.get_output_embeddings().weight.float().clone()
    print(f"  基座 embed 特殊 token 行绝对值均值: "
          f"{float(base_emb[SPECIAL_ROWS].abs().mean()):.6f}")
    print(f"  基座 embed 普通 token 行绝对值均值: "
          f"{float(base_emb[NORMAL_ROWS].abs().mean()):.6f}")

    print("\nmerge_and_unload ...")
    model = PeftModel.from_pretrained(model, LLAMOLE)
    model = model.merge_and_unload()

    merged_emb = model.get_input_embeddings().weight.float()
    merged_head = model.get_output_embeddings().weight.float()

    # 特殊 token 行：merge 后应 == adapter 的值
    d_special = float((merged_emb[SPECIAL_ROWS] - adapter_emb[SPECIAL_ROWS]).abs().max())
    d_normal = float((merged_emb[NORMAL_ROWS] - adapter_emb[NORMAL_ROWS]).abs().max())
    d_base_special = float((base_emb[SPECIAL_ROWS] - adapter_emb[SPECIAL_ROWS]).abs().max())

    print(f"\nmerge 后 embed_tokens:")
    print(f"  特殊 token 行 vs 适配器: max|diff| = {d_special:.3e}")
    print(f"  普通 token 行 vs 适配器: max|diff| = {d_normal:.3e}")
    print(f"  (merge 前 特殊 token 行 vs 适配器: max|diff| = {d_base_special:.3e})")

    print(f"\nmerge 后 lm_head:")
    d_head = float((merged_head - adapter_head).abs().max())
    print(f"  全部行 vs 适配器: max|diff| = {d_head:.3e}")

    ok_special = d_special < 1e-6
    ok_normal = d_normal < 1e-6
    ok_head = d_head < 1e-6

    print("\n结论:")
    print(f"  特殊 token embedding 被 merge 完整保留: {'✅' if ok_special else '❌'}")
    print(f"  普通 token embedding 与适配器一致:     {'✅' if ok_normal else '❌'}")
    print(f"  lm_head 与适配器一致:                 {'✅' if ok_head else '❌'}")

    if ok_special and ok_head:
        print("""
=> merge_and_unload 已把 embed_tokens / lm_head 全部吸收进基座。
   因此 GBM LoRA **不需要** 开 modules_to_save：
   特殊 token 表示不会丢，且避免 1.09B 参数在 ~1700 样本上过拟合。
=> 但推理时必须使用含特殊 token 的 tokenizer（Llamole 适配器目录里的），
   否则 <molecule> 等会被切成普通子词。
""")
    else:
        print("""
=> merge 未完整保留特殊 token 表示，
   GBM LoRA 需要开 modules_to_save=['embed_tokens','lm_head']。
""")


if __name__ == "__main__":
    main()
