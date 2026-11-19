"""
多适配器加载（Llamole + GBM LoRA）
==================================

问题背景:
    Llamole 适配器的 safetensors key 前缀是
        base_model.model.model.embed_tokens.weight
    GBM 适配器的 key 前缀是
        base_model.model.base_model.model.model.embed_tokens.weight
    （多了一层 base_model.，因为它是"在已套 PeftModel 的模型上再套一层"保存的）

    直接 PeftModel.from_pretrained(base, gbm_adapter) 会因为前缀不匹配而报
        KeyError: 'base_model.model.base_model.model.model.embed_tokens.weight'
    典型表现是加载 embed_tokens 时报 size mismatch。

解决方案:
    1) 先把 Llamole 适配器 merge 进基座（merge_and_unload），得到干净的基座模型
    2) 再把 GBM 适配器的 key 前缀规范化后加载
    3) 注意: 两者 target_modules 不同
       - Llamole: q/k/v/o + gate/up/down_proj (全 MLP)
       - GBM    : 仅 q/k/v/o_proj
       所以它们可以叠加而不冲突。

    若 GBM 适配器与 Llamole 适配器都含 embed_tokens/lm_head（vocab resize 结果），
    需要额外处理 vocab 尺寸 —— 见 _normalize_key_prefix 中的 embed 逻辑。
"""

import os
import re
import warnings
from typing import List, Optional, Tuple

import torch
from safetensors.torch import load_file

warnings.filterwarnings("ignore")


def _read_adapter_state(adapter_path: str) -> dict:
    """读取适配器 safetensors（同时支持 .safetensors 和 .bin）。"""
    st_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if os.path.isfile(st_path):
        return load_file(st_path)
    bin_path = os.path.join(adapter_path, "adapter_model.bin")
    if os.path.isfile(bin_path):
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"未找到适配器权重: {adapter_path}")


def _detect_prefix(state: dict) -> str:
    """检测适配器的 key 前缀风格。"""
    has_double = any(k.startswith("base_model.model.base_model.model.") for k in state)
    return "double" if has_double else "single"


def _normalize_key_prefix(state: dict, target_prefix: str) -> dict:
    """
    把适配器 state dict 的 key 前缀统一为目标前缀，并补齐 peft 的 ".default" 适配器名。

    target_prefix 取值:
        "base_model.model."        —— 目标模型是裸的 PreTrainedModel
        "base_model.model.base_model.model."  —— 目标模型已套一层 PeftModel

    关键: checkpoint 里的 key 形如
        base_model.model.base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    而 PeftModel.state_dict() 里的 key 形如
        base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
    两者差一个 ".default" 适配器名后缀。若不对齐，load_state_dict(strict=False)
    会静默跳过全部权重 -> 模型等于没用 LoRA，输出退化成基座行为。
    """
    out = {}
    for k, v in state.items():
        new_k = k
        if k.startswith("base_model.model.base_model.model."):
            new_k = k[len("base_model.model.base_model.model."):]
        elif k.startswith("base_model.model."):
            new_k = k[len("base_model.model."):]

        # 补齐 peft 适配器名后缀: lora_A.weight -> lora_A.default.weight
        if ".lora_" in new_k and not new_k.endswith(".default.weight"):
            new_k = new_k.replace(".lora_A.weight", ".lora_A.default.weight")
            new_k = new_k.replace(".lora_B.weight", ".lora_B.default.weight")

        out[target_prefix + new_k] = v
    return out


def _infer_target_modules(state: dict) -> List[str]:
    """
    从 checkpoint 的 key 反推真实被 LoRA 覆盖的模块名。

    注意: GBM 适配器的 adapter_config.json 里 target_modules 只写了
    [o_proj, v_proj, k_proj, q_proj]，但 checkpoint 里实际包含 mlp 的
    down_proj / gate_proj / up_proj。只信 config 会导致这些权重无处安放。
    """
    modules = set()
    for k in state.keys():
        if ".lora_A" not in k:
            continue
        parts = k.split(".lora_A")[0].split(".")
        if parts:
            modules.add(parts[-1])
    return sorted(modules)


def load_merged_llamole_gbm(
    base_model_path: str,
    llamole_adapter: str,
    gbm_adapter: str,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float16,
    merge_llamole: bool = True,
):
    """
    加载 基座 + Llamole 适配器 + GBM 适配器。

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=dtype, trust_remote_code=True
    )
    # 注意: base 模型保持在 CPU 上加载，待所有适配器都装好后再一次性搬到 GPU。
    # 若先上 GPU(15GB)，随后加载 LoRA 还要额外显存，GPU 被其他进程占用时会 OOM。

    # ---- 步骤 1: Llamole 适配器 ----
    if merge_llamole and os.path.isdir(llamole_adapter):
        print(f"  [1/3] 加载并合并 Llamole 适配器: {llamole_adapter}")
        model = PeftModel.from_pretrained(model, llamole_adapter)
        model = model.merge_and_unload()
        print("        ✓ Llamole 适配器已合并进基座")
    else:
        print("  [1/3] 跳过 Llamole 适配器")

    # ---- 步骤 2: GBM 适配器（规范化 key 前缀后加载）----
    gbm_state = _read_adapter_state(gbm_adapter)
    style = _detect_prefix(gbm_state)
    print(f"  [2/3] GBM 适配器 key 风格: {style}")

    # 目标前缀: 裸模型 -> "base_model.model."
    normalized = _normalize_key_prefix(gbm_state, "base_model.model.")
    print(f"        规范化后 {len(normalized)} 个 key")

    # ---- 步骤 3: 处理 vocab resize（embed_tokens / lm_head）----
    model_state_keys = set(model.state_dict().keys())
    embed_key = "base_model.model.model.embed_tokens.weight"
    lm_head_key = "base_model.model.lm_head.weight"
    has_resize = embed_key in normalized or lm_head_key in normalized

    if has_resize:
        cur_vocab = model.get_input_embeddings().weight.shape[0]
        new_emb = normalized.get(embed_key)
        new_head = normalized.get(lm_head_key)
        target_vocab = new_emb.shape[0] if new_emb is not None else cur_vocab

        print(f"        检测到 vocab resize: 基座={cur_vocab}, 适配器={target_vocab}")
        if target_vocab > cur_vocab:
            # 新增 special token（<design_start> 等）
            added = ["<design_start>", "<design_end>", "<design_body>",
                     "<molecule>", "<retro_start>", "<retro_end>", "<retro_body>",
                     "<rollback_start>", "<rollback_end>"]
            added = [t for t in added if t not in tokenizer.get_vocab()]
            if added:
                num = tokenizer.add_special_tokens(
                    dict(additional_special_tokens=added),
                    replace_additional_special_tokens=False,
                )
                print(f"        tokenizer 新增 {num} 个 special token")
            model.resize_token_embeddings(target_vocab)
            print(f"        ✓ 模型 embedding 扩展到 {target_vocab}")

        # 用适配器里的 embedding 覆盖（保留已训练的新 token 表示）
        model_state = model.state_dict()
        for k in (embed_key, lm_head_key):
            if k in normalized and k in model_state:
                src = normalized[k]
                dst = model_state[k]
                if src.shape == dst.shape:
                    model_state[k] = src.to(dst.dtype)
                    print(f"        ✓ 恢复 {k.split('.')[-1]} 权重 (shape {tuple(src.shape)})")
        model.load_state_dict(model_state, strict=False)

    # ---- 步骤 4: 用 PeftModel 加载规范化的 LoRA 权重 ----
    print("  [3/3] 加载 GBM LoRA 权重")
    from peft import LoraConfig, get_peft_model, PeftModel as _PM

    cfg_path = os.path.join(gbm_adapter, "adapter_config.json")
    import json

    with open(cfg_path) as f:
        acfg = json.load(f)

    # target_modules 以 checkpoint 实际内容为准（config 可能不全）
    inferred = _infer_target_modules(gbm_state)
    config_tm = acfg.get("target_modules") or []
    target_modules = sorted(set(list(config_tm) + inferred))
    if set(inferred) - set(config_tm):
        print(f"        ! adapter_config 缺少 {sorted(set(inferred) - set(config_tm))}，"
              f"已按 checkpoint 实际内容补全")

    lora_cfg = LoraConfig(
        r=acfg.get("r", 8),
        lora_alpha=acfg.get("lora_alpha", 16),
        lora_dropout=acfg.get("lora_dropout", 0.0),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 构造 PeftModel 但不加载权重（避免前缀再被包一层）
    model = get_peft_model(model, lora_cfg)
    # 此时权重 key 已是 "base_model.model....lora_A.default.weight"，与 normalized 一致
    missing, unexpected = model.load_state_dict(normalized, strict=False)

    # ---- 严格校验：确认 LoRA 权重真的进去了 ----
    missing_lora = [k for k in missing if ".lora_" in k]
    n_lora_expected = len([k for k in normalized if ".lora_" in k])
    n_lora_loaded = n_lora_expected - len(missing_lora)
    if missing_lora:
        raise RuntimeError(
            f"LoRA 权重未完全加载: 期望 {n_lora_expected} 个，实际缺失 {len(missing_lora)} 个。"
            f"首个缺失 key: {missing_lora[0]}"
        )
    print(f"        ✓ 加载 {n_lora_loaded}/{n_lora_expected} 个 LoRA 权重（0 缺失）")
    print(f"        ✓ target_modules: {target_modules}")

    model = model.to(device)
    model.eval()
    return model, tokenizer
