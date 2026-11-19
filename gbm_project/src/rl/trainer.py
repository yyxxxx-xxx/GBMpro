"""
RL 训练器 (GRPO / DAP-Lite)
============================

三 adapter 架构 (实测跑通, 见 docs/RL_INTEGRATION_DESIGN.md §5.7):

    base (Qwen2-7B-Instruct)
      └─ + Llamole LoRA (merged)          → 通用能力
          └─ + GBM SFT LoRA (frozen)       → prior (RL 的 KL 锚点)
              └─ + RL LoRA (trainable)     → agent (被优化的策略)

⚠️ 三个 adapter 的 requires_grad 必须**显式逐参数设置**:
    `add_adapter` 之后 SFT LoRA 的参数可能仍是 `requires_grad=True`,
    会被一起优化 -> **静默污染 prior**, RL 结果不可解释 (风险 R14)。
    训练后必须校验 SFT adapter 权重哈希未变。

实测性能 (单卡 3090):
    Rollout B=32      11.0 s   0.343 s/分子   峰值 15.6 GB
    fwd+bwd B=8        2.17 s                  峰值 19.5 GB  ← 训练 batch 上限
    fwd+bwd B=16       OOM
    单步 (32 分子)    ~52 s (Reward 28 s 占 54%, 是真正的瓶颈)

⚠️ 生成阶段必须关 gradient checkpointing (会强制 use_cache=False,
   KV cache 失效 -> 生成卡死)。生成/训练两个阶段切换。
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
GBM_ROOT = PROJECT_ROOT / "gbm_project"

if str(GBM_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(GBM_ROOT / "src"))
if str(GBM_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(GBM_ROOT / "scripts"))


@dataclass
class TrainerConfig:
    """训练配置。"""

    # --- 模型 ---
    base_model: str = "/root/Llamole-main/models/Qwen2-7B-Instruct"
    llamole_adapter: str = "/root/Llamole-main/saves/Llamole-Qwen2-7B-Instruct-Adapter"
    sft_adapter: str = "/root/Llamole-main/saves/Llamole-Qwen2-7B-Instruct-Adapter-gbm-active"
    target: str = "IDH1"

    # --- RL 超参 ---
    algorithm: str = "grpo"          # 'grpo' | 'dap'
    lr: float = 1e-6                 # 比 SFT (1e-4) 低 2 个数量级
    beta_kl: float = 0.02
    sigma0: float = 8.0              # DAP 自适应 sigma 的基准
    max_grad_norm: float = 1.0
    train_batch: int = 8             # ⚠️ 实测 16 会 OOM
    grad_accum: int = 4              # 有效 batch = 8 × 4 = 32
    kl_estimator: str = "k3"
    length_normalize: bool = True

    # --- Rollout (实测有效率, 见 rl/rollout.py) ---
    n_prompts: int = 4
    group_size: int = 8
    temperature: float = 0.8
    top_p: float = 0.92
    max_new_tokens: int = 512
    repetition_penalty: float = 1.1
    min_heavy_atoms: int = 12
    # 每 N 步存一次 `latest` checkpoint (state 含 optimizer, 约 190MB)
    save_every: int = 5

    # --- 多样性 ---
    diversity_bucket_size: int = 10  # 与 REINVENT 一致
    diversity_minscore: float = 0.4  # 与 REINVENT 一致
    enable_diversity_filter: bool = True

    # --- RL LoRA ---
    rl_lora_r: int = 16
    rl_lora_alpha: int = 32
    rl_lora_dropout: float = 0.05
    rl_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # --- 其它 ---
    seed: int = 0
    n_workers: int = 15
    admet_device: str = "cpu"


# ---------------------------------------------------------------------------
# 模型构建
# ---------------------------------------------------------------------------

def build_rl_model(cfg: TrainerConfig, device: str = "cuda:0"):
    """
    构建三 adapter 模型。

    Returns:
        (model, tokenizer)
        model 上的 adapter 名字: 'default'=SFT, 'rl'=可训练
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, LoraConfig

    print(f"[model] 加载基座 {cfg.base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=torch.float16, device_map=device)
    model.config.use_cache = False

    # 1) Llamole LoRA -> merge 进基座
    if cfg.llamole_adapter and Path(cfg.llamole_adapter).exists():
        print(f"[model] merge Llamole adapter: {cfg.llamole_adapter}", flush=True)
        model = PeftModel.from_pretrained(model, cfg.llamole_adapter)
        model = model.merge_and_unload()

    # 2) GBM SFT LoRA -> **也 merge 进基座**
    #
    # ⚠️⚠️ 实测踩坑 (关键): PEFT 的 `set_adapter(name)` **一次只能激活一个
    #    adapter**。如果保留 SFT adapter 为独立层再 `set_adapter("rl")`,
    #    SFT 就被停用了 -> rollout 跑的是「基座 + 随机初始化的 RL LoRA」,
    #    等于在噪声上采样 -> 有效 SMILES 率为 0。
    #
    #    正确做法: 把 SFT LoRA 也 merge 进基座, 只留一个可训练的 RL LoRA。
    #    此时:
    #       agent = merged_base(SFT) + RL LoRA      <- rollout & 训练用
    #       prior = merged_base(SFT)                 <- disable_adapter() 得到
    #    prior 是 merge 后的**同一个模型**, 所以 prior 不可能被污染 (R14 消除)。
    print(f"[model] merge SFT adapter (frozen prior): {cfg.sft_adapter}", flush=True)
    model = PeftModel.from_pretrained(model, cfg.sft_adapter, is_trainable=True)
    model = model.merge_and_unload()

    # 3) RL LoRA (trainable) —— agent
    #    初始为全零 (lora_B=0) -> agent 初始 == prior, 训练从"不变"开始
    #
    # ⚠️ 用 `get_peft_model` 而不是 `from_pretrained`:
    #    from_pretrained 会**加载 SFT adapter 的权重** (非零) -> RL 一开始
    #    就偏离 prior; 我们要的是"从 prior 出发做增量调整"。
    #    get_peft_model 创建全新的零初始化 LoRA。
    rl_cfg = LoraConfig(
        r=cfg.rl_lora_r, lora_alpha=cfg.rl_lora_alpha,
        lora_dropout=cfg.rl_lora_dropout, bias="none",
        target_modules=cfg.rl_target_modules, task_type="CAUSAL_LM")
    from peft import get_peft_model
    model = get_peft_model(model, rl_cfg, adapter_name="rl")
    model.set_adapter("rl")

    # ⚠️ 关键: 显式逐参数设置 requires_grad
    n_train = 0
    for n, p in model.named_parameters():
        p.requires_grad = ("lora" in n) and (".rl." in n)
        if p.requires_grad:
            n_train += p.numel()
    print(f"[model] RL 可训练参数: {n_train/1e6:.2f}M", flush=True)

    model.to(device)
    # ⚠️ 必须: 否则 gradient checkpointing 下 backward 报
    #    "element 0 of tensors does not require grad"
    model.enable_input_require_grads()
    return model, tokenizer


def sft_adapter_fingerprint(model, n_tensors: int = 64) -> str:
    """
    计算 **基座权重**的指纹 —— 用于校验 prior 未被污染。

    ⚠️ 架构变更后 (SFT LoRA 已 merge 进基座), 没有独立的 "sft" adapter 了。
    现在 prior = 基座权重 (merge 后), agent = 基座 + RL LoRA。

    ⚠️⚠️ 实测踩坑: 最初实现把**全部**基座参数转 float32 求哈希 ——
    7B 模型 = 28GB 数据, 每步要算几十秒, RL 训练直接卡死。
    改成只哈希**确定性采样**的 n_tensors 个张量 (按参数名排序后取
    均匀间隔的样本), 既能检测异常改动, 又只花几毫秒。

    由于 optimizer 只持有 RL LoRA 的参数, 基座权重在数学上不可能被改变;
    这个指纹只是**防御性**检查 (防止意外的 in-place 操作 / 精度漂移)。
    """
    import hashlib
    named = [(n, p) for n, p in model.named_parameters() if "lora" not in n]
    named.sort(key=lambda x: x[0])
    if not named:
        return "empty:0"
    # 均匀采样 n_tensors 个 (含首尾)
    if len(named) <= n_tensors:
        picked = named
    else:
        step = max(1, len(named) // n_tensors)
        picked = named[::step][:n_tensors]
    h = hashlib.sha256()
    for name, p in picked:
        h.update(name.encode())
        # ⚠️ 不要 .float() —— 7B 参数转 fp32 会产生几十 GB 的临时拷贝,
        #    实测耗时 24 秒。直接用原始 dtype 的字节即可。
        h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return f"{h.hexdigest()[:16]}:{len(picked)}/{len(named)}"


# ---------------------------------------------------------------------------
# logprob 计算 (最容易出错的地方)
# ---------------------------------------------------------------------------

def compute_logprobs(
    model,
    tokenizer,
    prompt_ids: List[int],
    response_ids: List[int],
    batch_size: int = 8,
    device: str = "cuda:0",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    一次性算出 agent 和 prior 的 per-token logprob。

    实现路径 (关键: 用 PEFT 的 adapter 切换, 而非加载两份模型):
        1. model.set_adapter("rl")      -> agent = SFT + RL LoRA
           forward (no_grad) -> agent_logps
        2. with model.disable_adapter():  -> prior = SFT only
           forward (no_grad) -> prior_logps

    Args:
        prompt_ids:  [P] prompt token ids (所有样本共享同一 prompt)
        response_ids: [R] response token ids

    Returns:
        (logps, mask) 两个 [B, T] tensor
    """
    device = torch.device(device)
    P = len(prompt_ids)
    R = len(response_ids)
    T = P + R

    input_ids = torch.tensor([list(prompt_ids) + list(response_ids)],
                             dtype=torch.long, device=device)
    attn = torch.ones_like(input_ids)

    # padding 到 batch 内最长
    all_logps = []
    all_mask = []
    B = input_ids.shape[0]

    # 逐条算 (batch>1 需要 padding, 这里先按样本数循环, 后面再 batch 化)
    for i in range(B):
        ids = input_ids[i:i+1]
        am = attn[i:i+1]

        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=am)
            logits = out.logits.float()          # [1, T, V]

            # logprob: 位置 t 的 logits 预测位置 t+1 的 token
            logp = torch.log_softmax(logits[:, :-1, :], dim=-1)   # [1, T-1, V]
            tgt = ids[:, 1:]                                         # [1, T-1]
            tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [1, T-1]

        # 只取 response 部分 (位置 P-1 .. T-2 对应 response token 0..R-1)
        resp_lp = tok_lp[:, P-1:]                # [1, R]
        mask = torch.ones_like(resp_lp)
        all_logps.append(resp_lp[0])
        all_mask.append(mask[0])

    return torch.stack(all_logps), torch.stack(all_mask)


def batched_logprobs(
    model,
    tokenizer,
    sequences: List[Tuple[List[int], List[int]]],
    batch_size: int = 8,
    device: str = "cuda:0",
    with_grad: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    批量算 logprob (带 padding)。

    ⚠️⚠️ **关键优化: 只对 response 位置算 logits**。
    直接调 `model(...)` 会对**全部**位置算 logits, 而后 Qwen2ForCausalLM
    还会做 `logits.float()` —— 4 × 650 token × 152k 词表 × 4 bytes
    = 1.58 GB 一份, 再加 log_softmax 和 backward 的梯度, **直接 OOM**
    (实测: `torch.OutOfMemoryError: Tried to allocate 484.00 MiB`)。

    而 prompt 占了序列的 85% (实测 prompt_len=379~592, response 只有 ~77),
    这些位置的 logprob 根本用不上 (mask=0)。

    做法: 只跑 transformer body 拿 hidden states, 然后把 lm_head
    **只作用在 response 对应的位置**上。
        B=4, R=77, V=152k  ->  4 × 77 × 152k × 4 bytes = 187 MB (可接受)

    Args:
        sequences: [(prompt_ids, response_ids), ...]
        with_grad: True=agent (要梯度); False=prior (no_grad)

    Returns:
        (logps [B, T_max], mask [B, T_max])
    """
    device = torch.device(device)
    B = len(sequences)
    T_max = max(len(p) + len(r) for p, r in sequences)

    input_ids = torch.full((B, T_max), tokenizer.pad_token_id,
                            dtype=torch.long, device=device)
    attn = torch.zeros((B, T_max), dtype=torch.long, device=device)
    resp_mask = torch.zeros((B, T_max), dtype=torch.bool, device=device)
    spans = []          # 每个样本的 (response_start, response_end)

    for i, (p, r) in enumerate(sequences):
        L = len(p) + len(r)
        input_ids[i, :L] = torch.tensor(list(p) + list(r), device=device)
        attn[i, :L] = 1
        resp_mask[i, len(p):L] = True
        spans.append((len(p), L))

    # --- 拿到 transformer body 与 lm_head (兼容 PEFT 包装) ---
    body, lm_head = _get_body_and_head(model)

    # ⚠️ 分块: 一次只处理 batch_size 个样本的 log_softmax。
    #    log_softmax 的中间结果 [chunk, R, V] 在反向时必须保留,
    #    R=77, V=152k -> 单样本 47MB, 6 个样本就是 280MB + 梯度副本。
    #    分块 + 保留计算图 -> 显存可控。
    chunk = max(1, batch_size)
    full_lp = torch.zeros((B, T_max), dtype=torch.float32, device=device)
    mask = torch.zeros((B, T_max), dtype=torch.bool, device=device)

    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    for start in range(0, B, chunk):
        end = min(start + chunk, B)
        b = end - start
        with ctx:
            out = body(input_ids=input_ids[start:end], attention_mask=attn[start:end])
            hidden = out.last_hidden_state          # [b, T_max, H]

            for j in range(b):
                i = start + j
                p_start, p_end = spans[i]
                # 位置 t 的 hidden 预测 token t+1
                # response token 在 [p_start, p_end), 对应 hidden 在
                # [p_start-1, p_end-1)
                h = hidden[j, p_start - 1:p_end - 1]          # [R, H]
                logits = lm_head(h).float()                    # [R, V]
                logp = torch.log_softmax(logits, dim=-1)       # [R, V]
                tgt = input_ids[i, p_start:p_end]              # [R]
                tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # [R]
                full_lp[i, p_start:p_end] = tok_lp
                mask[i, p_start:p_end] = True

    return full_lp, mask


def _get_body_and_head(model):
    """
    拆出 transformer body 与 lm_head (兼容 PEFT 各种包装层级)。

    ⚠️ 不能用 `hasattr(m, "lm_head")` 判断层级 —— PEFT 的 LoraModel 实现了
    `__getattr__` 转发, `hasattr(lora_model, "lm_head")` 会返回 True,
    导致取到的 body 其实是 ForCausalLM 而不是 Qwen2Model。

    层级: PeftModel -> LoraModel -> Qwen2ForCausalLM -> {model, lm_head}

    ⚠️ body 必须是 **LoRA 注入后的** decoder, 否则 RL LoRA 不生效
    (LoRA 层是挂在 Qwen2Model 的子模块上的)。
    """
    m = model
    # 剥 PeftModel
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model          # LoraModel.model -> ForCausalLM
    # 剥 LoraModel (如果上面没成功)
    if not hasattr(m, "lm_head") and hasattr(m, "model"):
        m = m.model
    if not hasattr(m, "lm_head"):
        raise RuntimeError(f"无法从 {type(model)} 找到 lm_head "
                           f"(当前层级: {type(m)})")
    body = m.model          # Qwen2Model
    head = m.lm_head
    if not hasattr(body, "layers"):
        raise RuntimeError(f"body 不是 transformer decoder: {type(body)}")
    return body, head


def _to_jsonable(obj):
    """
    把含 tensor 的嵌套结构转成 JSON 可序列化的形式。

    ⚠️ **已不再用于 optimizer 状态** —— 那样做会让 checkpoint 膨胀到 2.2GB
    (AdamW 的 exp_avg/exp_avg_sq 转成 Python list 体积暴涨 5-7 倍)。
    optimizer 状态现在用 `torch.save` 存二进制。见 save_checkpoint()。

    这里保留给小的元数据 (如 prompt_bank / history) 使用。
    """
    import numpy as np
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# 训练器
# ---------------------------------------------------------------------------

class RLTrainer:
    """
    GRPO / DAP-Lite 训练循环。

    单步流程:
        1. 从冻结的 prompt bank 抽 n_prompts 条
        2. Rollout: 每条采 group_size 个分子
        3. Reward: 并行 Vina + admet-ai 打分
        4. Advantage: 组内 z-score (GRPO) / batch z-score (DAP)
        5. Loss: GRPO 或 DAP-Lite
        6. 反向传播 + step
    """

    def __init__(self, cfg: TrainerConfig, out_dir: Path):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)

        from rl.rollout import RolloutConfig, RolloutEngine
        from rl.prompt_bank import PromptBank
        from rl.reward_service import RewardConfig, RewardService
        from rl.diversity import ScaffoldDiversityFilter

        # --- 模型 ---
        self.model, self.tokenizer = build_rl_model(cfg)
        self.rollout = RolloutEngine(self.model, self.tokenizer, RolloutConfig(
            group_size=cfg.group_size, temperature=cfg.temperature,
            top_p=cfg.top_p, max_new_tokens=cfg.max_new_tokens,
            min_heavy_atoms=cfg.min_heavy_atoms,
            repetition_penalty=cfg.repetition_penalty, seed=cfg.seed))

        # --- prompt bank (冻结) ---
        self.prompt_bank = PromptBank.build(cfg.target, n_prompts=cfg.n_prompts)
        print(f"[prompt] {len(self.prompt_bank.prompts)} 条, "
              f"fingerprint={self.prompt_bank.fingerprint}", flush=True)

        # --- reward service ---
        self.reward_service = RewardService(RewardConfig(
            target=cfg.target, n_workers=cfg.n_workers,
            admet_device=cfg.admet_device,
            cache_path=self.out_dir / "reward_cache.jsonl"))

        # --- 多样性 ---
        self.diversity = ScaffoldDiversityFilter(
            bucket_size=cfg.diversity_bucket_size,
            minscore=cfg.diversity_minscore)

        # --- 优化器 ---
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=cfg.lr,
                                           weight_decay=0.0, betas=(0.9, 0.999))
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: max(0.1, 1.0 - step / 100.0))

        self.step = 0
        self.history: List[dict] = []
        t_fp = time.time()
        self._sft_fp = sft_adapter_fingerprint(self.model)
        print(f"[init] 基座指纹 {self._sft_fp} ({time.time()-t_fp:.2f}s)", flush=True)

    # ------------------------------------------------------------------
    def save_checkpoint(self, tag: str = "latest"):
        """只存 RL LoRA (≈161MB), 不存完整模型 (磁盘只剩 148GB)。

        ⚠️⚠️ optimizer 状态**不能**用 JSON 存!
        实测: 40.37M 参数的 AdamW 状态 (exp_avg + exp_avg_sq)
        转成 Python list 后 JSON 体积暴涨到 **2.2 GB** (比二进制大 5-7 倍),
        60 步训练 + 每 5 步存一次 = 几十 GB, 直接撑爆磁盘。
        -> optimizer 状态用 `torch.save` 存二进制 .pt 文件
        -> JSON 只存元数据 (config / history / diversity)
        """
        from peft.utils import get_peft_model_state_dict
        sd = get_peft_model_state_dict(self.model, adapter_name="rl")
        path = self.out_dir / f"rl_lora_{tag}.safetensors"
        from safetensors.torch import save_file
        save_file({k: v.detach().cpu().contiguous() for k, v in sd.items()},
                  str(path))

        # optimizer 状态 -> 二进制 (161MB, 而不是 JSON 的 2.2GB)
        opt_path = self.out_dir / f"optimizer_{tag}.pt"
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.sched.state_dict(),
            "step": self.step,
        }, str(opt_path))

        # 元数据 -> JSON (小)
        state = {
            "step": self.step,
            "config": asdict(self.cfg),
            "prompt_bank": self.prompt_bank.to_dict(),
            "base_fingerprint": self._sft_fp,
            "diversity": self.diversity.state_dict(),
            "history": self.history,
        }
        with open(self.out_dir / f"state_{tag}.json", "w") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        print(f"[ckpt] {path.name} (+{opt_path.name})", flush=True)

    def load_checkpoint(self, tag: str = "latest"):
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict
        lora_path = self.out_dir / f"rl_lora_{tag}.safetensors"
        state_path = self.out_dir / f"state_{tag}.json"
        opt_path = self.out_dir / f"optimizer_{tag}.pt"
        if not state_path.exists():
            print(f"[ckpt] 找不到 {state_path}, 从头开始", flush=True)
            return False
        state = json.load(open(state_path))
        if state.get("base_fingerprint") != self._sft_fp:
            print("[ckpt] ⚠️ 基座 fingerprint 不一致, 先验可能已变!", flush=True)
        if lora_path.exists():
            sd = load_file(str(lora_path))
            set_peft_model_state_dict(self.model, sd, adapter_name="rl")
        self.step = state.get("step", 0)
        self.diversity.load_state_dict(state.get("diversity", {}))
        self.history = state.get("history", [])
        # optimizer / scheduler 从二进制恢复
        if opt_path.exists():
            try:
                blob = torch.load(str(opt_path), map_location="cpu",
                                  weights_only=False)
                self.optimizer.load_state_dict(blob["optimizer"])
                self.sched.load_state_dict(blob["scheduler"])
                print(f"[ckpt] optimizer 已恢复", flush=True)
            except Exception as e:
                print(f"[ckpt] ⚠️ optimizer 恢复失败 ({e}), 用空状态继续", flush=True)
        print(f"[ckpt] resume @ step {self.step}", flush=True)
        return True

    # ------------------------------------------------------------------
    def run_step(self) -> dict:
        """执行一个完整的 RL step。"""
        from rl.advantage import compute_group_advantages, summarize_groups
        from rl.losses import compute_loss

        cfg = self.cfg
        t_start = time.time()

        # --- 1. Rollout ---
        rng = random.Random(cfg.seed * 1000 + self.step)
        prompt_idxs = rng.sample(range(len(self.prompt_bank.prompts)),
                                  min(cfg.n_prompts, len(self.prompt_bank.prompts)))
        prompts = []
        for pi in prompt_idxs:
            from rl.prompt_bank import format_full_prompt
            prompts.append(format_full_prompt(self.prompt_bank.prompts[pi]))

        t0 = time.time()
        samples = self.rollout.generate(prompts, n_groups=len(prompts),
                                         group_size=cfg.group_size,
                                         seed=cfg.seed * 100 + self.step)
        t_rollout = time.time() - t0

        # ⚠️ 关键: RolloutEngine 返回的 prompt_id 是**组的位置** (0,1,2...),
        #    而 prompt_idxs 是**原始索引**。必须建立映射, 否则算 logprob 时
        #    用的 prompt 与 rollout 时的 prompt 不是同一个 -> logprob 全错。
        #    (实测踩坑: 组 1 用了 bank.prompts[1] 而实际生成用的是 bank.prompts[2])
        group_to_prompt = {gid: prompt_idxs[gid] for gid in range(len(prompt_idxs))}

        # --- 2. Reward ---
        valid = [s for s in samples if s.valid]
        t0 = time.time()
        print(f"[step {self.step}] rollout 完成: {len(valid)}/{len(samples)} 有效, "
              f"耗时 {t_rollout:.1f}s -> 开始 reward", flush=True)
        if valid:
            recs = self.reward_service.score_batch([s.smiles for s in valid])
        else:
            recs = []
        t_reward = time.time() - t0
        print(f"[step {self.step}] reward 完成: {len(recs)} 条, "
              f"耗时 {t_reward:.1f}s", flush=True)

        # 把 reward 映射回 samples
        rec_by_smiles = {}
        for r in recs:
            rec_by_smiles[r.smiles] = r

        # --- 3. 组内 reward ---
        rewards_by_group: Dict[int, List[float]] = {}
        for s in samples:
            rewards_by_group.setdefault(s.prompt_id, []).append(
                rec_by_smiles[s.smiles].reward if s.valid and s.smiles in rec_by_smiles
                else float("nan"))

        groups = compute_group_advantages(rewards_by_group)
        gstats = summarize_groups(groups)

        # --- 4. 构建训练 batch ---
        (agent_seqs, prior_seqs, advs) = self._build_training_batch(
            samples, rec_by_smiles, groups, group_to_prompt)

        if not agent_seqs:
            print(f"[step {self.step}] ⚠️ 没有可用样本, 跳过", flush=True)
            self.step += 1
            return {"step": self.step, "skipped": True, **gstats}

        # --- 5. Logprob + Loss ---
        # ⚠️⚠️ gradient checkpointing 必须在**前向之前**开启!
        #    之前放在 backward 阶段才开 -> 前向的完整激活图被全部保留在显存里,
        #    6 个样本直接 OOM (实测 OutOfMemoryError: Tried to allocate 94 MiB)。
        #    开 checkpointing 后前向只保存边界激活, 显存大幅下降。
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.config.use_cache = False

        t0 = time.time()
        agent_lp, agent_mask = batched_logprobs(
            self.model, self.tokenizer, agent_seqs,
            batch_size=cfg.train_batch, with_grad=True)

        # ⚠️ prior = 关掉 RL LoRA 后的模型 (= merge 后的 SFT 模型)
        #    绝不能直接用 agent 的 logprob 冒充 prior, 否则 KL 恒为 0,
        #    策略可以无约束漂移 (RL 最常见的静默失效)。
        with torch.no_grad(), self.model.disable_adapter():
            prior_lp, prior_mask = batched_logprobs(
                self.model, self.tokenizer, prior_seqs,
                batch_size=cfg.train_batch, with_grad=False)

        # mask 对齐 (agent 与 prior 的序列顺序一致, mask 应相同)
        if not torch.equal(agent_mask, prior_mask):
            prior_mask = agent_mask

        advs_t = torch.tensor(advs, dtype=torch.float32,
                               device=agent_lp.device)

        loss_out = compute_loss(
            cfg.algorithm, agent_lp, prior_lp, agent_mask, advs_t,
            beta=cfg.beta_kl, sigma0=cfg.sigma0,
            kl_estimator=cfg.kl_estimator,
            length_normalize=cfg.length_normalize,
        )
        t_logprob = time.time() - t0

        # --- 6. Backward ---
        t0 = time.time()
        self.model.train()

        total_loss = loss_out["loss"]
        (total_loss / cfg.grad_accum).backward()

        grad_norm = 0.0
        if (self.step + 1) % cfg.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                cfg.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.sched.step()
        t_bwd = time.time() - t0

        # --- 7. 多样性更新 ---
        if cfg.enable_diversity_filter and valid:
            self.diversity.update(
                [s.smiles for s in valid],
                [rec_by_smiles[s.smiles].reward for s in valid])

        # --- 8. 污染检查 ---
        fp_now = sft_adapter_fingerprint(self.model)
        contaminated = (fp_now != self._sft_fp)

        stats = {
            "step": self.step,
            "loss": float(loss_out["loss"].item()),
            "policy_loss": float(loss_out["policy_loss"].item()),
            "kl": float(loss_out["kl"].item()),
            "adv_abs_mean": gstats["adv_abs_mean"],
            "reward_mean": gstats["reward_mean"],
            "reward_max": gstats["reward_max"],
            "reward_std": gstats["reward_std"],
            "n_samples": gstats["n_samples"],
            "n_valid_smiles": len(valid),
            "n_total_samples": len(samples),
            "n_skipped_groups": gstats["n_skipped_groups"],
            "n_train_seqs": len(agent_seqs),
            "t_rollout": round(t_rollout, 1),
            "t_reward": round(t_reward, 1),
            "t_logprob": round(t_logprob, 1),
            "t_backward": round(t_bwd, 1),
            "t_total": round(time.time() - t_start, 1),
            "lr": self.sched.get_last_lr()[0],
            "n_buckets": self.diversity.n_buckets,
            "prior_contaminated": contaminated,
        }
        if "sigma" in loss_out:
            stats["sigma"] = float(loss_out["sigma"].item())
        if contaminated:
            print(f"[step {self.step}] 🔴 SFT 先验被污染! {self._sft_fp} -> {fp_now}",
                  flush=True)

        self.history.append(stats)
        self.step += 1
        return stats

    # ------------------------------------------------------------------
    def _build_training_batch(self, samples, rec_by_smiles, groups,
                              group_to_prompt):
        """
        构建 (agent_seqs, prior_seqs, advantages)。

        ⚠️ 多样性过滤: bucket 满的骨架的分子不参与训练 (与 REINVENT 一致)。
        ⚠️ 无效 SMILES (解析失败) 不参与训练 —— 它们的 reward=0 是"无效"
           而不是"低质量", 混进去会污染信号。
        ⚠️ pruned 分子**参与**训练 —— 它是有效 SMILES 但违反了硬约束,
           必须给负 advantage, 让模型学会"违反 BBB 要付代价"。
        """
        from rl.rollout import group_samples

        agent_seqs, prior_seqs, advs = [], [], []
        groups_map = group_samples(samples)

        for pid in sorted(groups_map.keys()):
            g = groups.get(pid)
            if g is None or g.skipped:
                continue

            grp = groups_map[pid]
            # advantage 与 valid reward 一一对应 (compute_group_advantages
            # 只保留了 valid 的), 需要重新对齐
            adv_iter = iter(g.advantages)
            for s in grp:
                if not s.valid or s.smiles not in rec_by_smiles:
                    continue
                rec = rec_by_smiles[s.smiles]
                if rec.guard_violation:
                    continue  # 结构守卫拦下的, 不训练
                if (self.cfg.enable_diversity_filter
                        and self.diversity.is_full(s.smiles)):
                    continue  # 骨架 bucket 满了
                try:
                    adv = next(adv_iter)
                except StopIteration:
                    break

                prompt_text = self._prompt_for(group_to_prompt[s.prompt_id])
                p_ids = self.tokenizer(prompt_text,
                                       return_tensors="pt")["input_ids"][0].tolist()
                # ⚠️ response 格式必须是 "SMILES: xxx" —— 与 SFT 训练分布一致。
                #    训练时 response 就是这个形态 (见 build_qa_dataset.py:16)。
                #    若只用裸 SMILES, 模型会学到"直接输出 SMILES",
                #    与 SFT 分布不一致, 且下一轮 rollout 的 prompt 也会漂移。
                resp_text = f"SMILES: {rec.smiles}{self.tokenizer.eos_token}"
                r_ids = self.tokenizer(resp_text,
                                       return_tensors="pt",
                                       add_special_tokens=False)["input_ids"][0].tolist()
                if len(r_ids) == 0:
                    continue
                agent_seqs.append((p_ids, r_ids))
                prior_seqs.append((p_ids, r_ids))
                advs.append(adv)

        return agent_seqs, prior_seqs, advs

    def _prompt_for(self, prompt_id: int) -> str:
        from rl.prompt_bank import format_full_prompt
        return format_full_prompt(self.prompt_bank.prompts[prompt_id])

    # ------------------------------------------------------------------
    def train(self, n_steps: int = 60, log_every: int = 1):
        """主循环。"""
        print(f"\n{'='*70}\n"
              f"[RL] 开始训练: algorithm={self.cfg.algorithm} "
              f"target={self.cfg.target} steps={n_steps}\n"
              f"{'='*70}", flush=True)

        for _ in range(n_steps):
            stats = self.run_step()
            if stats.get("skipped"):
                continue
            if (self.step - 1) % log_every == 0:
                print(
                    f"[step {stats['step']:3d}] "
                    f"loss={stats['loss']:.4f} "
                    f"kl={stats['kl']:.5f} "
                    f"rew={stats['reward_mean']:.4f} "
                    f"(max {stats['reward_max']:.4f}) "
                    f"valid={stats['n_valid_smiles']}/{stats['n_total_samples']} "
                    f"adv|={stats['adv_abs_mean']:.3f} "
                    f"t={stats['t_total']}s "
                    f"(ro {stats['t_rollout']} + rw {stats['t_reward']} "
                    f"+ lp {stats['t_logprob']} + bw {stats['t_backward']})",
                    flush=True)

            self._write_log()

            # ⚠️ 定期存 `latest` —— 否则中途被杀只能从头开始, 白跑。
            #    (state 文件含 optimizer 状态, 约 190MB, 每 5 步存一次)
            if self.cfg.save_every > 0 and self.step % self.cfg.save_every == 0:
                self.save_checkpoint("latest")

        self.save_checkpoint("final")
        print(f"\n[RL] 训练结束, 共 {self.step} 步", flush=True)
        return self.history

    def _write_log(self):
        path = self.out_dir / "train_log.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.history[-1], ensure_ascii=False) + "\n")
