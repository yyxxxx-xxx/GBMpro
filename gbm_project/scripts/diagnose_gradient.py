#!/usr/bin/env python3
"""
梯度有效性诊断
================

验证 RL 的三个前提:
  1. backward 后 LoRA 参数的 .grad 非零 (梯度真的传到了可训练层)
  2. optimizer.step() 后参数真的改变了 (优化器真的在工作)
  3. agent 与 prior 的 logprob 开始出现差异 (KL > 0)

⚠️ 为什么要单独测:
   RL 最危险的失败模式是**静默失效** —— 训练跑完、loss 有值、日志好看,
   但梯度根本没进 LoRA (比如 PEFT adapter 没激活 / 冻结了全部参数 /
   optimizer 拿到的 param group 是空的)。
   这些都不会报错, 只是学不到东西。

用法:
    python3 scripts/diagnose_gradient.py
"""

from __future__ import annotations

import sys
from pathlib import Path

GBM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GBM / "src"))
sys.path.insert(0, str(GBM / "scripts"))

import torch


def main():
    from rl.trainer import TrainerConfig, build_rl_model, batched_logprobs

    cfg = TrainerConfig(target="IDH1", lr=3e-6)
    model, tok = build_rl_model(cfg)

    # --- 1. 可训练参数检查 ---
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    print(f"\n{'='*66}")
    print(f"1. 可训练参数: {len(trainable)} 个 tensor, "
          f"{sum(p.numel() for _, p in trainable):,} 元素")
    if not trainable:
        print("   ❌ 没有可训练参数! PEFT adapter 没激活")
        return
    # 确认都是 LoRA
    non_lora = [n for n, _ in trainable if "lora" not in n.lower()]
    print(f"   非 LoRA 的可训练参数: {len(non_lora)} 个 "
          f"{'(异常!)' if non_lora else '(正常, 全是 LoRA)'}")
    for n, _ in trainable[:3]:
        print(f"     e.g. {n[:80]}")

    # --- 2. 构造一个假的训练 batch ---
    prompt = "Instruction: generate a molecule\nInput:\nResponse:"
    p_ids = tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
    r_ids = tok("SMILES: CCOc1ccc2nc(S(N)(=O)=O)sc2c1" + tok.eos_token,
                return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    seqs = [(p_ids, r_ids)] * 3

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()

    lp, mask = batched_logprobs(model, tok, seqs, batch_size=2, with_grad=True)
    print(f"\n2. logprob shape={tuple(lp.shape)} "
          f"finite={bool(torch.isfinite(lp).all())} "
          f"mean={lp.mean().item():.4f}")

    # 造一个假的 advantage (正负各半)
    # ⚠️ lp 是 (B, T) 的 **逐 token** logprob, advantage 是 **每序列** 一个标量,
    #    所以要先按 mask 求每序列的平均 logprob 再加权 (与 grpo_loss 一致)。
    m = mask.float()
    seq_logp = (lp * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
    advs = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float32, device=lp.device)
    loss = -(advs * seq_logp).mean()
    loss.backward()

    # --- 3. 梯度检查 ---
    grads = [(n, p.grad) for n, p in trainable if p.grad is not None]
    nonzero = [(n, g) for n, g in grads if g.abs().sum() > 0]
    print(f"\n3. 梯度检查:")
    print(f"   有 .grad 的参数: {len(grads)}/{len(trainable)}")
    print(f"   梯度非零的参数: {len(nonzero)}/{len(trainable)}")
    if not nonzero:
        print("   ❌ 梯度全零! 梯度没有传到 LoRA")
        return
    for n, g in nonzero[:3]:
        print(f"     {n[:60]}: |g|={g.abs().sum().item():.6f}")

    # --- 4. optimizer.step() 后参数是否变化 ---
    before = {n: p.detach().clone() for n, p in trainable[:3]}
    opt = torch.optim.AdamW([p for _, p in trainable], lr=cfg.lr)
    torch.nn.utils.clip_grad_norm_([p for _, p in trainable], cfg.max_grad_norm)
    opt.step()
    delta = 0.0
    for n, p in trainable[:3]:
        d = (p.detach() - before[n]).abs().max().item()
        delta = max(delta, d)
    print(f"\n4. optimizer.step() 后参数最大变化量: {delta:.3e}")
    print(f"   (lr={cfg.lr}, 期望量级 ~{cfg.lr:.0e})")
    print(f"   {'✅ 参数确实在变' if delta > 0 else '❌ 参数没变!'}")

    # --- 5. agent vs prior 的 logprob 差异 ---
    with torch.no_grad(), model.disable_adapter():
        prior_lp, _ = batched_logprobs(model, tok, seqs, batch_size=2,
                                        with_grad=False)
    diff = (lp.detach() - prior_lp).abs().max().item()
    print(f"\n5. agent vs prior logprob 最大差异: {diff:.3e}")
    print(f"   (LoRA 零初始化时应该 ~0, 即 agent == prior)")
    print(f"   {'✅ agent/prior 分离正确' if diff < 1e-3 else '⚠️ 差异非零, 检查 LoRA 初始化'}")

    print(f"\n{'='*66}")
    ok = len(nonzero) > 0 and delta > 0
    print(f"结论: {'✅ 梯度链路正常, RL 可以训练' if ok else '❌ 梯度链路断裂'}")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()
