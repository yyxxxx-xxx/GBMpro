"""
RL 损失函数: GRPO 与 DAP-Lite
=============================

两种算法都实现, 用于消融对照 (§8.2 A2: "LLM 场景下哪种 RL 算法更好")。

---------------------------------------------------------------------------
1. GRPO (主算法)
---------------------------------------------------------------------------

    L = -(1/N) * Σ_i  A_i * (1/|y_i|) * Σ_t log p_θ(y_{i,t} | x, y_{i,<t})
        + β * KL(π_θ || π_prior)

其中 A_i 是组内 z-score 后的 advantage (§advantage.py)。

KL 用 k3 估计量 (无偏, Δ→0 时方差小, 是 TRL / OpenRLHF 的默认选择):
    KL ≈ exp(Δ) - Δ - 1,   Δ = log p_θ - log p_prior

**为什么用 GRPO 而不是 REINVENT 的 DAP**:
    - DAP 的 target 是 `prior_lls + σ·R`, 没有 baseline, σ 对 reward 尺度
      极度敏感 (REINVENT 的 σ=128 是对字符级 SMILES NLL 标定的, LLM 侧
      完全不适用)。GRPO 的组内 z-score 让算法与 reward 的绝对尺度无关。
    - DAP 没有显式 KL 惩罚, 漂移难控。GRPO 有独立的 β 项。
    - 我们的 reward 是**确定性函数**, 不需要 critic, GRPO 天然适用。

---------------------------------------------------------------------------
2. DAP-Lite (对照组)
---------------------------------------------------------------------------

REINVENT 原始 DAP (reinvent/runmodes/RL/reward.py:51-52):
    augmented_lls = prior_lls + sigma * scores
    loss = (augmented_lls - agent_lls) ** 2

移植到 LLM:
    L = (prior_lls + σ·R' - agent_lls)²

**两处必须的修正** (否则在 LLM 上跑不出结果):
    (a) σ 必须重标定: 原始 σ=128 对字符级 SMILES NLL 标定。LLM 侧 BPE
        token 的 logprob 尺度、序列长度分布都不同。用**自适应 σ**:
            σ_eff = σ0 * std(prior_lls) / std(R')
        使 σ·R 的方差与 log-likelihood 的方差同阶。
    (b) R' 必须做 z-score: 原始 DAP 没有 baseline, 组内 reward 一律同向
        偏移时模型无法区分"更好"和"好"。用 batch 内 z-score 修正。

**参考实现** (对照 DAP 的实际语义):
    - REINVENT `sigma = 128` (configs/reinvent_tlrl_idh1.toml)
    - REINVENT `prior_lls`/`agent_lls` 来自 `runmodes/RL/learning.py`
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


# ---------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------

def grpo_loss(
    agent_logps: torch.Tensor,
    prior_logps: torch.Tensor,
    mask: torch.Tensor,
    advantages: torch.Tensor,
    beta: float = 0.02,
    kl_estimator: str = "k3",
    length_normalize: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    GRPO 损失。

    Args:
        agent_logps: [B, T] agent 对每个 response token 的 logprob
        prior_logps: [B, T] 参考模型 (SFT) 对同样 token 的 logprob
        mask:        [B, T] response token 的 mask (1 = 计入, 0 = padding)
        advantages:  [B]    每条样本的 advantage (组内 z-score)
        beta:        KL 惩罚系数
        kl_estimator: 'k3' | 'k1' | 'k2'
        length_normalize: 是否按有效 token 数归一 (True 推荐, 否则长 SMILES
            会被系统性惩罚/奖励)

    Returns:
        {"loss", "policy_loss", "kl", "n_valid", "entropy_proxy"}
    """
    assert agent_logps.shape == prior_logps.shape == mask.shape
    assert agent_logps.shape[0] == advantages.shape[0]

    m = mask.float()
    n_valid = m.sum(dim=1)                       # [B] 每条样本的有效 token 数
    valid_rows = n_valid > 0
    if valid_rows.sum() == 0:
        zero = agent_logps.sum() * 0.0
        return {"loss": zero, "policy_loss": zero, "kl": zero,
                "n_valid": torch.tensor(0.0, device=agent_logps.device),
                "entropy_proxy": zero}

    # --- 序列级 logprob ---
    seq_agent = (agent_logps * m).sum(dim=1)     # [B]
    seq_prior = (prior_logps * m).sum(dim=1)     # [B]

    # --- Policy loss ---
    if length_normalize:
        seq_agent_norm = seq_agent / n_valid.clamp(min=1.0)
    else:
        seq_agent_norm = seq_agent
    policy = -(advantages.detach() * seq_agent_norm)
    policy_loss = policy[valid_rows].mean()

    # --- KL penalty ---
    delta = (agent_logps - prior_logps) * m      # [B, T]
    if kl_estimator == "k3":
        # k3 = exp(Δ) - Δ - 1, 逐 token 求和
        kl_tok = torch.exp(delta.clamp(max=10.0)) - delta - 1.0
        kl = (kl_tok * m).sum(dim=1) / n_valid.clamp(min=1.0)
    elif kl_estimator == "k2":
        kl_tok = 0.5 * delta.pow(2)
        kl = (kl_tok * m).sum(dim=1) / n_valid.clamp(min=1.0)
    else:  # k1
        kl = ((agent_logps - prior_logps) * m).sum(dim=1) / n_valid.clamp(min=1.0)
    kl_mean = kl[valid_rows].mean()

    loss = policy_loss + beta * kl_mean

    return {
        "loss": loss,
        "policy_loss": policy_loss.detach(),
        "kl": kl_mean.detach(),
        "n_valid": valid_rows.sum().float(),
        "entropy_proxy": (-(agent_logps * m).sum(dim=1)
                          / n_valid.clamp(min=1.0)).mean().detach(),
    }


# ---------------------------------------------------------------------------
# DAP-Lite
# ---------------------------------------------------------------------------

def adaptive_sigma(
    prior_lls: torch.Tensor,
    rewards: torch.Tensor,
    sigma0: float = 8.0,
    min_sigma: float = 0.5,
    max_sigma: float = 200.0,
) -> torch.Tensor:
    """
    自适应 sigma: 使 sigma*R 的方差与 prior_lls 的方差同阶。

        sigma_eff = sigma0 * std(prior_lls) / std(R)

    原始 REINVENT 的 sigma=128 是对字符级 SMILES NLL 标定的。
    LLM 侧 BPE token 的 logprob 尺度完全不同, 必须重新标定,
    否则 augmented_lls 变成不可达量级 -> loss 退化成常数 -> 梯度消失。

    Args:
        prior_lls: [B] 参考模型的序列 logprob
        rewards:   [B] reward (已 z-score)
    Returns:
        标量 tensor
    """
    if prior_lls.numel() < 2:
        return torch.tensor(sigma0, device=prior_lls.device, dtype=prior_lls.dtype)
    s_ll = prior_lls.std()
    s_r = rewards.std()
    if s_r < 1e-8:
        return torch.tensor(sigma0, device=prior_lls.device, dtype=prior_lls.dtype)
    sigma = sigma0 * s_ll / s_r
    return sigma.clamp(min=min_sigma, max=max_sigma)


def dap_loss(
    agent_logps: torch.Tensor,
    prior_logps: torch.Tensor,
    mask: torch.Tensor,
    rewards: torch.Tensor,
    sigma0: float = 8.0,
    reward_zscore: bool = True,
    use_adaptive_sigma: bool = True,
    length_normalize: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    DAP-Lite 损失 (REINVENT DAP 移植到 LLM, 带两处必要修正)。

    Args:
        agent_logps: [B, T]
        prior_logps: [B, T]
        mask:        [B, T]
        rewards:     [B] 原始 reward (或已归一化的 advantage)
        sigma0:      σ 基准值 (自适应模式下作为缩放因子)
        reward_zscore: 是否对 reward 做 z-score (修正原始 DAP 缺 baseline)
        use_adaptive_sigma: 是否自适应标定 σ (修正原始 σ=128 的量纲问题)
        length_normalize: 是否按有效 token 数归一 (与 GRPO 保持一致;
            否则长 SMILES 的序列 logprob 绝对值更大, 会被系统性惩罚)
    """
    assert agent_logps.shape == prior_logps.shape == mask.shape

    m = mask.float()
    n_valid = m.sum(dim=1)
    valid_rows = n_valid > 0
    if valid_rows.sum() == 0:
        zero = agent_logps.sum() * 0.0
        return {"loss": zero, "policy_loss": zero, "kl": zero,
                "n_valid": torch.tensor(0.0, device=agent_logps.device),
                "sigma": torch.tensor(0.0, device=agent_logps.device)}

    seq_agent = (agent_logps * m).sum(dim=1)
    seq_prior = (prior_logps * m).sum(dim=1)

    # ⚠️ 长度归一: 与 GRPO 保持一致。
    #    DAP 的 `augmented = seq_prior + sigma * r` 是**序列级**目标,
    #    若不归一, 长 SMILES 的 |seq_logprob| 天然更大 -> 平方误差被
    #    长分子主导 -> 模型会系统性偏好短分子。
    if length_normalize:
        n_norm = n_valid.clamp(min=1.0)
        seq_agent = seq_agent / n_norm
        seq_prior = seq_prior / n_norm

    # --- reward z-score (修正 1: 缺 baseline) ---
    if reward_zscore:
        rv = rewards[valid_rows]
        r = (rewards - rv.mean()) / (rv.std() + 1e-6)
        r = torch.where(valid_rows, r, torch.zeros_like(r))
    else:
        r = rewards

    # --- 自适应 sigma (修正 2: 量纲) ---
    if use_adaptive_sigma:
        sigma = adaptive_sigma(seq_prior[valid_rows].detach(),
                               r[valid_rows].detach(), sigma0=sigma0)
    else:
        sigma = torch.tensor(sigma0, device=agent_logps.device,
                             dtype=agent_logps.dtype)

    augmented = seq_prior + sigma * r
    sq = (augmented - seq_agent).pow(2)
    policy_loss = sq[valid_rows].mean()

    loss = policy_loss

    return {
        "loss": loss,
        "policy_loss": policy_loss.detach(),
        "kl": torch.tensor(0.0, device=agent_logps.device),
        "n_valid": valid_rows.sum().float(),
        "sigma": sigma.detach(),
    }


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

def compute_loss(
    algorithm: str,
    agent_logps: torch.Tensor,
    prior_logps: torch.Tensor,
    mask: torch.Tensor,
    advantages: torch.Tensor,
    beta: float = 0.02,
    sigma0: float = 8.0,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """按算法名分发。

    ⚠️⚠️ **实测踩坑**: trainer 的调用点会**无条件**传
    `kl_estimator` / `length_normalize` (它们是 GRPO 专有参数)。
    早期版本直接 `dap_loss(..., **kwargs)` → 抛
    `TypeError: dap_loss() got an unexpected keyword argument 'kl_estimator'`,
    而 DAP 路径**从未被实跑验证过**, 单元测试也没覆盖到 → 这个 bug
    潜伏了整个 M0–M4 阶段 (2026-11-24 扩大实验前审计才发现)。

    修法: 按算法**显式筛选**参数, 不做盲目透传。
    """
    if algorithm == "grpo":
        return grpo_loss(agent_logps, prior_logps, mask, advantages,
                         beta=beta, **kwargs)
    if algorithm == "dap":
        # DAP 只认这几个参数, 其余 (kl_estimator / beta) 对 DAP 无意义。
        # length_normalize 语义上仍适用 (与 GRPO 保持一致)。
        dap_kwargs = {k: v for k, v in kwargs.items()
                      if k in ("length_normalize", "reward_zscore",
                               "use_adaptive_sigma")}
        # DAP 无显式 KL 项 -> beta 不参与
        return dap_loss(agent_logps, prior_logps, mask, advantages,
                        sigma0=sigma0, **dap_kwargs)
    raise ValueError(f"未知算法: {algorithm} (支持 'grpo' / 'dap')")
