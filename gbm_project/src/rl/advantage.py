"""
GRPO 组内 Advantage 计算
========================

GRPO (Group Relative Policy Optimization) 的核心:
    对同一个 prompt 采样 G 个分子, 用**组内**的 reward 标准化得到 advantage:

        A_i = (R_i - mean_j(R_j)) / (std_j(R_j) + eps)

为什么用组内而不是 batch 内:
    - 组内所有分子共享同一个 prompt, 条件完全一致 -> 比较公平
    - 不需要 critic / value model (省掉一半显存和复杂度)
    - 对 reward 的绝对尺度不敏感 (不需要像 DAP 那样重新标定 sigma)

设计要点:

1. **只对有效 SMILES 算 advantage**:
   rollout 失败的样本 (RDKit 解析不了 / 重原子数不足) 不给梯度。
   理由: 垃圾输出的 reward=0 是"无效"而非"低质量", 混在一起会污染信号。

2. **但保留 pruned 分子**:
   pruned (违反 BBB 硬约束等) 的分子是**有效 SMILES**, 只是违反了约束,
   必须给它负 advantage, 否则模型学不到"违反 BBB 要付代价"。

3. **整组跳过的情况**:
   - 组内无有效分子
   - 组内 reward 方差 < eps (全部一样, 没有梯度信号)
   这两种情况标 skipped=True, 训练时整组不贡献梯度。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class GroupAdvantage:
    """一个 group (= 一个 prompt) 的 advantage 计算结果。"""

    prompt_id: int
    rewards: List[float]
    advantages: List[float]
    n_valid: int
    n_total: int
    skipped: bool
    skip_reason: str = ""

    @property
    def reward_mean(self) -> float:
        return float(np.mean(self.rewards)) if self.rewards else 0.0

    @property
    def reward_max(self) -> float:
        return float(np.max(self.rewards)) if self.rewards else 0.0

    @property
    def reward_std(self) -> float:
        return float(np.std(self.rewards)) if self.rewards else 0.0


def compute_group_advantages(
    rewards_by_group: Dict[int, List[float]],
    eps: float = 1e-6,
    min_std: float = 1e-4,
) -> Dict[int, GroupAdvantage]:
    """
    对每个 group 做组内 z-score。

    Args:
        rewards_by_group: {prompt_id: [r1, r2, ...]} —— 顺序必须与
            rollout 采样顺序一致 (无效样本也要占位, 用 NaN 表示)。
        eps: 防止除零
        min_std: reward 标准差低于此值认为"组内无差异", 整组跳过

    Returns:
        {prompt_id: GroupAdvantage}
    """
    out: Dict[int, GroupAdvantage] = {}

    for pid, raw_rewards in rewards_by_group.items():
        n_total = len(raw_rewards)
        valid_mask = [r is not None and not math.isnan(r) for r in raw_rewards]
        valid_rewards = [r for r, ok in zip(raw_rewards, valid_mask) if ok]
        n_valid = len(valid_rewards)

        if n_valid == 0:
            out[pid] = GroupAdvantage(
                prompt_id=pid, rewards=[], advantages=[],
                n_valid=0, n_total=n_total, skipped=True,
                skip_reason="no_valid_samples")
            continue

        mean = float(np.mean(valid_rewards))
        std = float(np.std(valid_rewards))

        if std < min_std:
            # 组内 reward 几乎一样 —— 没有相对好坏, 无梯度信号
            out[pid] = GroupAdvantage(
                prompt_id=pid, rewards=list(valid_rewards),
                advantages=[0.0] * n_valid, n_valid=n_valid,
                n_total=n_total, skipped=True,
                skip_reason="zero_variance")
            continue

        advs = [(r - mean) / (std + eps) for r in valid_rewards]
        out[pid] = GroupAdvantage(
            prompt_id=pid, rewards=list(valid_rewards), advantages=advs,
            n_valid=n_valid, n_total=n_total, skipped=False)

    return out


def flatten_advantages(
    groups: Dict[int, GroupAdvantage],
) -> List[tuple]:
    """
    把 GroupAdvantage 展平成 (prompt_id, reward, advantage) 的列表,
    只保留未跳过的 group。

    展平后的顺序必须与 trainer 里 samples 的顺序一致 —— trainer 负责
    按同样的规则重建这个列表 (见 trainer.py 的 _build_training_batch)。
    """
    flat = []
    for pid in sorted(groups.keys()):
        g = groups[pid]
        if g.skipped:
            continue
        for r, a in zip(g.rewards, g.advantages):
            flat.append((pid, r, a))
    return flat


def summarize_groups(groups: Dict[int, GroupAdvantage]) -> dict:
    """把各 group 的统计汇总成一行指标, 用于训练日志。"""
    all_rewards = []
    for g in groups.values():
        all_rewards.extend(g.rewards)
    n_skipped = sum(1 for g in groups.values() if g.skipped)
    if not all_rewards:
        return {
            "reward_mean": 0.0, "reward_max": 0.0, "reward_std": 0.0,
            "n_groups": len(groups), "n_skipped_groups": n_skipped,
            "n_samples": 0, "adv_abs_mean": 0.0,
        }
    advs = [abs(a) for g in groups.values() if not g.skipped
            for a in g.advantages]
    return {
        "reward_mean": float(np.mean(all_rewards)),
        "reward_max": float(np.max(all_rewards)),
        "reward_std": float(np.std(all_rewards)),
        "n_groups": len(groups),
        "n_skipped_groups": n_skipped,
        "n_samples": len(all_rewards),
        "adv_abs_mean": float(np.mean(advs)) if advs else 0.0,
    }
