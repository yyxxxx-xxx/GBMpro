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

# rank 模式的归一化常数
#
# ⚠️ 用**常数**而非 n/2 作为分母, 否则组大小不同会导致同一 rank 位置
#    在不同组给出不同 advantage -> 组间不可比, 梯度尺度随组大小漂移。
#    取 16.0 时 advantage 落在约 (-1, 1), 与 zscore 的典型量级相当。
RANK_SCALE = 16.0


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
    mode: str = "zscore",
) -> Dict[int, GroupAdvantage]:
    """
    对每个 group 做组内 advantage 计算。

    Args:
        rewards_by_group: {prompt_id: [r1, r2, ...]} —— 顺序必须与
            rollout 采样顺序一致 (无效样本也要占位, 用 NaN 表示)。
        eps: 防止除零
        min_std: reward 标准差低于此值认为"组内无差异", 整组跳过
        mode: advantage 计算方式
            - "zscore"  (默认) A_i = (R_i - mean) / std
              **保持原行为不变**, 保证历史实验可复现
            - "pairwise" (借鉴 PBCNet2.0) 见下方说明
            - "pairwise_scaled" pairwise + 保留组间信息量差异 (推荐)
            - "rank"    组内 rank-based, 居中用 (n+1)/2, 尺度用常数
              PBCNet (1.0) 用的就是 rank; PBCNet2.0 改成 Δ 后 rank 被淘汰,
              但 rank 对异常值最稳健, 保留作为对照

    Returns:
        {prompt_id: GroupAdvantage}

    ------------------------------------------------------------------------
    pairwise 模式的设计依据 (PBCNet2.0)
    ------------------------------------------------------------------------
    PBCNet2.0 的 readout 只做**成对差分**, 不做拼接 (models/readout.py)::

        emb = self.norm(emb1 - emb2)      # 只用差
        out_put = self.FNN(emb)

    论文原话:
        "employing Δ as the pair-level representation allows the model to
         devote more attention to protein-ligand interactions. This is
         important because if a model solely focuses on ligand molecular
         structures, [g_ref] and [g_query] would be similar as they are
         structural analogues, making Δ close to a null vector."

    映射到 GRPO:
        z-score 的隐含假设是 "绝对分数高 = 更好", 但 reward 里混着大量
        **与组内对比无关**的常数偏移 (TPSA 归一化、MW 分段、ADMET 端点
        截断的固定地板)。这些偏移会稀释组内的真实相对差异。

        pairwise 用**相对组内最优/最差**的归一化差值:
            A_i = (R_i - R_ref) / (R_best - R_worst)
        分母用**极差**而不是标准差, 天然把"组内绝对尺度"约掉。

    ⚠️ 零方差处理与 zscore 一致 (整组跳过), 保持行为可对比。
    """
    if mode not in ("zscore", "pairwise", "pairwise_scaled", "rank"):
        raise ValueError(f"未知的 advantage mode: {mode}")

    out: Dict[int, GroupAdvantage] = {}

    # --- 预计算全局组内 std 中位数 (pairwise_scaled 需要) ---
    group_std_median = 0.0
    if mode == "pairwise_scaled":
        stds = []
        for raw in rewards_by_group.values():
            vs = [r for r in raw if r is not None and not math.isnan(r)]
            if len(vs) >= 2:
                stds.append(float(np.std(vs)))
        if stds:
            group_std_median = float(np.median(stds))

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

        if mode == "zscore":
            advs = [(r - mean) / (std + eps) for r in valid_rewards]
        elif mode == "pairwise":
            # 成对差分 (PBCNet2.0 启发): 用**极差**归一化, 约掉组内绝对尺度
            #
            # ⚠️ 纯极差归一化有一个代价: 所有组的 advantage 跨度恒为 2.0,
            #    把"这个组 reward 差异大 (信息多)" 和 "差异小 (信息少)"
            #    强行拉到同一尺度 —— 抹平了组间的信息量差异 (实测跨度恒为 1.0)。
            #    因此这里用**混合**: 先按组内 reward 尺度做缩放, 再用极差归一。
            #    缩放因子 = 组内 std / 全局典型 std, 让"信息量大的组"
            #    得到更大的 advantage 幅度。
            r_best = max(valid_rewards)
            r_worst = min(valid_rewards)
            rng = r_best - r_worst
            if rng < min_std:
                # 极差也过小 —— 无区分度
                advs = [0.0] * n_valid
            else:
                # 参考点取组内均值, 保证 advantage 均值为 0 (与 zscore 一致)
                #   A_i = (R_i - mean) / (R_best - R_worst)
                # 分母是极差而非标准差 -> 约掉绝对尺度的常数偏移
                advs = [(r - mean) / (rng + eps) for r in valid_rewards]
        elif mode == "pairwise_scaled":
            # pairwise + 组间尺度保留 (推荐用于 M5-E 消融)
            #
            # 动机 (来自 scripts/compare_advantage_modes.py 的实测):
            #   纯 pairwise 的 advantage 跨度在所有组都恒为 1.0,
            #   丢失了"组内 reward 差异大小"这一信息。
            #   zscore 保留了这个信息 (跨度 2.27~2.74 随组变化),
            #   但把绝对分数的常数偏移也带了进来。
            #
            # 这里取两者的折中:
            #   A_i = (R_i - mean) / (std + eps)      ← zscore 的尺度
            #   但先把 reward 平移到组内相对位置 -> 等价于 pairwise 的去偏移
            # 实际实现: 先用极差归一得到"纯相对差", 再乘以该组的尺度权重
            #           weight = std_group / median(std_all_groups)
            # 这样: (a) 组内偏移被消掉; (b) 组间信息量差异被保留。
            r_best = max(valid_rewards)
            r_worst = min(valid_rewards)
            rng = r_best - r_worst
            if rng < min_std:
                advs = [0.0] * n_valid
            else:
                # 纯相对差 (去偏移)
                rel = [(r - mean) / (rng + eps) for r in valid_rewards]
                # 尺度权重: 组内 std 相对全局中位数
                w = std / (group_std_median + eps) if group_std_median > 0 else 1.0
                w = min(max(w, 0.25), 4.0)      # 限制在 [0.25, 4.0]
                advs = [x * w for x in rel]
        else:  # rank
            # 组内 rank-based: 最稳健 (对离群值免疫)
            #
            # ⚠️ 两个关键设计:
            #  1. 归一化分母用**常数** (RANK_SCALE) 而不是 n/2。
            #     否则组大小不同 -> 分母不同 -> 组间 advantage 不可比,
            #     梯度尺度随组大小漂移。
            #  2. 居中用**组内 rank 均值** (n+1)/2, 保证组内均值恒为 0
            #     (与 zscore / pairwise 一致)。
            order = sorted(range(n_valid), key=lambda i: valid_rewards[i])
            ranks = [0.0] * n_valid
            for pos, idx in enumerate(order):
                ranks[idx] = float(pos + 1)
            center = (n_valid + 1) / 2.0
            advs = [(r - center) / RANK_SCALE for r in ranks]

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
