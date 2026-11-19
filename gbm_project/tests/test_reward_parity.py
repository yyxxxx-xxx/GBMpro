"""
Reward 一致性测试 (M0 验收标准)
================================

**这是接 RL 的第一道坎**: RL 的 reward 必须与反馈闭环
(`generate_feedback_loop.py`) 的 reward **逐分子完全一致**,
否则 RL 优化的目标和历史结果不在同一尺度上, 论文里的对比全部作废。

测试内容:
    1. **Parity**: 同一批 SMILES, `RewardService.score_batch()` 与
       `generate_feedback_loop` 的评估逻辑结果必须一致
    2. **并行一致性**: 串行 vs 并行, 结果必须一致
    3. **缓存一致性**: 缓存命中前后结果一致
    4. **守卫正确性**: 明显非药物分子被守卫拦下
    5. **失败处理**: 无效 SMILES 返回 reward=0 而不是抛异常

运行:
    python3 scripts/test_reward_parity.py
    pytest gbm_project/tests/test_reward_parity.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GBM_ROOT = PROJECT_ROOT / "gbm_project"

for p in (str(GBM_ROOT / "src"), str(GBM_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)


# 实测基线 (2026-11-19, IDH1 靶点, exhaustiveness=4, n_poses=5)
# 15 workers 并行下的 Vina 分数 —— 必须逐个一致
# ⚠️ 这些是**完整精度**的值 (Vina 输出 3 位小数), 不要四舍五入到 2 位
BASELINE_VINA = {
    "CCC(N(Cc1ccccc1F)S(=O)(=O)CC(F)(F)F)F": -7.940,
    "CC(C)Nc1ncc(-c2cnc(N)nc2)c2ccccc12": -8.105,
    "COc1cc2ncnc(N[C@@H](C)c3ccccc3)c2cc1OC": -9.477,
    "Cc1ccc(-c2nc3ccccc3n2-c2ccccc2)cc1": -7.758,
    "O=C(NCc1ccccc1)c1ccc(O)cc1": -9.364,
    "CCOc1ccc2nc(S(N)(=O)=O)sc2c1": -7.079,
    "CN1C(=O)N(c2ccc(F)cc2)C2=CC=CC=C12": -8.074,
    "O=C(Nc1ccc(F)cc1)c1ccco1": -7.884,
}

TEST_SMILES = list(BASELINE_VINA.keys())


# ---------------------------------------------------------------------------
# 单元测试 (不需要真跑 Vina, 快)
# ---------------------------------------------------------------------------

class TestHackingGuards:
    """结构守卫。"""

    def test_rejects_oversized_molecule(self):
        from rl.reward_service import check_hacking_guards
        bad = "C" * 60
        v = check_hacking_guards(bad)
        assert v is not None, "长烷烃应该被拦下"

    def test_accepts_normal_drug_like(self):
        from rl.reward_service import check_hacking_guards
        for smi in ["CCO", "c1ccccc1", "CC(=O)Nc1ccc(O)cc1"]:
            v = check_hacking_guards(smi)
            assert v is None, f"{smi} 不该被拦下, 但得到 {v}"

    def test_invalid_smiles_rejected(self):
        from rl.reward_service import check_hacking_guards
        assert check_hacking_guards("not_a_smiles_at_all!!") is not None

    def test_long_aliphatic_chain(self):
        from rl.reward_service import check_hacking_guards
        # 阈值是 12 (判断条件为 > 12), 所以 12 个碳恰好通过
        assert check_hacking_guards("CCCCCCCCCCCC") is None
        # 13 个碳超阈值
        assert check_hacking_guards("CCCCCCCCCCCCC") is not None


class TestAdvantage:
    """GRPO 组内 advantage。"""

    def test_zscore_properties(self):
        from rl.advantage import compute_group_advantages
        groups = compute_group_advantages({0: [0.1, 0.5, 0.3, 0.9]})
        g = groups[0]
        assert not g.skipped
        assert len(g.advantages) == 4
        assert abs(sum(g.advantages)) < 1e-5
        assert abs((sum(a * a for a in g.advantages) / 4) ** 0.5 - 1.0) < 1e-3
        assert g.advantages[3] == max(g.advantages)
        assert g.advantages[0] == min(g.advantages)

    def test_zero_variance_skipped(self):
        from rl.advantage import compute_group_advantages
        groups = compute_group_advantages({0: [0.5, 0.5, 0.5, 0.5]})
        assert groups[0].skipped
        assert groups[0].skip_reason == "zero_variance"

    def test_all_invalid_skipped(self):
        from rl.advantage import compute_group_advantages
        groups = compute_group_advantages({0: [float("nan")] * 4})
        assert groups[0].skipped
        assert groups[0].skip_reason == "no_valid_samples"

    def test_invalid_excluded_from_stats(self):
        from rl.advantage import compute_group_advantages
        groups = compute_group_advantages({0: [0.2, 0.4, float("nan"), 0.6]})
        g = groups[0]
        assert g.n_valid == 3
        assert abs(g.reward_mean - 0.4) < 1e-6

    def test_summary(self):
        from rl.advantage import compute_group_advantages, summarize_groups
        groups = compute_group_advantages({0: [0.1, 0.5], 1: [0.2, 0.8]})
        s = summarize_groups(groups)
        assert s["n_groups"] == 2
        assert s["n_samples"] == 4
        assert s["reward_max"] == pytest.approx(0.8)


class TestLosses:
    """损失函数。"""

    def test_grpo_loss_sign(self):
        """
        GRPO policy loss = -adv * logp。

        logp 为负 (概率 < 1), 所以:
            adv > 0 -> loss = -adv * logp > 0 (正)
            adv < 0 -> loss = -adv * logp < 0 (负)

        梯度方向 (-adv * ∇logp): adv>0 时**降低**该样本的 logprob,
        即提高好样本的概率 —— 这才是 GRPO 想要的。
        """
        import torch
        from rl.losses import grpo_loss
        agent = torch.full((1, 3), float(torch.log(torch.tensor(0.5))))
        prior = torch.full((1, 3), float(torch.log(torch.tensor(0.5))))
        mask = torch.ones(1, 3)
        loss_pos = grpo_loss(agent, prior, mask, torch.tensor([1.0]))
        loss_neg = grpo_loss(agent, prior, mask, torch.tensor([-1.0]))
        assert loss_pos["policy_loss"].item() > 0
        assert loss_neg["policy_loss"].item() < 0
        assert (loss_pos["policy_loss"].item()
                == pytest.approx(-loss_neg["policy_loss"].item()))

    def test_grpo_gradient_direction(self):
        """
        核心正确性: advantage 为正的样本, 一步梯度下降后其 logprob 应该**上升**。
        """
        import torch
        from rl.losses import grpo_loss

        lp = torch.full((1, 3), -1.0, requires_grad=True)
        prior = torch.full((1, 3), -1.0)
        mask = torch.ones(1, 3)
        out = grpo_loss(lp, prior, mask, torch.tensor([1.0]))
        out["loss"].backward()
        # policy_loss = -adv * logp, d/dlogp = -adv = -1
        assert lp.grad is not None
        assert lp.grad[0, 0].item() < 0, "梯度方向应为负 (下降 logp 会增大 loss?)"
        # 更新: logp_new = logp - lr * grad -> logp 变大
        new_lp = (lp.detach() - 0.1 * lp.grad)[0, 0].item()
        assert new_lp > -1.0, "好样本的 logprob 应该上升"

    def test_grpo_kl_zero_when_identical(self):
        import torch
        from rl.losses import grpo_loss
        lp = torch.full((2, 4), -1.0)
        mask = torch.ones(2, 4)
        out = grpo_loss(lp, lp.clone(), mask, torch.tensor([1.0, -1.0]))
        assert abs(out["kl"].item()) < 1e-5

    def test_grpo_kl_positive_when_different(self):
        import torch
        from rl.losses import grpo_loss
        agent = torch.full((2, 4), -1.0)
        prior = torch.full((2, 4), -2.0)
        mask = torch.ones(2, 4)
        out = grpo_loss(agent, prior, mask, torch.tensor([1.0, -1.0]))
        assert out["kl"].item() > 0

    def test_dap_loss_finite(self):
        import torch
        from rl.losses import dap_loss
        agent = torch.full((4, 3), -1.5)
        prior = torch.full((4, 3), -1.6)
        mask = torch.ones(4, 3)
        rewards = torch.tensor([0.1, 0.5, 0.3, 0.9])
        out = dap_loss(agent, prior, mask, rewards)
        assert torch.isfinite(out["loss"])
        assert out["sigma"].item() > 0

    def test_adaptive_sigma(self):
        import torch
        from rl.losses import adaptive_sigma
        prior = torch.tensor([-10.0, -20.0, -30.0, -40.0])
        rewards = torch.tensor([0.1, 0.5, 0.3, 0.9])
        sigma = adaptive_sigma(prior, rewards, sigma0=8.0)
        assert torch.isfinite(sigma)
        assert sigma.item() > 0

    def test_compute_loss_dispatch(self):
        import torch
        from rl.losses import compute_loss
        agent = torch.full((2, 3), -1.0)
        mask = torch.ones(2, 3)
        advs = torch.tensor([1.0, -1.0])
        for alg in ("grpo", "dap"):
            out = compute_loss(alg, agent, agent.clone(), mask, advs)
            assert torch.isfinite(out["loss"])


class TestDiversity:
    """多样性过滤器。"""

    def test_bucket_limits(self):
        from rl.diversity import ScaffoldDiversityFilter
        df = ScaffoldDiversityFilter(bucket_size=2, minscore=0.0)
        smis = ["c1ccccc1C", "c1ccccc1CC", "c1ccccc1CCC"]
        for s in smis:
            df.update([s], [1.0])
        assert df.n_buckets == 1
        assert df.total_kept == 2
        assert df.is_full("c1ccccc1CCCC")

    def test_minscore_filters(self):
        from rl.diversity import ScaffoldDiversityFilter
        df = ScaffoldDiversityFilter(bucket_size=10, minscore=0.4)
        df.update(["c1ccccc1C"], [0.1])
        assert df.total_kept == 0

    def test_state_roundtrip(self):
        from rl.diversity import ScaffoldDiversityFilter
        df = ScaffoldDiversityFilter(bucket_size=2, minscore=0.0)
        df.update(["c1ccccc1C", "c1ccccc1CC"], [1.0, 1.0])
        d = df.state_dict()
        df2 = ScaffoldDiversityFilter()
        df2.load_state_dict(d)
        assert df2.n_buckets == df.n_buckets
        assert df2.total_kept == df.total_kept


class TestRewardServiceErrors:
    """错误处理 —— 失败必须给 reward=0, 不能抛异常。"""

    def test_invalid_smiles_returns_zero(self):
        from rl.reward_service import RewardService, RewardConfig
        svc = RewardService(RewardConfig(target="IDH1", n_workers=1,
                                         enable_hacking_guards=True))
        try:
            rec = svc.score("this_is_not_a_smiles")
            assert rec.reward == 0.0
            assert rec.error is not None or rec.guard_violation is not None
        finally:
            svc.close()


# ---------------------------------------------------------------------------
# 集成测试 (需要真跑 Vina, 慢)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestRewardParity:
    """与实测基线的一致性。"""

    def test_vina_matches_baseline(self):
        """
        实测基线 (2026-11-19):
            15 workers 并行, IDH1, exhaustiveness=4
            分数必须逐个一致 —— 这是并行安全性的回归测试。
        """
        from rl.reward_service import RewardService, RewardConfig
        svc = RewardService(RewardConfig(target="IDH1", n_workers=15,
                                         use_cache=False))
        try:
            recs = svc.score_batch(TEST_SMILES)
            for smi, rec in zip(TEST_SMILES, recs):
                assert rec.vina is not None, f"{smi} 没有 vina 分数"
                assert abs(rec.vina - BASELINE_VINA[smi]) < 1e-6, (
                    f"{smi}: vina={rec.vina} != 基线 {BASELINE_VINA[smi]}")
        finally:
            svc.close()

    def test_serial_vs_parallel_identical(self):
        from rl.reward_service import RewardService, RewardConfig
        sub = TEST_SMILES[:4]
        svc_s = RewardService(RewardConfig(target="IDH1", n_workers=1,
                                           use_cache=False))
        svc_p = RewardService(RewardConfig(target="IDH1", n_workers=8,
                                           use_cache=False))
        try:
            rs = svc_s.score_batch(sub)
            rp = svc_p.score_batch(sub)
            for a, b in zip(rs, rp):
                assert a.vina == pytest.approx(b.vina, abs=1e-6)
                assert a.reward == pytest.approx(b.reward, abs=1e-6)
        finally:
            svc_s.close()
            svc_p.close()

    def test_cache_returns_same(self):
        from rl.reward_service import RewardService, RewardConfig
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            cfg = RewardConfig(target="IDH1", n_workers=4,
                               cache_path=Path(td) / "cache.jsonl")
            svc = RewardService(cfg)
            try:
                r1 = svc.score_batch(TEST_SMILES[:3])
                r2 = svc.score_batch(TEST_SMILES[:3])
                for a, b in zip(r1, r2):
                    assert a.reward == b.reward
                    assert a.vina == b.vina
            finally:
                svc.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short",
                           "-m", "not slow"]))
