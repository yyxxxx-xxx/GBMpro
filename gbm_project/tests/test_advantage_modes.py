"""
Advantage 计算模式的单元测试
=============================

覆盖三种模式: zscore (默认, 必须与历史行为完全一致) / pairwise / rank

关键不变量:
  1. zscore 模式必须与改动前的实现**逐位一致** (可复现性保证)
  2. 三种模式都满足: advantage 均值为 0 (组内)
  3. pairwise: advantage 范围被限制在 [-1, 1] (极差归一化)
  4. pairwise: 对**常数偏移**不敏感 (PBCNet2.0 Δ 的核心动机)
  5. rank: 对离群值免疫
  6. 零方差组在三种模式下都被跳过

运行:
    python3 tests/test_advantage_modes.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

GBM_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GBM_ROOT / "src"))

from rl.advantage import compute_group_advantages, flatten_advantages

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}  {detail}")


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_zscore_backward_compat():
    """zscore 必须与原始实现逐位一致 (保证历史实验可复现)。"""
    print("\n[1] zscore 向后兼容")
    groups = {0: [0.1, 0.3, 0.5, 0.2], 1: [1.0, 2.0, 3.0]}
    res = compute_group_advantages(groups, mode="zscore")

    # 手算原始实现
    for pid, rewards in groups.items():
        mean = sum(rewards) / len(rewards)
        var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        std = math.sqrt(var)
        expected = [(r - mean) / (std + 1e-6) for r in rewards]
        got = res[pid].advantages
        ok = all(close(g, e) for g, e in zip(got, expected))
        check(f"group {pid} 与手算一致", ok, f"got={got} exp={expected}")


def test_zero_mean():
    """三种模式的组内 advantage 均值都应为 0。"""
    print("\n[2] 组内均值恒为 0")
    groups = {0: [0.1, 0.3, 0.5, 0.2], 1: [1.0, 2.0, 3.0, 5.0, 4.0]}
    for mode in ("zscore", "pairwise", "rank"):
        res = compute_group_advantages(groups, mode=mode)
        ok = True
        for pid, g in res.items():
            m = sum(g.advantages) / len(g.advantages)
            if not close(m, 0.0, 1e-6):
                ok = False
        check(f"{mode}: 所有组均值≈0", ok)


def test_pairwise_range():
    """pairwise: advantage 落在 [-1, 1] (极差归一化)。"""
    print("\n[3] pairwise 值域 [-1, 1]")
    groups = {0: [0.0, 0.25, 0.5, 1.0, 0.75], 1: [-3.0, 2.0, 0.0]}
    res = compute_group_advantages(groups, mode="pairwise")
    ok = True
    for pid, g in res.items():
        for a in g.advantages:
            if a < -1.0001 or a > 1.0001:
                ok = False
    check("所有 advantage ∈ [-1, 1]", ok)


def test_pairwise_shift_invariance():
    """pairwise 对常数偏移不敏感 —— PBCNet2.0 Δ 设计的核心动机。

    组内所有 reward 加同一个常数, 组内相对关系不变,
    advantage 应该**完全不变**。
    """
    print("\n[4] pairwise 对常数偏移不敏感")
    base = {0: [0.1, 0.3, 0.5, 0.2]}
    shifted = {0: [0.1 + 5.0, 0.3 + 5.0, 0.5 + 5.0, 0.2 + 5.0]}

    for mode in ("pairwise", "rank"):
        a = compute_group_advantages(base, mode=mode)[0].advantages
        b = compute_group_advantages(shifted, mode=mode)[0].advantages
        ok = all(close(x, y) for x, y in zip(a, b))
        check(f"{mode}: 加常数后 advantage 不变", ok, f"{a} vs {b}")

    # zscore 本来就是平移不变的, 顺带确认
    a = compute_group_advantages(base, mode="zscore")[0].advantages
    b = compute_group_advantages(shifted, mode="zscore")[0].advantages
    check("zscore: 加常数后 advantage 不变 (回归)", all(close(x, y) for x, y in zip(a, b)))


def test_pairwise_scale_invariance():
    """pairwise 对**正比例缩放**也不敏感 (极差归一化的性质)。

    ⚠️ 这是 pairwise 相比 zscore 的关键优势:
       z-score 会把"组内差异大"和"组内差异小"这两种情况给出**相同尺度**
       的 advantage (都是标准正态), 丢失了"这个组值得多学"的信号。
       pairwise 用极差归一化后, 组内差异的大小信息被保留在分母里。
    """
    print("\n[5] pairwise 对正比例缩放不敏感")
    base = {0: [0.1, 0.3, 0.5, 0.2]}
    scaled = {0: [0.1 * 10, 0.3 * 10, 0.5 * 10, 0.2 * 10]}
    a = compute_group_advantages(base, mode="pairwise")[0].advantages
    b = compute_group_advantages(scaled, mode="pairwise")[0].advantages
    # eps=1e-6 会引入 ~1e-7 的浮点尾差, 容差取 1e-5
    check("pairwise: ×10 后 advantage 不变",
          all(close(x, y, 1e-5) for x, y in zip(a, b)), f"{a} vs {b}")

    # zscore 同样平移+缩放不变 (标准化的定义), 但**不保留幅度信息**
    az = compute_group_advantages(base, mode="zscore")[0].advantages
    bz = compute_group_advantages(scaled, mode="zscore")[0].advantages
    check("zscore: ×10 后 advantage 不变 (对照)",
          all(close(x, y, 1e-5) for x, y in zip(az, bz)))


def test_rank_outlier_robustness():
    """rank: 对离群值免疫 (组内排名不受极端值影响)。"""
    print("\n[6] rank 对离群值免疫")
    normal = {0: [0.1, 0.2, 0.3, 0.4]}
    with_outlier = {0: [0.1, 0.2, 0.3, 0.4, 1000.0]}

    a = compute_group_advantages(normal, mode="rank")[0].advantages
    b = compute_group_advantages(with_outlier, mode="rank")[0].advantages
    # 加入离群值后, 前 4 个样本的 rank 位置不变 (离群值排最后),
    # 但因为组大小变了 (居中项 (n+1)/2 变了), advantage 会等比平移。
    # 关键性质: **相对次序不变**, 且变化幅度是均匀平移 (不是被离群值拉偏)。
    diffs = [y - x for x, y in zip(a, b[:4])]
    check("rank: 加入离群值后相对次序不变",
          all(a[i] < a[i + 1] for i in range(3))
          and all(b[i] < b[i + 1] for i in range(3)))
    check("rank: 变化是均匀平移 (未被离群值拉偏)",
          all(close(diffs[0], d, 1e-9) for d in diffs),
          f"diffs={diffs}")

    # zscore 会被离群值拉偏 (非均匀变化)
    az = compute_group_advantages(normal, mode="zscore")[0].advantages
    bz = compute_group_advantages(with_outlier, mode="zscore")[0].advantages
    dz = [y - x for x, y in zip(az, bz[:4])]
    check("zscore: 变化不均匀 —— 被离群值拉偏 (对照)",
          not all(close(dz[0], d, 1e-6) for d in dz), f"dz={dz}")


def test_zero_variance_skipped():
    """零方差组在三种模式下都被跳过。"""
    print("\n[7] 零方差组被跳过")
    groups = {0: [0.5, 0.5, 0.5]}
    for mode in ("zscore", "pairwise", "rank"):
        g = compute_group_advantages(groups, mode=mode)[0]
        check(f"{mode}: skipped=True", g.skipped and g.skip_reason == "zero_variance")


def test_invalid_mode():
    """非法 mode 报错。"""
    print("\n[8] 非法 mode 抛错")
    try:
        compute_group_advantages({0: [1.0, 2.0]}, mode="bogus")
        check("抛 ValueError", False)
    except ValueError:
        check("抛 ValueError", True)


def test_nan_handling():
    """NaN (无效样本) 被正确排除。"""
    print("\n[9] NaN 样本被排除")
    groups = {0: [0.5, float("nan"), 0.3]}
    for mode in ("zscore", "pairwise", "rank"):
        g = compute_group_advantages(groups, mode=mode)[0]
        check(f"{mode}: n_valid=2, n_total=3",
              g.n_valid == 2 and g.n_total == 3, f"got {g.n_valid}/{g.n_total}")


def test_flatten_consistency():
    """flatten_advantages 的顺序与 groups 一致。"""
    print("\n[10] flatten 顺序一致")
    groups = {1: [0.1, 0.5], 0: [0.2, 0.9], 2: [0.3]}
    res = compute_group_advantages(groups, mode="pairwise")
    flat = flatten_advantages(res)
    check("prompt_id 按升序", [f[0] for f in flat] == sorted(f[0] for f in flat))
    # group 2 只有 1 个样本 -> 零方差 -> 被跳过, 所以是 4 不是 5
    check("长度正确 (单样本组被跳过)", len(flat) == 4, f"got {len(flat)}")
    check("跳过原因正确", res[2].skipped and res[2].skip_reason == "zero_variance")


def test_pairwise_scaled_properties():
    """pairwise_scaled: 去偏移 + 保留组间信息量差异。

    这是 M5-E 消融的推荐模式:
      - 继承 pairwise 的"消掉绝对尺度常数偏移"
      - 同时不像纯 pairwise 那样把所有组拉到同一尺度
    """
    print("\n[11] pairwise_scaled 性质")
    base = {0: [0.1, 0.3, 0.5, 0.2]}
    shifted = {0: [x + 5.0 for x in [0.1, 0.3, 0.5, 0.2]]}
    a = compute_group_advantages(base, mode="pairwise_scaled")[0].advantages
    b = compute_group_advantages(shifted, mode="pairwise_scaled")[0].advantages
    check("对常数偏移不敏感 (继承 pairwise)",
          all(close(x, y, 1e-5) for x, y in zip(a, b)), f"{a} vs {b}")

    # 组间尺度差异被保留 (而不是像纯 pairwise 那样恒为 1)
    groups = {0: [0.0, 0.1, 0.2], 1: [0.0, 1.0, 2.0]}
    res = compute_group_advantages(groups, mode="pairwise_scaled")
    span0 = max(res[0].advantages) - min(res[0].advantages)
    span1 = max(res[1].advantages) - min(res[1].advantages)
    check("不同信息量的组给出不同尺度", span1 > span0 * 1.2,
          f"span0={span0:.4f} span1={span1:.4f}")

    # 对比: 纯 pairwise 会把两者拉到同一尺度 (跨度都 ≈ 1)
    # ⚠️ 容差 1e-4: pairwise 分母是 (rng + eps), eps=1e-6 会引入 ~5e-6 尾差
    resp = compute_group_advantages(groups, mode="pairwise")
    sp0 = max(resp[0].advantages) - min(resp[0].advantages)
    sp1 = max(resp[1].advantages) - min(resp[1].advantages)
    check("对照: 纯 pairwise 把所有组拉到同一尺度",
          close(sp0, sp1, 1e-4), f"span0={sp0:.4f} span1={sp1:.4f}")
    check("对照: 纯 pairwise 跨度恒为 1.0",
          close(sp0, 1.0, 1e-4) and close(sp1, 1.0, 1e-4),
          f"span0={sp0:.4f} span1={sp1:.4f}")


if __name__ == "__main__":
    test_zscore_backward_compat()
    test_zero_mean()
    test_pairwise_range()
    test_pairwise_shift_invariance()
    test_pairwise_scale_invariance()
    test_rank_outlier_robustness()
    test_zero_variance_skipped()
    test_invalid_mode()
    test_nan_handling()
    test_flatten_consistency()
    test_pairwise_scaled_properties()

    print(f"\n{'='*50}")
    print(f"通过 {PASS} / {PASS + FAIL}")
    if FAIL:
        print(f"❌ {FAIL} 个测试失败")
        sys.exit(1)
    print("✅ 全部通过")
