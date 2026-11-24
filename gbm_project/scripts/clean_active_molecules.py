#!/usr/bin/env python3
"""
活性分子数据清洗: 多标签冲突处理
=================================

借鉴自 PBCNet2.0 的训练数据构建 (见 docs/PBCNET2_0_BENCHMARKS.md §7)::

    "if a molecule in a chemical series had multiple affinity labels, we
     followed a specific protocol: if the maximum affinity label of the
     molecule was less than three times its minimum affinity label, we used
     the average value as the final label; if the maximum label exceeded
     three times the minimum, we removed the molecule to ensure data quality."

为什么需要这一步
----------------

同一个 CID (PubChem 化合物) 在不同 assay 下可能有多个活性值:
  - 多个 assay 的重复测量 (正常, 应取平均)
  - 来自不同文献/不同实验条件 (矛盾, 说明数据不可靠)

我们的 SFT 数据 active_qa_train.json 直接取了其中一个值, 等于
**随机选了一个可能矛盾的标签**。这会让模型学到噪声。

处理规则
--------

1. 同 (cid, activity_name, target) 分组
2. 取该组所有 value_um 的最小值与最大值
3. 若 max <= 3 * min  -> 取**几何平均** (保留数量级信息, 亲和力是对数尺度)
4. 若 max >  3 * min  -> **丢弃该条记录**, 但**不丢弃整个分子**:
   保留该分子**最强的一条** (min value_um, 即活性最好的测量),
   并标记 `conflict=True` 供人工复核

为什么用几何平均而不是算术平均
------------------------------

亲和力 (IC50/EC50/Ki/Kd) 跨越多个数量级, 在**对数尺度**上取平均
才对应"几何平均"。算术平均会被一个极端离群值主导。
这与 PBCNet2.0 的 pAct 线性化处理一致 (pAct = -log10(IC50))。

关于"丢弃整个分子" vs "保留最强测量"
-------------------------------------

PBCNet2.0 的原文是 "removed the molecule"。但我们的数据场景不同:
原始 900 条里只有 **583 个不同分子**, 重复来自 PubChem 多 assay 检索
(同一分子在多个 assay 下的多条测量)。如果照搬"丢弃整个分子",
EGFR 会从 150 条掉到 67 条 —— 损失的不只是冲突样本, 还包括
那些"多次测量但都一致"的好数据。

所以这里采取更保守的策略:
- 一致的多条测量 -> 合并为 1 条 (几何平均), **分子数不丢**
- 冲突的分子 -> 保留最强的那条测量 (活性最好), 标记 conflict
- 冲突分子的其他测量 -> 丢弃

这样既去除了矛盾标签的噪声, 又不损失靶点覆盖度。
净效果: 900 条 -> 583 条 (每分子 1 条), 其中 21 条标记了 conflict。

用法:
    python3 scripts/clean_active_molecules.py                # 报告模式
    python3 scripts/clean_active_molecules.py --apply        # 写回文件
    python3 scripts/clean_active_molecules.py --apply --out data/lora_datasets/active_molecules_6targets_clean.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

GBM_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = GBM_ROOT / "data" / "lora_datasets" / "active_molecules_6targets.json"

CONFLICT_RATIO = 3.0


def geometric_mean(values):
    """几何平均 —— 对数量级跨度大的亲和力更稳健。"""
    vs = [v for v in values if v is not None and v > 0]
    if not vs:
        return None
    logs = [math.log10(v) for v in vs]
    return 10 ** (sum(logs) / len(logs))


def group_key(rec):
    """分组键: 同一化合物 + 同一活性类型 + 同一靶点。"""
    return (str(rec.get("cid", "")),
            str(rec.get("activity_name", "")).strip().upper(),
            str(rec.get("target", "")))


def analyze(records):
    """分组并判定冲突。返回 (cleaned, stats)。"""
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[group_key(r)].append((i, r))

    kept = []
    dropped = []
    merged = []
    conflicts = []

    for key, items in groups.items():
        # 剔除非法值 (None / <=0) —— 这些记录本身不可用
        usable = [(i, r) for i, r in items
                  if isinstance(r.get("value_um"), (int, float))
                  and r.get("value_um") > 0]
        for i, r in items:
            if (i, r) not in usable:
                dropped.append((i, r, "invalid_value"))

        if not usable:
            continue

        vals = [r["value_um"] for _, r in usable]
        vmin, vmax = min(vals), max(vals)
        is_conflict = len(usable) > 1 and vmax > CONFLICT_RATIO * vmin

        if is_conflict:
            # 冲突: 只保留最强的一条 (value_um 最小 = 活性最好)
            usable.sort(key=lambda x: x[1]["value_um"])
            base_idx, base = usable[0]
            new_rec = dict(base)
            new_rec["value_um"] = base["value_um"]
            new_rec["n_labels_merged"] = 1
            new_rec["conflict"] = True
            new_rec["label_range_um"] = [vmin, vmax]
            new_rec["conflict_ratio"] = round(vmax / vmin, 2)
            for i, r in usable[1:]:
                dropped.append((i, r, "conflict_weak_measurement"))
            kept.append((base_idx, new_rec))
            conflicts.append((base_idx, new_rec, len(usable), vmin, vmax))
        else:
            # 一致: 几何平均合并
            base_idx, base = usable[0]
            new_rec = dict(base)
            new_rec["value_um"] = geometric_mean(vals)
            new_rec["n_labels_merged"] = len(usable)
            if len(usable) > 1:
                new_rec["label_range_um"] = [vmin, vmax]
            kept.append((base_idx, new_rec))
            if len(usable) > 1:
                merged.append((base_idx, new_rec, len(usable), vmin, vmax))

    kept.sort(key=lambda x: x[0])
    out = [r for _, r in kept]
    return out, {"dropped": dropped, "merged": merged,
                 "conflicts": conflicts, "n_groups": len(groups)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default=None,
                    help="输出路径 (不给则只覆盖 --apply 指定的源文件)")
    ap.add_argument("--apply", action="store_true",
                    help="真正写出文件 (默认只报告)")
    args = ap.parse_args()

    src = Path(args.src)
    records = json.loads(src.read_text(encoding="utf-8"))
    print(f"源文件: {src}")
    print(f"原始条数: {len(records)}")

    cleaned, stats = analyze(records)

    dropped = stats["dropped"]
    merged = stats["merged"]
    conflicts = stats["conflicts"]
    reasons = defaultdict(int)
    for _, _, r in dropped:
        reasons[r] += 1

    print(f"\n分组数: {stats['n_groups']}")
    print(f"清洗后条数: {len(cleaned)}")
    print(f"  合并多标签的分子: {len(merged)}")
    print(f"  冲突分子 (保留最强测量): {len(conflicts)}")
    print(f"  丢弃的记录: {len(dropped)}")
    for r, n in reasons.items():
        print(f"    - {r}: {n}")

    if merged:
        print("\n多标签合并样例 (前 6 条):")
        print(f"  {'CID':<12}{'靶点':<18}{'类型':<8}{'原值范围':<26}{'聚合值'}")
        for _, r, n, vmin, vmax in merged[:6]:
            print(f"  {str(r.get('cid')):<12}{str(r.get('target')):<18}"
                  f"{str(r.get('activity_name')):<8}"
                  f"{vmin:.1e} ~ {vmax:.1e}   {r['value_um']:.3e}")

    if conflicts:
        print("\n冲突分子样例 (保留最强测量):")
        print(f"  {'CID':<12}{'靶点':<18}{'类型':<8}{'原值范围':<26}"
              f"{'倍数':>7}{'保留值'}")
        for _, r, n, vmin, vmax in conflicts[:6]:
            print(f"  {str(r.get('cid')):<12}{str(r.get('target')):<18}"
                  f"{str(r.get('activity_name')):<8}"
                  f"{vmin:.1e} ~ {vmax:.1e}   {r.get('conflict_ratio', 0):>6.1f}x"
                  f"  {r['value_um']:.3e}")

    # 靶点分布变化
    from collections import Counter
    before = Counter(r.get("target") for r in records)
    after = Counter(r.get("target") for r in cleaned)
    print(f"\n靶点分布:")
    print(f"  {'靶点':<20}{'清洗前':>8}{'清洗后':>8}{'Δ':>8}")
    for t in sorted(set(before) | set(after)):
        d = after[t] - before[t]
        flag = "  ⚠️" if d < 0 and after[t] < before[t] * 0.9 else ""
        print(f"  {str(t):<20}{before[t]:>8}{after[t]:>8}{d:>+8}{flag}")

    if not args.apply:
        print("\n(报告模式, 未写文件。加 --apply 生效)")
        return

    out_path = Path(args.out) if args.out else src
    # 备份原文件
    if out_path == src:
        bak = src.with_suffix(src.suffix + ".bak")
        bak.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"\n原文件已备份: {bak}")

    out_path.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"已写出: {out_path} ({len(cleaned)} 条)")


if __name__ == "__main__":
    main()
