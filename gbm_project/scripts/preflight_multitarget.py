"""
靶点特异性预检 (M5 前置验证)
============================

在花 1.5 小时训练之前, 先确认 EGFR / VEGFR / IDH1 三个靶点的 reward
**确实互相区分**。如果三个靶点给同一个分子的分数完全一样, 说明
靶点映射 (receptors.yaml 的 target_aliases) 或 receptor pdbqt 有问题,
训练再久也没意义。

检查项:
  1. 每个靶点的 Vina 后端与受体文件
  2. 共晶配体重对接 —— 应该落在合理的亲和力区间 (sanity check)
  3. 靶点交叉矩阵 —— 同一分子在不同靶点下分数应不同
  4. reward 与 vina 的相关性 —— 应强正相关 (reward 的核心项)

用法:
    python3 scripts/preflight_multitarget.py
    python3 scripts/preflight_multitarget.py --targets EGFR VEGFR IDH1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GBM_ROOT = PROJECT_ROOT / "gbm_project"
for p in (GBM_ROOT / "src",):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np

# ⚠️ admet-ai 会自动抢 GPU 显存 —— 锁死 CPU, 否则和训练抢卡
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 各靶点的共晶配体 (来自 configs/receptors.yaml)
CRYSTAL_LIGANDS = {
    "EGFR": "CCOC1CC2NCC(CN)C(NC3CCC(OCC4CCCCN4)C(Cl)C3)C2CC1NC(O)CCCN(C)C",
    "IDH1": "C[C@H](NC1=CC=C(C#N)N(C)C1=O)C2=Cc3cc(Cl)ccc3NC2=O",
}

# 三个靶点共用的探测分子 —— 用于交叉矩阵
# ⚠️ 必须是**类药性**分子 (MW 250-400): reward 里有 MW/TPSA 硬截断,
#    分子太小会直接归零 (reward=0), 在退化数据上算相关系数没有意义。
PROBE_MOLECULES = [
    # (名称, SMILES)  —— 均为真实上市药物, MW 250-400
    ("阿司匹林衍生物", "CC(=O)Oc1ccccc1C(=O)Nc1ccc(C)cc1"),      # MW 269
    ("萘普生",         "COc1ccc2cc(ccc2c1)[C@@H](C)C(=O)O"),      # MW 230
    ("卡洛芬",         "CC(C)Cc1ccc(C(C)C(=O)O)cc1"),              # MW 206
    ("布洛芬",         "CC(C)Cc1ccc(C(C)C(=O)O)c(C)c1"),          # MW 206
    ("吲哚美辛",       "COc1ccc2[nH]c(C(=O)c3ccc(Cl)cc3Cl)c2c1"), # MW 357
    ("吉非替尼类似",   "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"),  # MW 447
    ("来曲唑类似",     "N#Cc1ccc(C(c2ccc(C#N)cc2)n2cncn2)cc1"),   # MW 326
    ("塞来昔布类似",   "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"),  # MW 381
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=["EGFR", "VEGFR", "IDH1"])
    ap.add_argument("--n_workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from rl.reward_service import RewardService, RewardConfig
    from gbm_target_evaluator import build_evaluator_for_target

    results = {}
    matrices = {}

    for tgt in args.targets:
        print(f"\n{'='*64}")
        print(f"靶点: {tgt}")
        print(f"{'='*64}")

        # --- 0. 受体 / 后端检查 (直接构造, 不走进程池) ---
        ev = build_evaluator_for_target(tgt, exhaustiveness=4,
                                        cpu=4, use_admet_ai=True)
        if ev is None:
            print(f"  ❌ 无法为 {tgt} 构建评估器 (受体缺失或别名映射失败)")
            results[tgt] = {"error": "evaluator build failed"}
            continue

        tool = getattr(ev, "_vina_tool", None)
        backend = getattr(tool, "_backend", "unknown")
        receptor = getattr(tool, "receptor_pdbqt", None)
        print(f"  Vina 后端 : {backend}")
        print(f"  受体文件 : {receptor}")
        if receptor and Path(str(receptor)).exists():
            print(f"  受体大小 : {Path(str(receptor)).stat().st_size/1024:.0f} KB")
        print(f"  对接盒   : center={getattr(tool,'center',None)} "
              f"size={getattr(tool,'size',None)}")

        svc = RewardService(RewardConfig(
            target=tgt, n_workers=args.n_workers,
            cache_path=Path(f"/tmp/preflight_{tgt}.jsonl")))
        try:
            # --- 1. 共晶配体 sanity check ---
            if tgt in CRYSTAL_LIGANDS:
                recs = svc.score_batch([CRYSTAL_LIGANDS[tgt]])
                v = recs[0].vina
                ok = v is not None and -12.0 <= v <= -4.0
                print(f"  共晶配体重对接: vina = {v}  "
                      f"{'✅ 合理区间' if ok else '⚠️ 超出 [-12,-4]'}")
                results.setdefault(tgt, {})["crystal_vina"] = v

            # --- 2. 探测分子 ---
            recs = svc.score_batch([s for _, s in PROBE_MOLECULES])
            rewards = [r.reward for r in recs]
            vines = [r.vina for r in recs]
            print(f"  探测分子 n={len(recs)}")
            for (name, smi), r, v in zip(PROBE_MOLECULES, rewards, vines):
                vtxt = "None" if v is None else f"{v:.3f}"
                print(f"    {name:14s} reward={r:+.4f}  vina={vtxt:>9s}  {smi[:40]}")

            matrices[tgt] = {"reward": rewards, "vina": vines}

            # --- 3. reward 与 vina 相关性 (只统计 reward>0 的非退化样本) ---
            ok_pairs = [(r, v) for r, v in zip(rewards, vines)
                        if v is not None and r > 0]
            print(f"  非零 reward 样本: {len(ok_pairs)}/{len(rewards)}")
            if len(ok_pairs) >= 3:
                rr = np.array([p[0] for p in ok_pairs])
                vv = np.array([p[1] for p in ok_pairs])
                # reward 越高越好, vina 越负越好 -> 应该正相关
                corr = float(np.corrcoef(rr, vv)[0, 1])
                print(f"  corr(reward, vina) = {corr:+.3f}  "
                      f"{'✅ 强正相关' if corr > 0.5 else '⚠️ 弱相关'}")
                results.setdefault(tgt, {})["corr_reward_vina"] = corr
            else:
                print(f"  ⚠️ 非零 reward 样本不足, 跳过相关性检验")

        finally:
            svc.close()

    # --- 4. 靶点交叉矩阵 ---
    print(f"\n{'='*64}")
    print("靶点交叉矩阵 (同一分子, 不同靶点的 vina)")
    print(f"{'='*64}")
    tgts = list(matrices.keys())
    header = "  molecule".ljust(20) + "".join(f"{t:>12s}" for t in tgts)
    print(header)
    for i, (name, smi) in enumerate(PROBE_MOLECULES):
        row = f"  {name:18s}"
        for t in tgts:
            v = matrices[t]["vina"][i]
            row += f"{'None' if v is None else format(v, '>12.3f')}"
        print(row)
    # --- 5. 判据: 不同靶点必须给出不同分数 ---
    print(f"\n{'='*64}")
    print("靶点特异性判定")
    print(f"{'='*64}")
    ok_all = True
    for t in tgts:
        if t == tgts[0]:
            continue
        a = np.array([v if v is not None else 0.0
                      for v in matrices[tgts[0]]["vina"]])
        b = np.array([v if v is not None else 0.0
                      for v in matrices[t]["vina"]])
        d = float(np.abs(a - b).mean())
        same = d < 0.5
        print(f"  {tgts[0]} vs {t}: 平均 |Δvina| = {d:.3f}  "
              f"{'❌ 分数几乎相同 —— 靶点没生效!' if same else '✅ 已区分'}")
        ok_all &= not same

    print(f"\n总体判定: {'✅ 通过, 可以开始训练' if ok_all else '❌ 未通过, 先修靶点映射'}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"results": results, "matrices": matrices},
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"已保存: {args.out}")


if __name__ == "__main__":
    main()
