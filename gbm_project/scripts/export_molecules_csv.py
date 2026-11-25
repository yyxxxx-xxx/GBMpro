#!/usr/bin/env python3
"""
RL 生成分子导出 CSV (ID, SMILES)
=================================

把 RL 训练产出的分子整理成两列 CSV:
    ID,SMILES
    EGFR_m5_egfr_zscore_RL_0000,Cc1cc(C#N)c([O-])c(OCc2ccc(F)c(Br)c2)n1
    ...

ID 格式: {靶点大写}_{实验标签}_{RL|BASE}_{4位序号}
    - 靶点名: EGFR / VEGFR / IDH1 ...
    - 实验标签: 与训练目录同名, 如 m5_egfr_zscore / m5_egfr_pairwise
    - RL / BASE: 区分强化学习产物与 SFT 基线产物
    - 4 位序号: 从 0000 开始, 按首次出现顺序 (与 reward_cache 的记录顺序一致,
                便于和打分缓存逐行对照)

过滤规则
--------

⚠️ **默认剔除被硬约束截断 (pruned) 的分子**。
   这些分子触发了 reward 硬约束 (结构守卫违规 / 端点越界),
   它们在训练中被 prune 掉, 不进入 advantage, 化学上也不可接受。
   用 --include_pruned 可以保留 (用于排查), 但会在 ID 后加 "_P" 标记。

去重
----
   同一分子可能在多个 rollout 步被重复生成。
   --dedup (默认开): 只保留**首次出现**的, 保证 ID 连续且无重复行
   --keep_dup: 保留全部 (用于统计多样性/重复率)

用法
----

    # 导出全部 M5 实验
    python3 scripts/export_molecules_csv.py --m5 --out_dir ../experiments/csv

    # 导出单个实验
    python3 scripts/export_molecules_csv.py \
        --exp_dir ../experiments/m5_egfr_zscore --target EGFR

    # 保留被 prune 的分子 (排查用)
    python3 scripts/export_molecules_csv.py --m5 --include_pruned

    # 保留重复分子 (统计用)
    python3 scripts/export_molecules_csv.py --m5 --keep_dup
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

GBM_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = GBM_ROOT.parent
EXP_ROOT = REPO_ROOT / "experiments"


def infer_target(exp_dir: Path) -> str:
    """从 train_config.json 读靶点, 读不到就从目录名猜。"""
    cfg = exp_dir / "train_config.json"
    if cfg.exists():
        try:
            d = json.loads(cfg.read_text(encoding="utf-8"))
            if d.get("target"):
                return str(d["target"]).upper()
        except Exception:
            pass
    # 目录名如 m5_egfr_zscore -> egfr
    m = re.search(r"(egfr|vegfr|idh1|mgmt|pi3k[a-z_]*|p53[a-z_]*)", exp_dir.name.lower())
    return (m.group(1).upper() if m else "UNK")


def exp_tag(exp_dir: Path) -> str:
    """实验标签: 目录名去 experiments/ 前缀。"""
    return exp_dir.name


def load_molecules(exp_dir: Path, include_pruned: bool):
    """从 reward_cache.jsonl 读取分子。

    返回 [(smiles, reward, pruned, first_step_seen)]
    """
    cache = exp_dir / "reward_cache.jsonl"
    if not cache.exists():
        return []

    out = []
    with cache.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            smi = d.get("smiles")
            if not smi:
                continue
            pruned = bool(d.get("pruned")) or bool(d.get("guard_violation"))
            if pruned and not include_pruned:
                continue
            out.append((smi, d.get("reward"), pruned))
    return out


def write_csv(path: Path, rows, tag_target: str, tag: str, kind: str):
    """写 CSV。rows = [(smiles, reward, pruned), ...]"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")   # 显式 LF, 避免 Windows CRLF
        w.writerow(["ID", "SMILES"])
        for i, (smi, reward, pruned) in enumerate(rows):
            suffix = "_P" if pruned else ""
            mid = f"{tag_target}_{tag}_{kind}_{i:04d}{suffix}"
            w.writerow([mid, smi])
    return len(rows)


def summarize(rows, total_raw):
    n_pruned = sum(1 for r in rows if r[2])
    uniq = len({r[0] for r in rows})
    rews = [r[1] for r in rows if isinstance(r[1], (int, float))]
    return {
        "n_exported": len(rows),
        "n_raw": total_raw,
        "n_pruned_excluded": total_raw - len(rows) + n_pruned,
        "n_unique": uniq,
        "duplicate_rate": round(1 - uniq / len(rows), 4) if rows else 0.0,
        "reward_mean": round(sum(rews) / len(rews), 4) if rews else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", default=None, help="单个实验目录")
    ap.add_argument("--m5", action="store_true", help="导出所有 m5_* 实验")
    ap.add_argument("--target", default=None, help="覆盖靶点名")
    ap.add_argument("--out_dir", default=None, help="输出目录")
    ap.add_argument("--out", default=None, help="单个输出文件路径")
    ap.add_argument("--include_pruned", action="store_true",
                    help="保留被硬约束截断的分子 (ID 加 _P 标记)")
    ap.add_argument("--keep_dup", action="store_true",
                    help="保留重复分子 (默认去重, 只留首次出现)")
    args = ap.parse_args()

    if args.exp_dir:
        exp_dirs = [Path(args.exp_dir)]
    elif args.m5:
        exp_dirs = sorted([d for d in EXP_ROOT.glob("m5_*")
                            if d.is_dir() and (d / "reward_cache.jsonl").exists()])
    else:
        ap.error("需要 --exp_dir 或 --m5")

    if not exp_dirs:
        print("没有找到实验目录")
        return

    out_dir = Path(args.out_dir) if args.out_dir else EXP_ROOT / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"输出目录: {out_dir}")
    print(f"实验数: {len(exp_dirs)}\n")

    all_summaries = []
    for d in exp_dirs:
        target = (args.target or infer_target(d)).upper()
        tag = exp_tag(d)

        raw = load_molecules(d, include_pruned=True)
        rows = load_molecules(d, include_pruned=args.include_pruned)

        if not args.keep_dup:
            seen = set()
            deduped = []
            for smi, rew, pr in rows:
                if smi in seen:
                    continue
                seen.add(smi)
                deduped.append((smi, rew, pr))
            rows = deduped

        if args.out:
            out_path = Path(args.out)
        else:
            out_path = out_dir / f"{target.lower()}_{tag}_molecules.csv"

        n = write_csv(out_path, rows, target, tag, "RL")

        s = summarize(rows, len(raw))
        s["experiment"] = tag
        s["target"] = target
        s["file"] = str(out_path)
        all_summaries.append(s)

        print(f"── {tag} ({target})")
        print(f"   原始分子      : {s['n_raw']}")
        print(f"   剔除 prune 后 : {s['n_exported']}  "
              f"(剔除 {s['n_pruned_excluded']})")
        print(f"   唯一分子      : {s['n_unique']}  "
              f"(重复率 {s['duplicate_rate']*100:.1f}%)")
        if s["reward_mean"] is not None:
            print(f"   平均 reward   : {s['reward_mean']}")
        print(f"   → {out_path}\n")

    # 汇总表
    print("=" * 74)
    print(f"{'实验':<24}{'靶点':<10}{'导出':>7}{'唯一':>7}"
          f"{'重复率':>9}{'平均reward':>12}")
    print("-" * 74)
    for s in all_summaries:
        print(f"{s['experiment']:<24}{s['target']:<10}{s['n_exported']:>7}"
              f"{s['n_unique']:>7}{s['duplicate_rate']*100:>8.1f}%"
              f"{(s['reward_mean'] if s['reward_mean'] is not None else 0):>12.4f}")
    print("=" * 74)

    (out_dir / "export_summary.json").write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n汇总: {out_dir / 'export_summary.json'}")


if __name__ == "__main__":
    main()
