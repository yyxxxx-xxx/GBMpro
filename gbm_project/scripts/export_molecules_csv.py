"""
RL 生成分子 -> CSV 导出
======================

把独立评估产出的分子整理成两列 CSV:  `ID,SMILES`

ID 格式:  `{靶点}_{来源标签}_{编号}`
  - 靶点名称: IDH1 (从评估目录的 train_config / 靶点配置推断)
  - 来源标签: RL (强化学习生成) / SFT (基线生成)
  - 编号: 4 位零填充, 便于排序与引用

⚠️ 多组实验的 ID 冲突: 不同消融组 (m4_a / m4_b / ...) 的分子都是
`IDH1_RL_0000` 起, 合并多个 CSV 时会撞号。用 `--tag` 给每组加前缀:
    IDH1_RL_a_0000 / IDH1_RL_b_0000 ...

用法:
    # 单个评估目录
    python3 scripts/export_molecules_csv.py ../experiments/eval_m3_kl008

    # 同时导出 RL + SFT 基线, 输出到指定路径
    python3 scripts/export_molecules_csv.py ../experiments/eval_m3_kl008 \
        --include_baseline --out ../experiments/eval_m3_kl008/molecules.csv

    # 批量导出所有评估目录 (自动从目录名推导 tag: eval_m4_a -> a)
    python3 scripts/export_molecules_csv.py ../experiments --batch

    # 合并多组到一个 CSV (自动去重)
    python3 scripts/export_molecules_csv.py ../experiments --batch \
        --merge ../experiments/all_molecules.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def infer_target(eval_dir: Path) -> str:
    """从目录名或上级 train_config 推断靶点名。"""
    # 1) 上溯找 train_config.json (训练目录里有)
    for parent in [eval_dir, *eval_dir.parents]:
        cfg = parent / "train_config.json"
        if cfg.exists():
            try:
                return json.loads(cfg.read_text(encoding="utf-8")).get(
                    "target", "IDH1")
            except Exception:
                pass
    # 2) 从 reward_cache 里的字段找 (Vina 通常带 target)
    for f in eval_dir.glob("*_reward_cache.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    d = json.loads(line)
                except Exception:
                    break
                for k in ("target", "target_name"):
                    if d.get(k):
                        return str(d[k])
                break
    # 3) 目录名兜底
    name = eval_dir.name
    for t in ("IDH1", "EGFR", "VEGFR", "VEGF"):
        if t in name.upper():
            return t
    return "IDH1"


def load_smiles(path: Path):
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for m in data:
        smi = m.get("smiles") if isinstance(m, dict) else m
        if smi:
            out.append(smi)
    return out


def write_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "SMILES"])
        w.writerows(rows)
    return len(rows)


def export_dir(eval_dir: Path, target: str, include_baseline: bool,
               out: Path | None, tag: str = ""):
    """导出一个评估目录。返回 [(路径, 行数)] 列表。"""
    written = []
    mid = f"{target}_{tag}" if tag else target

    rl = load_smiles(eval_dir / "rl_smiles.json")
    if rl:
        rows = [[f"{mid}_RL_{i:04d}", s] for i, s in enumerate(rl)]
        p = out if out else eval_dir / "rl_molecules.csv"
        n = write_csv(rows, p)
        written.append((p, n))

    if include_baseline:
        base = load_smiles(eval_dir / "baseline_sft_smiles.json")
        if base:
            rows = [[f"{mid}_SFT_{i:04d}", s] for i, s in enumerate(base)]
            p = (out.parent / (out.stem + "_baseline.csv")) if out \
                else eval_dir / "baseline_sft_molecules.csv"
            n = write_csv(rows, p)
            written.append((p, n))
    return written


def derive_tag(eval_dir: Path) -> str:
    """从目录名推导分组标签: eval_m4_a -> a, eval_m3_kl008 -> m3_kl008"""
    name = eval_dir.name
    for prefix in ("eval_",):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", help="评估目录 (含 rl_smiles.json)")
    ap.add_argument("--out", default=None, help="输出 CSV 路径")
    ap.add_argument("--include_baseline", action="store_true",
                    help="同时导出 SFT 基线分子")
    ap.add_argument("--batch", action="store_true",
                    help="批量处理: 输入目录下的所有 eval_* 子目录")
    ap.add_argument("--tag", default=None,
                    help="ID 中的分组标签 (默认单目录=无, 批量=从目录名推导)")
    ap.add_argument("--merge", default=None,
                    help="把所有分子合并导出到一个 CSV (自动去重)")
    args = ap.parse_args()

    root = Path(args.eval_dir)

    if args.batch:
        dirs = sorted(p for p in root.glob("eval_*") if p.is_dir())
        if not dirs:
            print(f"⚠️ {root} 下没有 eval_* 子目录")
            return
        merged = []
        seen = set()
        for d in dirs:
            tgt = infer_target(d)
            tag = args.tag if args.tag is not None else derive_tag(d)
            for p, n in export_dir(d, tgt, args.include_baseline, None, tag):
                print(f"{p}  ({n} 个分子, 靶点={tgt}, tag={tag})")
                if args.merge:
                    with open(p, newline="", encoding="utf-8") as f:
                        for r in csv.DictReader(f):
                            if r["SMILES"] not in seen:
                                seen.add(r["SMILES"])
                                merged.append([r["ID"], r["SMILES"]])
        if args.merge:
            mp = Path(args.merge)
            write_csv(merged, mp)
            print(f"\n合并: {mp}  ({len(merged)} 个唯一分子)")
        return

    target = infer_target(root)
    out = Path(args.out) if args.out else None
    tag = args.tag or ""
    written = export_dir(root, target, args.include_baseline, out, tag)
    if not written:
        print(f"⚠️ {root} 里没找到 rl_smiles.json")
        return
    for p, n in written:
        print(f"{p}  ({n} 个分子, 靶点={target})")


if __name__ == "__main__":
    main()
