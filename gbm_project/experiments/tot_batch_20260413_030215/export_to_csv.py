#!/usr/bin/env python3
"""
将 tot_batch 实验结果整理为 CSV 格式
每个靶点一个 CSV 文件，格式: ID,SMILES
"""
import json
import csv
from pathlib import Path


def build_id(smiles: str, idx: int, target: str) -> str:
    """构建分子ID，格式: <靶点>_<smiles_hash>_<序号>"""
    # 用SMILES前6个字符的哈希做简短标识（去掉特殊字符）
    clean = ''.join(c for c in smiles[:20] if c.isalnum())
    short_hash = abs(hash(clean)) % 100000
    return f"{target}_{short_hash:05d}_{idx:03d}"


def export_target_to_csv(experiment_dir: Path, target: str, output_dir: Path):
    """导出单个靶点的分子到CSV"""
    results_file = experiment_dir / target / "tot_generation_results.json"
    if not results_file.exists():
        print(f"  [SKIP] {target}: 无结果文件")
        return 0

    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    molecules = data.get('molecules', [])
    print(f"  {target}: {len(molecules)} 个分子")

    if not molecules:
        return 0

    # 统计 TPSA 分布
    tpsas = [m.get('tpsa', 0) for m in molecules if m.get('tpsa')]
    avg_tpsa = sum(tpsas) / len(tpsas) if tpsas else 0
    print(f"    TPSA 平均: {avg_tpsa:.1f} Å², 范围: [{min(tpsas):.0f}, {max(tpsas):.0f}]" if tpsas else "    TPSA: 无数据")

    output_file = output_dir / f"{target}.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'SMILES', 'TPSA', 'MW'])
        for idx, mol in enumerate(molecules, 1):
            smiles = mol.get('smiles', '').strip()
            if smiles:
                mol_id = build_id(smiles, idx, target)
                writer.writerow([mol_id, smiles,
                                mol.get('tpsa', ''),
                                mol.get('mw', '')])

    print(f"    -> {output_file.name} ({len(molecules)} 行)")
    return len(molecules)


def main():
    experiment_dir = Path(__file__).parent
    output_dir = experiment_dir / "csv_output"
    output_dir.mkdir(exist_ok=True)

    # 找出所有靶点子目录（排除非靶点目录和隐藏文件）
    targets = sorted([
        d.name for d in experiment_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.') and d.name not in ('csv_output', '__pycache__')
    ])

    print(f"实验目录: {experiment_dir}")
    print(f"输出目录: {output_dir}")
    print(f"靶点: {targets}")
    print("=" * 60)

    total = 0
    for target in targets:
        n = export_target_to_csv(experiment_dir, target, output_dir)
        total += n

    print("=" * 60)
    print(f"完成: {len(targets)} 个靶点, 共 {total} 个分子")
    print(f"输出目录: {output_dir}")

    # 同时生成一份汇总 CSV
    all_csv = output_dir / "all_molecules.csv"
    with open(all_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'SMILES', 'TPSA', 'MW', 'Target'])
        global_idx = 1
        for target in targets:
            results_file = experiment_dir / target / "tot_generation_results.json"
            if not results_file.exists():
                continue
            with open(results_file, 'r', encoding='utf-8') as rf:
                data = json.load(rf)
            for mol in data.get('molecules', []):
                smiles = mol.get('smiles', '').strip()
                if smiles:
                    mol_id = build_id(smiles, global_idx, target)
                    writer.writerow([mol_id, smiles,
                                    mol.get('tpsa', ''),
                                    mol.get('mw', ''),
                                    target])
                    global_idx += 1

    print(f"汇总文件: {all_csv} ({global_idx - 1} 行)")


if __name__ == "__main__":
    main()
