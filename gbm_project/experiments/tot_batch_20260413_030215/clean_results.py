#!/usr/bin/env python3
"""
用 RDKit 重新清理已生成结果中的无效分子（离子、片段、过短分子、TPSA越界）
"""
import json
import re
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors


TPSA_MIN = 20.0
TPSA_MAX = 120.0


def rdkit_validate(smiles: str):
    """
    用 RDKit 全面验证分子：可解析性、TPSA、MW 范围。
    返回 (是否有效, reason_if_invalid, tpsa, mw).
    """
    s = smiles.strip()
    if len(s) < 5:
        return False, "too_short", 0.0, 0.0

    # 仅允许 SMILES 字符
    if not re.match(r'^[A-Za-z0-9@+\-\[\]\(\)=#%/\\.]+$', s):
        return False, "invalid_chars", 0.0, 0.0

    # 方括号离子 [X+]/[X-]
    if re.match(r'^\[[A-Za-z][A-Za-z0-9]*[+-]?\]$', s):
        return False, "bare_ion", 0.0, 0.0

    # 带电原子 [N+], [O-] 等
    if re.search(r'\[[A-Za-z][A-Za-z0-9]*[+-]', s):
        return False, "charged_atom", 0.0, 0.0

    # 括号平衡
    if sum(1 for c in s if c in '([{') != sum(1 for c in s if c in ')]}'):
        return False, "unbalanced_brackets", 0.0, 0.0

    # RDKit 解析
    try:
        mol = Chem.MolFromSmiles(s)
    except Exception:
        return False, "rdkit_parse_fail", 0.0, 0.0

    if mol is None:
        return False, "rdkit_null", 0.0, 0.0

    try:
        tpsa = Descriptors.TPSA(mol)
        mw = Descriptors.MolWt(mol)
    except Exception:
        return False, "rdkit_property_fail", 0.0, 0.0

    # TPSA 过滤
    if not (TPSA_MIN <= tpsa <= TPSA_MAX):
        return False, f"tpsa_{tpsa:.1f}_out_of_range", tpsa, mw

    # MW 过滤（宽松范围）
    if mw < 150 or mw > 700:
        return False, f"mw_{mw:.1f}_out_of_range", tpsa, mw

    # 至少2个C原子
    if s.count('C') + s.count('c') < 2:
        return False, "too_few_carbons", tpsa, mw

    return True, "", tpsa, mw


def clean_target(experiment_dir: Path, target: str):
    """清理单个靶点的结果"""
    results_file = experiment_dir / target / "tot_generation_results.json"
    if not results_file.exists():
        print(f"  [SKIP] {target}: 无结果文件")
        return None

    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    molecules = data.get('molecules', [])
    original_count = len(molecules)

    reason_counts = {}
    seen = set()
    valid_molecules = []

    for m in molecules:
        smiles = m.get('smiles', '').strip()
        if smiles in seen:
            continue  # 去重
        seen.add(smiles)

        ok, reason, tpsa, mw = rdkit_validate(smiles)
        if not ok:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            continue

        m['tpsa'] = round(tpsa, 2)
        m['mw'] = round(mw, 2)
        valid_molecules.append(m)

    removed = original_count - len(valid_molecules)
    print(f"  {target}: {original_count} -> {len(valid_molecules)} (移除 {removed})")
    if reason_counts:
        for reason, cnt in sorted(reason_counts.items()):
            print(f"    - {reason}: {cnt}")

    data['molecules'] = valid_molecules
    data['num_generated'] = len(valid_molecules)
    data['cleaned_rdkits'] = True
    data['original_count'] = original_count
    data['tpsa_range_filter'] = f"{TPSA_MIN}-{TPSA_MAX}"

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {'target': target, 'original': original_count,
            'valid': len(valid_molecules), 'removed': removed}


def main():
    experiment_dir = Path(__file__).parent
    skip_dirs = {'csv_output', '__pycache__', 'scripts', 'models', 'data', 'configs', 'experiments'}
    targets = sorted([
        d.name for d in experiment_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.') and d.name not in skip_dirs
    ])

    print(f"实验目录: {experiment_dir}")
    print(f"RDKit TPSA 过滤范围: {TPSA_MIN}-{TPSA_MAX} Å²")
    print("=" * 60)

    results = [clean_target(experiment_dir, t) for t in targets if clean_target(experiment_dir, t) is not None]

    print("=" * 60)
    print("清理完成:")
    total_orig = sum(r['original'] for r in results)
    total_valid = sum(r['valid'] for r in results)
    total_rem = sum(r['removed'] for r in results)
    print(f"  总计: {total_orig} -> {total_valid} (移除 {total_rem})")
    for r in results:
        print(f"    {r['target']}: {r['original']} -> {r['valid']} (移除 {r['removed']})")


if __name__ == "__main__":
    main()
