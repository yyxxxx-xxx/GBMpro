#!/usr/bin/env python3
"""
基于现有 CSV 数据续跑 Ablation 2 / 4 实验。
从 CSV 读取已生成的分子（去重），只补充缺失的靶点。
"""
import os, sys, json, torch, argparse, traceback
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "gbm_project" / "scripts"))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TARGETS = ["EGFR", "IDH1_IDH2", "VEGF_VEGFR"]
TARGET_MOLS = 50


def load_existing_from_csv(checkpoint_dir, target):
    """从 CSV 文件读取已生成的分子"""
    csv_file = checkpoint_dir / "csv_output" / f"{target}.csv"
    if not csv_file.exists():
        return [], set()
    molecules = []
    smiles_set = set()
    with open(csv_file) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                smiles = parts[1].strip()
                if smiles and smiles not in smiles_set:
                    molecules.append({
                        'smiles': smiles,
                        'tpsa': float(parts[2]) if len(parts) > 2 and parts[2] else 0,
                        'mw': float(parts[3]) if len(parts) > 3 and parts[3] else 0,
                        'target': target,
                    })
                    smiles_set.add(smiles)
    return molecules, smiles_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--ablation', type=str, choices=['2', '4'], required=True)
    parser.add_argument('--max_attempts', type=int, default=300)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_file = checkpoint_dir / "all_results_checkpoint.json"
    gpu_id = args.gpu_id
    ablation_num = args.ablation

    # ── 加载现有分子（从 CSV）───────────────────────────────────────────
    all_results = {}
    for target in TARGETS:
        mols, _ = load_existing_from_csv(checkpoint_dir, target)
        all_results[target] = mols
        logger.info(f"  {target}: {len(mols)} molecules from CSV")

    # ── 确定需要补充的靶点 ────────────────────────────────────────────
    remaining = [t for t in TARGETS if len(all_results.get(t, [])) < TARGET_MOLS]
    logger.info(f"需要继续生成的靶点 ({len(remaining)}): {remaining}")

    if not remaining:
        logger.info("所有靶点均已满足 50 个分子")
        _save_results(all_results, checkpoint_dir, ablation_num)
        return

    # ── 构建配置 ────────────────────────────────────────────────────────
    from ablation_base import build_base_config
    _gpu_id = gpu_id
    _ablation = ablation_num
    _max_attempts = args.max_attempts

    class Opt:
        gpu_id = _gpu_id
        targets = TARGETS
        num_molecules = TARGET_MOLS
        max_attempts = _max_attempts
        max_no_new_streak = 0
        k, b = 3, 2
        output_dir = None
        use_8bit = False
    config = build_base_config(Opt())
    if ablation_num == '2':
        config['tot_refinement_rounds'] = 1

    # ── 加载模型 ────────────────────────────────────────────────────────
    from generate_tot_molecules import TreeOfThoughtsGenerator
    generator = TreeOfThoughtsGenerator(config)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    generator.load_models(use_8bit=False)

    # ── 应用 Ablation 特定 patch ───────────────────────────────────────
    if ablation_num == '2':
        orig = generator._generate_smiles_fallback
        def patched(*, scaffold, assembly, warhead, bbb_enhancers, target_mw,
                    physical_feedback=None, finetuned_prompt=None, max_new_tokens=450):
            return orig(scaffold=scaffold, assembly=assembly, warhead=warhead,
                        bbb_enhancers=bbb_enhancers, target_mw=target_mw,
                        physical_feedback=None, finetuned_prompt=finetuned_prompt,
                        max_new_tokens=max_new_tokens)
        generator._generate_smiles_fallback = patched
        logger.info("[Ablation 2] Physical feedback loop DISABLED")
    else:
        class GenericPromptOnly:
            def __init__(self, constraints):
                self.constraints = constraints
            def generate_domain_prompt(self, target_name, constraints=None):
                return ("You are a medicinal chemist designing drug candidates.\n"
                        "Target: {target_name}\n"
                        "Requirements:\n"
                        "- MW: 300-500 Da | BBB: HIGH | LogP: 2.0-4.0 | TPSA: 40-120\n\n"
                        "Output SMILES 1: <smiles>\n").format(target_name=target_name)
            def build_tot_propose_prompt(self, domain_prompt, current_state, step_type):
                return domain_prompt
            def build_tot_propose_prompt_with_feedback(self, domain_prompt, cs, st, pf=None):
                return domain_prompt
        generator.prompt_generator = GenericPromptOnly(generator.constraints)
        logger.info("[Ablation 4] Domain knowledge injection DISABLED")

    # ── 生成缺失靶点 ────────────────────────────────────────────────────
    for target in remaining:
        existing_smiles = {m['smiles'] for m in all_results.get(target, [])}
        logger.info(f"=== 生成靶点: {target} (已有 {len(existing_smiles)} 个) ===")
        try:
            mols = generator.generate_molecules(target)
            new_mols = [m for m in mols if m['smiles'] not in existing_smiles]
            all_results.setdefault(target, []).extend(new_mols)
            all_results[target] = all_results[target][:TARGET_MOLS]
            logger.info(f"  新增 {len(new_mols)} 个，总计 {len(all_results[target])} 个")
        except Exception as e:
            logger.error(f"  靶点 {target} 生成失败: {e}")
            traceback.print_exc()

        _save_results(all_results, checkpoint_dir, ablation_num)
        _append_csv(all_results[target], target, checkpoint_dir)

    _save_results(all_results, checkpoint_dir, ablation_num)
    logger.info(f"\n{'='*60}\nAblation {ablation_num} 续跑完成\n{'='*60}")
    for t, m in all_results.items():
        logger.info(f"  {t}: {len(m)} / {TARGET_MOLS}")


def _save_results(all_results, checkpoint_dir, ablation_num):
    checkpoint_file = checkpoint_dir / "all_results_checkpoint.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    with open(checkpoint_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    with open(checkpoint_dir / "summary.json", 'w') as f:
        json.dump({
            'experiment_name': f'ablation_{ablation_num}',
            'targets_processed': list(all_results.keys()),
            'per_target': {t: len(all_results.get(t, [])) for t in TARGETS},
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)


def _append_csv(molecules, target_name, output_dir):
    csv_dir = output_dir / "csv_output"
    csv_dir.mkdir(exist_ok=True)
    csv_file = csv_dir / f"{target_name}.csv"
    existing_smiles = set()
    if csv_file.exists():
        with open(csv_file) as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(',', 2)
                if len(parts) >= 2:
                    existing_smiles.add(parts[1])
    lines = []
    for i, mol in enumerate(molecules):
        smiles = mol.get('smiles', '')
        if smiles and smiles not in existing_smiles:
            lines.append(f"{target_name}_{i:05d},{smiles},{mol.get('tpsa',0):.2f},{mol.get('mw',0):.2f},{target_name}\n")
            existing_smiles.add(smiles)
    if lines:
        with open(csv_file, 'a') as f:
            f.writelines(lines)


if __name__ == "__main__":
    main()
