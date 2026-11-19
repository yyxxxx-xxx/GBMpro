#!/usr/bin/env python3
"""
续跑 Ablation 4 (No Domain Knowledge) 实验，只处理 checkpoint 中分子数不足 50 的靶点。
读取现有的 all_results_checkpoint.json，追加生成缺失的靶点，最后合并结果。
"""
import logging
import sys
import json
import torch
import argparse
import traceback
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "gbm_project" / "scripts"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(PROJECT_ROOT / "gbm_project" / "experiments"))
from generate_tot_molecules import TreeOfThoughtsGenerator
from ablation_base import build_base_config

TARGETS = ["EGFR", "IDH1_IDH2", "VEGF_VEGFR"]
TARGET_MOLS = 50
GPU_ID = 2


GENERIC_DOMAIN_PROMPT = """You are a medicinal chemist designing drug candidates.

Target: {target_name}

Design Requirements:
- Molecular Weight (MW): 300-500 Da
- Blood-Brain Barrier (BBB) penetration potential: HIGH
- LogP: 2.0-4.0
- Topological Polar Surface Area (TPSA): 40-120 Å²
- Chemically valid and synthesizable

Output your candidates as SMILES strings in the format:
SMILES 1: <smiles>
SMILES 2: <smiles>
SMILES 3: <smiles>
"""


class GenericPromptOnly:
    """禁用 GBM 知识库，只用通用 prompt。"""
    def __init__(self, constraints):
        self.constraints = constraints

    def generate_domain_prompt(self, target_name, constraints=None):
        return GENERIC_DOMAIN_PROMPT.format(target_name=target_name)

    def build_tot_propose_prompt(self, domain_prompt, current_state, step_type):
        if step_type == "scaffold":
            return (
                f"{domain_prompt}\n\n"
                "Step: Propose 3 heterocyclic scaffolds suitable for BBB-penetrating drugs.\n"
                "Requirements:\n"
                "  - MW: 100-250 Da\n"
                "  - Known BBB-friendly scaffolds\n"
                "  - Easy to functionalize\n\n"
                "Output format:\n"
                "Scaffold 1:\n"
                "Name: <scaffold_name>\n"
                "Rationale: <one-sentence>\n"
                "Base MW: <N> Da\n"
                "BBB Potential: <high/medium/low>\n\n"
                "Scaffold 2: ...\n"
                "Scaffold 3: ..."
            )
        elif step_type == "assembly":
            scaffold = current_state.get("selected_scaffold", "heterocycle")
            return (
                f"{domain_prompt}\n\n"
                f"Selected scaffold: {scaffold}\n\n"
                "Step: Design molecular assembly with warhead and BBB enhancers.\n"
                "  - Warhead: electrophilic group for target binding\n"
                "  - BBB enhancers: lipophilic groups (F, OCH3, CF3)\n"
                "  - Keep total MW 300-500 Da\n\n"
                "Output format:\n"
                "Strategy 1:\n"
                "Warhead: <warhead>\n"
                "BBB Enhancers: <enhancers>\n"
                "Estimated Total MW: <N> Da\n"
                "Expected LogP: <value>\n"
                "Expected TPSA: <value> Å²\n"
                "Rationale: <one-sentence>\n\n"
                "Strategy 2: ...\n"
                "Strategy 3: ..."
            )
        elif step_type == "smiles":
            scaffold = current_state.get("selected_scaffold", "heterocycle")
            warhead = current_state.get("warhead_type", "electrophile")
            enhancers = current_state.get("bbb_enhancers", "lipophilic groups")
            return (
                f"{domain_prompt}\n\n"
                f"Scaffold: {scaffold}\n"
                f"Warhead: {warhead}\n"
                f"BBB Enhancers: {enhancers}\n\n"
                "Step: Generate the final SMILES string.\n"
                "Requirements:\n"
                "  - TPSA: 40-120 Å²\n"
                "  - MW: 300-500 Da\n"
                "  - LogP: 2.0-4.0\n\n"
                "Output format:\n"
                "SMILES 1: <valid_smiles>\n"
                "SMILES 2: <valid_smiles>\n"
                "SMILES 3: <valid_smiles>"
            )
        return domain_prompt

    def build_tot_propose_prompt_with_feedback(self, domain_prompt, current_state, step_type, physical_feedback=None):
        base = self.build_tot_propose_prompt(domain_prompt, current_state, step_type)
        if physical_feedback:
            base += f"\n\nPhysical feedback from previous round: {physical_feedback}"
        return base


def main():
    parser = argparse.ArgumentParser(description="续跑 Ablation 4 No Domain Knowledge 实验")
    parser.add_argument('--checkpoint_dir', type=str,
                        default=str(PROJECT_ROOT / "gbm_project" / "experiments" / "ablation_4_no_domain_knowledge_20260423_144034"))
    parser.add_argument('--num_molecules', type=int, default=TARGET_MOLS)
    parser.add_argument('--gpu_id', type=int, default=GPU_ID)
    parser.add_argument('--max_attempts', type=int, default=300)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_file = checkpoint_dir / "all_results_checkpoint.json"

    all_results = {}
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            all_results = json.load(f)
        logger.info(f"已加载 checkpoint，现有结果: {[(t, len(m)) for t, m in all_results.items()]}")

    remaining = [t for t in TARGETS if len(all_results.get(t, [])) < TARGET_MOLS]
    logger.info(f"需继续生成的靶点 ({len(remaining)}): {remaining}")

    if not remaining:
        logger.info("所有靶点均已满足 50 个分子，无需续跑")
        return

    class Args:
        gpu_id = args.gpu_id
        targets = TARGETS
        num_molecules = args.num_molecules
        max_attempts = args.max_attempts
        max_no_new_streak = 0
        k = 3
        b = 2
        output_dir = None
        use_8bit = False

    config = build_base_config(Args())

    logger.info("初始化生成器...")
    generator = TreeOfThoughtsGenerator(config)

    logger.info("加载模型...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    generator.load_models(use_8bit=False)

    # 替换为通用 prompt（禁用 GBM 知识库）
    generator.prompt_generator = GenericPromptOnly(generator.constraints)
    logger.info("[Ablation 4] Domain knowledge injection DISABLED (using generic prompts)")

    for target in remaining:
        existing_smiles = {m['smiles'] for m in all_results.get(target, [])}
        logger.info(f"=== 续跑靶点: {target} (已有 {len(existing_smiles)} 个) ===")

        attempt = 0
        while len(all_results.get(target, [])) < TARGET_MOLS:
            attempt += 1
            try:
                molecules = generator.generate_molecules(target)
                new_mols = [m for m in molecules if m['smiles'] not in existing_smiles]
                all_results.setdefault(target, []).extend(new_mols)
                for m in new_mols:
                    existing_smiles.add(m['smiles'])
                all_results[target] = all_results[target][:TARGET_MOLS]

                logger.info(f"  Attempt {attempt}: 新增 {len(new_mols)} 个分子，总计 {len(all_results[target])}/{TARGET_MOLS}")

                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                _append_csv(all_results[target], target, checkpoint_dir)

                if len(new_mols) == 0 and attempt >= 20:
                    logger.warning(f"  连续 20 次无新分子，提前终止 {target}")
                    break

            except Exception as e:
                logger.error(f"  Attempt {attempt} 失败: {e}")
                traceback.print_exc()
                break

    with open(checkpoint_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open(checkpoint_dir / "summary.json", 'w') as f:
        json.dump({
            'experiment_name': 'ablation_4_no_domain_knowledge',
            'targets_processed': list(all_results.keys()),
            'per_target': {t: len(all_results.get(t, [])) for t in TARGETS},
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*60}\nAblation 4 续跑完成\n{'='*60}")
    for t, mols in all_results.items():
        logger.info(f"  {t}: {len(mols)} / {TARGET_MOLS}")
    logger.info(f"输出目录: {checkpoint_dir}")


def _append_csv(molecules, target_name, output_dir):
    csv_dir = output_dir / "csv_output"
    csv_dir.mkdir(exist_ok=True)
    csv_file = csv_dir / f"{target_name}.csv"

    existing_smiles = set()
    if csv_file.exists():
        with open(csv_file) as f:
            for line in f.readlines()[1:]:
                if line.strip():
                    parts = line.strip().split(',', 2)
                    if len(parts) >= 2:
                        existing_smiles.add(parts[1])

    lines = []
    for i, mol in enumerate(molecules):
        smiles = mol.get('smiles', '')
        if smiles not in existing_smiles:
            tpsa = mol.get('tpsa', 0)
            mw = mol.get('mw', 0)
            lines.append(f"{target_name}_{mol.get('id', i):05d},{smiles},{tpsa:.2f},{mw:.2f},{target_name}\n")
            existing_smiles.add(smiles)

    if lines:
        with open(csv_file, 'a') as f:
            f.writelines(lines)


if __name__ == "__main__":
    main()
