#!/usr/bin/env python3
"""
续跑 Ablation 2 (No Feedback) 实验，只处理 checkpoint 中分子数不足 50 的靶点。
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
GPU_ID = 1


def main():
    parser = argparse.ArgumentParser(description="续跑 Ablation 2 No Feedback 实验")
    parser.add_argument('--checkpoint_dir', type=str,
                        default=str(PROJECT_ROOT / "gbm_project" / "experiments" / "ablation_2_no_feedback_20260423_144034"))
    parser.add_argument('--num_molecules', type=int, default=TARGET_MOLS)
    parser.add_argument('--gpu_id', type=int, default=GPU_ID)
    parser.add_argument('--max_attempts', type=int, default=300)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_file = checkpoint_dir / "all_results_checkpoint.json"

    # 读取已完成的靶点
    all_results = {}
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            all_results = json.load(f)
        logger.info(f"已加载 checkpoint，现有结果: {[(t, len(m)) for t, m in all_results.items()]}")

    # 确定剩余靶点（分子数 < TARGET_MOLS）
    remaining = [t for t in TARGETS if len(all_results.get(t, [])) < TARGET_MOLS]
    logger.info(f"需继续生成的靶点 ({len(remaining)}): {remaining}")

    if not remaining:
        logger.info("所有靶点均已满足 50 个分子，无需续跑")
        return

    # 构建配置
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
    config['tot_refinement_rounds'] = 1  # Ablation 2 强制 1 轮

    logger.info("初始化生成器...")
    generator = TreeOfThoughtsGenerator(config)

    logger.info("加载模型...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    generator.load_models(use_8bit=False)

    # 模拟 NoFeedbackGenerator 的 patch
    from generate_tot_molecules import TreeOfThoughtsGenerator as ToT
    original_generate_smiles = generator._generate_smiles_fallback
    def patched_generate_smiles(finetuned_prompt, *, scaffold, assembly, warhead,
                                 bbb_enhancers, target_mw, physical_feedback=None, max_new_tokens=450):
        return original_generate_smiles(
            finetuned_prompt, scaffold=scaffold, assembly=assembly, warhead=warhead,
            bbb_enhancers=bbb_enhancers, target_mw=target_mw,
            physical_feedback=None, max_new_tokens=max_new_tokens
        )
    generator._generate_smiles_fallback = patched_generate_smiles
    logger.info("[Ablation 2] Physical feedback loop DISABLED (always None)")

    # 生成缺失靶点：循环直到达到目标数量
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

                # 立即写 checkpoint 和 CSV
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                _append_csv(all_results[target], target, checkpoint_dir)

                # 如果连续 3 次都没新增新分子，提前退出
                if len(new_mols) == 0 and attempt >= 20:
                    logger.warning(f"  连续 20 次无新分子，提前终止 {target}")
                    break

            except Exception as e:
                logger.error(f"  Attempt {attempt} 失败: {e}")
                traceback.print_exc()
                break

    # 写最终结果
    with open(checkpoint_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open(checkpoint_dir / "summary.json", 'w') as f:
        json.dump({
            'experiment_name': 'ablation_2_no_feedback',
            'targets_processed': list(all_results.keys()),
            'per_target': {t: len(all_results.get(t, [])) for t in TARGETS},
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*60}\nAblation 2 续跑完成\n{'='*60}")
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
