#!/usr/bin/env python3
"""
从 checkpoint 恢复，继续处理剩余靶点
"""
import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime
import logging

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(PROJECT_ROOT / "gbm_project" / "scripts"))
from generate_tot_molecules import TreeOfThoughtsGenerator
from generate_tot_batch_candidates import generate_molecules_for_target, load_targets


def main():
    import argparse
    parser = argparse.ArgumentParser(description='从 checkpoint 恢复生成剩余靶点')
    parser.add_argument('--checkpoint_dir', type=str,
                        default=str(PROJECT_ROOT / "gbm_project" / "experiments" / "tot_batch_20260413_030215"),
                        help='checkpoint 目录')
    parser.add_argument('--num_molecules', type=int, default=100)
    parser.add_argument('--gpu_id', type=int, default=3)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)

    # 加载已完成的靶点
    checkpoint_file = checkpoint_dir / "all_results_checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            completed = set(json.load(f).keys())
    else:
        completed = set()

    # 确定剩余靶点
    all_targets = load_targets()
    remaining = [t for t in all_targets if t not in completed]
    logger.info(f"已完成: {list(completed)}")
    logger.info(f"剩余靶点 ({len(remaining)}): {remaining}")

    if not remaining:
        logger.info("所有靶点已完成，无需恢复")
        return

    config = {
        'gpu_id': args.gpu_id,
        'base_model_path': str(PROJECT_ROOT / "models" / "Qwen2-7B-Instruct"),
        'llamole_adapter_path': str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter"),
        'gbm_adapter_path': str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter-gbm-with-graph-models"),
        'prompts_config_path': str(PROJECT_ROOT / "gbm_project" / "configs" / "english_gbm_prompts.yaml"),
        'enable_guardrail_fallback': True,
        'enable_default_strategy_fallback': True,
        'generation': {
            'max_new_tokens': 400,
            'temperature': 0.5,
            'top_p': 0.9,
            'do_sample': True
        },
        'tot_step_max_new_tokens': {
            'scaffold': 180,
            'assembly': 220,
            'smiles': 220,
            'evaluate': 30,
            'vote': 60
        },
        'tot_max_llm_calls_per_search': 50,
        'tot_max_search_seconds': 300,
        'tot_depth': 3,
        'tot_k': 3,
        'tot_b': 2,
        'constraints': {
            'mw_range': '300-500',
            'bbb_requirement': 'high',
            'logp_range': '2.0-4.0',
            'tpsa_range': '40-120'
        }
    }

    logger.info("初始化生成器...")
    generator = TreeOfThoughtsGenerator(config)

    logger.info("加载模型...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    generator.load_models(use_8bit=False)

    # 加载已完成的 results
    all_results = {}
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            all_results = json.load(f)

    summary_stats = {}
    for target in remaining:
        logger.info("=" * 80)
        logger.info(f"恢复处理靶点: {target}")
        logger.info("=" * 80)

        try:
            molecules = generate_molecules_for_target(
                generator=generator,
                target_name=target,
                num_molecules=args.num_molecules,
                k=3, b=2,
                max_attempts=200,
                max_no_new_streak=0
            )

            all_results[target] = molecules
            summary_stats[target] = {
                'total_generated': len(molecules),
                'unique_smiles': len(set(m['smiles'] for m in molecules)),
                'avg_path_length': sum(len(m.get('tot_path', [])) for m in molecules) / len(molecules) if molecules else 0
            }

            target_output_dir = checkpoint_dir / target
            target_output_dir.mkdir(exist_ok=True)
            generator.save_results(molecules, target, target_output_dir)

            checkpoint_file = checkpoint_dir / "all_results_checkpoint.json"
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

            logger.info(f"✓ 靶点 {target} 完成: {len(molecules)} 个分子")

        except Exception as e:
            logger.error(f"✗ 靶点 {target} 生成失败: {e}")
            import traceback
            traceback.print_exc()
            all_results[target] = []
            summary_stats[target] = {
                'total_generated': 0,
                'error': str(e)
            }
            continue

    # 保存最终结果
    summary_data = {
        'experiment_config': config,
        'summary_stats': summary_stats,
        'total_targets': len(all_targets),
        'total_molecules': sum(len(mols) for mols in all_results.values()),
        'targets_breakdown': {
            t: len(mols) for t, mols in all_results.items()
        }
    }

    summary_file = checkpoint_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    all_results_file = checkpoint_dir / "all_results.json"
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 80)
    logger.info("实验完成！")
    logger.info(f"输出目录: {checkpoint_dir}")
    logger.info(f"总生成分子数: {summary_data['total_molecules']}")
    for t, s in summary_stats.items():
        logger.info(f"  {t}: {s.get('total_generated', 0)} 个分子")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
