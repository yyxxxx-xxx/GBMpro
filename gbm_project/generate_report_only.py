#!/usr/bin/env python3
"""
仅生成GBM对比分析报告的脚本
使用已保存的结果数据
"""

import os
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 设置GPU设备
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def generate_comparison_report_from_saved_data(data_file, output_dir):
    """从保存的数据生成对比分析报告"""
    print("📊 从保存数据生成对比分析报告...")

    # 加载数据
    with open(data_file, 'r', encoding='utf-8') as f:
        all_results = json.load(f)

    # 转换为DataFrame
    data = []
    for result in all_results:
        if 'evaluation' in result and result['evaluation'].get('valid', False):
            row = {
                'method': result['method'],
                'test_case_id': result['test_case_id'],
                'target': result['target'],
                'composite_score': result['evaluation']['scores']['composite_score'],
                'bbb_permeability': result['evaluation']['scores']['bbb_permeability'],
                'gbm_activity': result['evaluation']['scores']['gbm_activity'],
                'toxicity': result['evaluation']['scores']['normal_cell_toxicity'],
                'selectivity_index': result['evaluation']['scores']['selectivity_index'],
                'molecular_weight': result['evaluation']['properties']['molecular_weight'],
                'logp': result['evaluation']['properties']['logp']
            }
            data.append(row)

    if not data:
        print("❌ 没有有效的评估结果")
        return

    df = pd.DataFrame(data)

    # 计算统计信息
    stats = {}
    for method in ['baseline', 'gbm_knowledge']:
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            stats[method] = {
                'count': len(method_data),
                'avg_score': method_data['composite_score'].mean(),
                'std_score': method_data['composite_score'].std(),
                'max_score': method_data['composite_score'].max(),
                'success_rate': len(method_data) / 10  # 10个测试案例
            }

    # 生成可视化
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (16, 12)

    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Composite Score Comparison by Target
    ax = axes[0, 0]
    if len(df) > 0:
        sns.barplot(data=df, x='target', y='composite_score', hue='method', ax=ax)
        ax.set_title('Composite Score Comparison by Target', fontsize=12, fontweight='bold')
        ax.set_ylabel('Composite Score')
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=45)

    # 2. BBB Permeability Comparison
    ax = axes[0, 1]
    if len(df) > 0:
        sns.barplot(data=df, x='target', y='bbb_permeability', hue='method', ax=ax)
        ax.set_title('BBB Permeability Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel('BBB Permeability (logBB)')
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=45)

    # 3. GBM Activity Comparison
    ax = axes[0, 2]
    if len(df) > 0:
        sns.barplot(data=df, x='target', y='gbm_activity', hue='method', ax=ax)
        ax.set_title('GBM Activity Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel('GBM Activity')
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=45)

    # 4. Molecular Properties Comparison
    ax = axes[1, 0]
    properties_data = df.melt(id_vars=['method', 'target'],
                             value_vars=['molecular_weight', 'logp'],
                             var_name='property', value_name='value')
    sns.barplot(data=properties_data, x='target', y='value', hue='method', ax=ax)
    ax.set_title('Molecular Properties Comparison', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value')
    plt.setp(ax.get_xticklabels(), rotation=45)

    # 5. Radar Chart - Method Performance Comparison
    ax = axes[1, 1]
    if len(stats) >= 2:
        categories = ['Avg Score', 'BBB Permeability', 'GBM Activity', 'Selectivity']
        baseline_vals = [
            stats['baseline']['avg_score'],
            df[df['method'] == 'baseline']['bbb_permeability'].mean(),
            df[df['method'] == 'baseline']['gbm_activity'].mean(),
            df[df['method'] == 'baseline']['selectivity_index'].mean()
        ]
        gbm_vals = [
            stats['gbm_knowledge']['avg_score'],
            df[df['method'] == 'gbm_knowledge']['bbb_permeability'].mean(),
            df[df['method'] == 'gbm_knowledge']['gbm_activity'].mean(),
            df[df['method'] == 'gbm_knowledge']['selectivity_index'].mean()
        ]

        # Normalization
        all_vals = baseline_vals + gbm_vals
        min_val, max_val = min(all_vals), max(all_vals)
        if max_val > min_val:
            baseline_norm = [(v - min_val) / (max_val - min_val) for v in baseline_vals]
            gbm_norm = [(v - min_val) / (max_val - min_val) for v in gbm_vals]
        else:
            baseline_norm = [0.5] * len(baseline_vals)
            gbm_norm = [0.5] * len(gbm_vals)

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        baseline_norm += baseline_norm[:1]
        gbm_norm += gbm_norm[:1]

        ax.plot(angles, baseline_norm, 'o-', linewidth=2, label='Baseline', color='blue')
        ax.fill(angles, baseline_norm, alpha=0.25, color='blue')
        ax.plot(angles, gbm_norm, 'o-', linewidth=2, label='GBM Knowledge', color='green')
        ax.fill(angles, gbm_norm, alpha=0.25, color='green')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title('Method Performance Radar Chart', fontsize=12, fontweight='bold')
        ax.legend()

    # 6. Score Distribution Comparison
    ax = axes[1, 2]
    if len(df) > 0:
        for method in ['baseline', 'gbm_knowledge']:
            method_scores = df[df['method'] == method]['composite_score']
            if len(method_scores) > 0:
                sns.histplot(method_scores, ax=ax, label=method, alpha=0.7)
        ax.set_title('Score Distribution Comparison', fontsize=12, fontweight='bold')
        ax.set_xlabel('Composite Score')
        ax.set_ylabel('Frequency')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "gbm_comparison_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 生成文本报告
    report = f"""# GBM Baseline vs GBM Knowledge Method Comparison Analysis Report

## 📊 Test Overview
- **Test Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **GPU Device**: GPU 3 (RTX 3090)
- **Test Cases**: 10 GBM domain cases
- **Model**: Qwen2-7B-Instruct + Pre-trained Llamole LoRA adapter
- **LoRA Parameters**: 392 parameters (confirmed pre-trained weights)

## 📈 Performance Comparison

| Method | Successful Cases | Average Score | Std Dev | Max Score | Success Rate |
|--------|------------------|---------------|---------|-----------|--------------|
"""

    for method, method_stats in stats.items():
        report += f"| {method} | {method_stats['count']} | {method_stats['avg_score']:.3f} | {method_stats['std_score']:.3f} | {method_stats['max_score']:.3f} | {method_stats['success_rate']:.1f} |\n"

    if len(stats) >= 2:
        improvement = ((stats['gbm_knowledge']['avg_score'] - stats['baseline']['avg_score']) /
                      stats['baseline']['avg_score'] * 100) if stats['baseline']['avg_score'] != 0 else 0
        report += f"\n**Overall Performance Improvement**: {improvement:+.1f}%\n\n"

    report += "## 🎯 Detailed Results\n\n"

    # 按测试案例分组显示结果
    for case_id in range(1, 11):  # 10个测试案例
        case_results = [r for r in all_results if r['test_case_id'] == case_id]
        if not case_results:
            continue

        # 找到对应的测试案例信息
        test_case_info = None
        for r in case_results:
            if 'description' in r:
                test_case_info = r
                break

        if test_case_info:
            report += f"### Test Case {case_id}: {test_case_info['target']} - {test_case_info['description']}\n\n"

            # 显示两种方法的对比
            for method in ['baseline', 'gbm_knowledge']:
                method_result = next((r for r in case_results if r.get('method') == method), None)
                if method_result and 'evaluation' in method_result and method_result['evaluation'].get('valid', False):
                    eval_scores = method_result['evaluation']['scores']
                    report += f"#### {method.title()} Method:\n"
                    report += f"- **SMILES**: `{method_result['smiles']}`\n"
                    report += f"- **Composite Score**: {eval_scores['composite_score']:.3f}\n"
                    report += f"- **BBB Classification**: {eval_scores['bbb_classification']}\n"
                    report += f"- **GBM Activity**: {eval_scores['gbm_activity']:.2f}\n"
                    report += f"- **Selectivity Index**: {eval_scores['selectivity_index']:.2f}\n\n"
                else:
                    report += f"#### {method.title()} Method:\n"
                    report += "- **Status**: Failed to generate valid molecule\n\n"

            # 如果两种方法都有结果，显示对比
            baseline_result = next((r for r in case_results if r.get('method') == 'baseline' and 'evaluation' in r), None)
            gbm_result = next((r for r in case_results if r.get('method') == 'gbm_knowledge' and 'evaluation' in r), None)

            if baseline_result and gbm_result:
                baseline_score = baseline_result['evaluation']['scores']['composite_score']
                gbm_score = gbm_result['evaluation']['scores']['composite_score']
                if baseline_score != 0:
                    improvement = (gbm_score - baseline_score) / baseline_score * 100
                    report += f"**Comparison**: GBM Knowledge vs Baseline = {gbm_score:.3f} vs {baseline_score:.3f} ({improvement:+.1f}%)\n\n"
                else:
                    report += f"**Comparison**: GBM Knowledge vs Baseline = {gbm_score:.3f} vs {baseline_score:.3f} (baseline score is 0)\n\n"
            else:
                report += "\n"

    report += "## 📁 Output Files\n\n"
    report += "- `gbm_comparison_results.json`: Complete test results\n"
    report += "- `gbm_comparison_visualization.png`: Visualization comparison charts\n"
    report += "- `gbm_comparison_report.md`: This detailed report\n\n"
    report += "- `gbm_comparison_data.csv`: CSV format data\n\n"

    # 保存报告
    with open(output_dir / "gbm_comparison_report.md", 'w', encoding='utf-8') as f:
        f.write(report)

    # 保存数据
    with open(output_dir / "gbm_comparison_results.json", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 保存CSV数据
    if data:
        df.to_csv(output_dir / "gbm_comparison_data.csv", index=False)

    print(f"✓ 对比分析报告已生成: {output_dir}")
    print(f"  - 包含 {len([r for r in all_results if 'evaluation' in r])} 个成功生成的分子")
    print(f"  - 可视化图表: gbm_comparison_visualization.png")
    print(f"  - 详细报告: gbm_comparison_report.md")

def main():
    """主函数"""
    # 使用有完整数据的目录
    results_dir = Path("experiments/final_gbm_comparison_corrected_20260112_144739")
    data_file = results_dir / "gbm_comparison_results.json"

    if not data_file.exists():
        print(f"❌ 找不到结果文件: {data_file}")
        return

    print("🧬 生成GBM对比分析报告")
    print("=" * 60)

    # 确保输出目录存在
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        generate_comparison_report_from_saved_data(data_file, results_dir)
        print("\n" + "=" * 60)
        print("🎉 报告生成完成！")
        print("=" * 60)
        print(f"📁 结果已保存至: {results_dir}")

    except Exception as e:
        print(f"❌ 报告生成失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
