#!/usr/bin/env python3
"""
GBM实验结果可视化脚本
创建综合的可视化图表展示GBM候选分子的性能表现
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

def load_experiment_data(results_file: str) -> pd.DataFrame:
    """加载实验结果数据"""
    with open(results_file, 'r', encoding='utf-8') as f:
        molecules = json.load(f)

    # 转换为DataFrame
    data = []
    for mol in molecules:
        if mol.get('evaluation', {}).get('valid', False):
            row = {
                'id': mol['id'],
                'target': mol['target'],
                'smiles': mol['smiles'],
                'composite_score': mol['evaluation']['scores']['composite_score'],
                'bbb_permeability': mol['evaluation']['scores']['bbb_permeability'],
                'gbm_activity': mol['evaluation']['scores']['gbm_activity'],
                'toxicity': mol['evaluation']['scores']['normal_cell_toxicity'],
                'selectivity_index': mol['evaluation']['scores']['selectivity_index'],
                'synthetic_accessibility': mol['evaluation']['scores']['synthetic_accessibility'],
                'clinical_similarity': mol['evaluation']['scores']['clinical_similarity'],
                'bbb_classification': mol['evaluation']['scores']['bbb_classification'],
                'molecular_weight': mol['evaluation']['properties']['molecular_weight'],
                'logp': mol['evaluation']['properties']['logp'],
                'tpsa': mol['evaluation']['properties']['tpsa'],
                'hbd': mol['evaluation']['properties']['hbd'],
                'hba': mol['evaluation']['properties']['hba'],
                'rotatable_bonds': mol['evaluation']['properties']['rotatable_bonds'],
                'ring_count': mol['evaluation']['properties']['ring_count'],
                'qed': mol['evaluation']['properties']['qed']
            }
            data.append(row)

    return pd.DataFrame(data)

def create_comprehensive_visualization(df: pd.DataFrame, output_dir: Path):
    """创建综合的可视化图表"""
    # 设置matplotlib参数
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    sns.set_style("whitegrid")

    # 创建子图布局
    fig = plt.figure(figsize=(20, 16))

    # 1. 综合得分分布 (左上)
    ax1 = plt.subplot(3, 4, 1)
    sns.histplot(df['composite_score'], bins=10, kde=True, color='skyblue', ax=ax1)
    ax1.set_title('GBM Candidate Molecules\nComposite Score Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Composite Score')
    ax1.set_ylabel('Frequency')
    ax1.axvline(df['composite_score'].mean(), color='red', linestyle='--', linewidth=2,
               label='.3f')
    ax1.legend()

    # 2. BBB穿透性分类饼图 (右上)
    ax2 = plt.subplot(3, 4, 2)
    bbb_counts = df['bbb_classification'].value_counts()
    colors = ['red' if x == 'low' else 'orange' if x == 'medium' else 'green' for x in bbb_counts.index]
    ax2.pie(bbb_counts.values, labels=bbb_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('BBB Permeability\nClassification', fontsize=12, fontweight='bold')

    # 3. 多指标雷达图 (中间左)
    ax3 = plt.subplot(3, 4, 3, projection='polar')
    metrics = ['composite_score', 'gbm_activity', 'selectivity_index', 'synthetic_accessibility']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    for idx, (_, row) in enumerate(df.iterrows()):
        values = [row[metric] for metric in metrics]
        # 归一化到0-1范围
        values = [(v - df[metric].min()) / (df[metric].max() - df[metric].min()) for v, metric in zip(values, metrics)]
        values += values[:1]

        ax3.plot(angles, values, 'o-', linewidth=2, label=f'Molecule {row["id"]}')
        ax3.fill(angles, values, alpha=0.1)

    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax3.set_ylim(0, 1)
    ax3.set_title('Multi-metric Performance\nComparison', fontsize=12, fontweight='bold', pad=20)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # 4. 分子性质散点图矩阵 (中间右 + 底部)
    properties = ['molecular_weight', 'logp', 'tpsa', 'qed']
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))

    # 分子量 vs LogP
    ax4 = plt.subplot(3, 4, 4)
    scatter = ax4.scatter(df['molecular_weight'], df['logp'],
                         c=df['composite_score'], cmap='viridis', s=100, alpha=0.7)
    ax4.set_xlabel('Molecular Weight (Da)')
    ax4.set_ylabel('LogP')
    ax4.set_title('MW vs LogP\n(Color: Composite Score)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax4, shrink=0.8)

    # 5. 综合得分 vs BBB穿透性
    ax5 = plt.subplot(3, 4, 5)
    sns.scatterplot(data=df, x='bbb_permeability', y='composite_score',
                   hue='bbb_classification', palette='RdYlGn_r', s=100, ax=ax5)
    ax5.set_xlabel('BBB Permeability (logBB)')
    ax5.set_ylabel('Composite Score')
    ax5.set_title('BBB Permeability vs\nComposite Score', fontsize=12, fontweight='bold')
    ax5.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='High Potential Threshold')
    ax5.legend()

    # 6. 活性 vs 毒性散点图
    ax6 = plt.subplot(3, 4, 6)
    sns.scatterplot(data=df, x='toxicity', y='gbm_activity', s=100, ax=ax6)
    ax6.set_xlabel('Normal Cell Toxicity (pIC50)')
    ax6.set_ylabel('GBM Activity (pIC50)')
    ax6.set_title('GBM Activity vs\nNormal Cell Toxicity', fontsize=12, fontweight='bold')
    ax6.plot([df['toxicity'].min(), df['toxicity'].max()],
            [df['toxicity'].min(), df['toxicity'].max()],
            'r--', alpha=0.7, label='Toxicity = Activity')
    ax6.legend()

    # 7. 分子性质箱线图
    ax7 = plt.subplot(3, 4, 7)
    props_to_plot = ['molecular_weight', 'logp', 'tpsa', 'qed']
    df_melted = df[props_to_plot].melt(var_name='Property', value_name='Value')
    sns.boxplot(data=df_melted, x='Property', y='Value', ax=ax7, palette='Set2')
    ax7.set_title('Molecular Properties\nDistribution', fontsize=12, fontweight='bold')
    ax7.set_xticklabels([p.replace('_', ' ').title() for p in props_to_plot], rotation=45)

    # 8. 合成可行性 vs 临床相似度
    ax8 = plt.subplot(3, 4, 8)
    sns.scatterplot(data=df, x='synthetic_accessibility', y='clinical_similarity',
                   size='composite_score', sizes=(50, 200), ax=ax8)
    ax8.set_xlabel('Synthetic Accessibility')
    ax8.set_ylabel('Clinical Similarity')
    ax8.set_title('Synthetic Accessibility vs\nClinical Similarity', fontsize=12, fontweight='bold')

    # 9. 性能指标汇总表 (底部大图)
    ax9 = plt.subplot(3, 4, (9, 12))
    ax9.axis('off')

    # 创建性能汇总表格
    metrics_summary = {
        'Total Molecules': len(df),
        'Average Composite Score': '.3f',
        'BBB High': f"{(df['bbb_classification'] == 'high').sum()} ({(df['bbb_classification'] == 'high').mean()*100:.1f}%)",
        'BBB Medium': f"{(df['bbb_classification'] == 'medium').sum()} ({(df['bbb_classification'] == 'medium').mean()*100:.1f}%)",
        'BBB Low': f"{(df['bbb_classification'] == 'low').sum()} ({(df['bbb_classification'] == 'low').mean()*100:.1f}%)",
        'High Potential (>0.7)': f"{(df['composite_score'] > 0.7).sum()} ({(df['composite_score'] > 0.7).mean()*100:.1f}%)",
        'Best Composite Score': '.3f',
        'Average GBM Activity': '.2f',
        'Average Selectivity Index': '.2f',
        'Average Synthetic Accessibility': '.1f'
    }

    # 创建表格
    table_data = [[k, v] for k, v in metrics_summary.items()]
    table = ax9.table(cellText=table_data, colLabels=['Metric', 'Value'],
                     loc='center', cellLoc='left', colLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    ax9.set_title('GBM Experiment Performance Summary', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / "gbm_experiment_comprehensive_visualization.png",
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Comprehensive visualization saved to: {output_dir / 'gbm_experiment_comprehensive_visualization.png'}")

def create_detailed_metrics_plots(df: pd.DataFrame, output_dir: Path):
    """创建详细的指标分析图表"""

    # 设置样式
    plt.rcParams['figure.figsize'] = (15, 10)
    sns.set_style("whitegrid")

    # 1. 各项得分分布对比
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    score_columns = ['composite_score', 'bbb_permeability', 'gbm_activity',
                    'toxicity', 'synthetic_accessibility', 'clinical_similarity']

    for i, col in enumerate(score_columns):
        if i < len(axes):
            sns.histplot(df[col], bins=8, kde=True, ax=axes[i], color='steelblue')
            axes[i].set_title(f'{col.replace("_", " ").title()} Distribution', fontweight='bold')
            axes[i].set_xlabel(col.replace("_", " ").title())
            axes[i].set_ylabel('Frequency')

            # 添加统计信息
            mean_val = df[col].mean()
            axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=1.5,
                          label='.3f')
            axes[i].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "detailed_metrics_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 分子性质相关性热力图
    plt.figure(figsize=(12, 10))
    property_cols = ['molecular_weight', 'logp', 'tpsa', 'hbd', 'hba', 'rotatable_bonds', 'ring_count', 'qed']
    score_cols = ['composite_score', 'gbm_activity', 'toxicity', 'bbb_permeability']

    # 计算相关性
    corr_data = df[property_cols + score_cols].corr()

    # 只显示与得分相关的相关性
    score_corr = corr_data.loc[property_cols, score_cols]

    sns.heatmap(score_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
    plt.title('Molecular Properties vs Performance Metrics Correlation', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "properties_performance_correlation.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. BBB分类与性能的关系
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # BBB分类 vs 综合得分
    bbb_order = ['low', 'medium', 'high'] if 'high' in df['bbb_classification'].values else ['low', 'medium']
    sns.boxplot(data=df, x='bbb_classification', y='composite_score',
               order=bbb_order, ax=axes[0], palette='RdYlGn')
    axes[0].set_title('BBB Classification vs Composite Score', fontweight='bold')
    axes[0].set_xlabel('BBB Permeability')
    axes[0].set_ylabel('Composite Score')

    # BBB分类 vs GBM活性
    sns.boxplot(data=df, x='bbb_classification', y='gbm_activity',
               order=bbb_order, ax=axes[1], palette='RdYlGn')
    axes[1].set_title('BBB Classification vs GBM Activity', fontweight='bold')
    axes[1].set_xlabel('BBB Permeability')
    axes[1].set_ylabel('GBM Activity (pIC50)')

    # BBB分类 vs 合成可行性
    sns.boxplot(data=df, x='bbb_classification', y='synthetic_accessibility',
               order=bbb_order, ax=axes[2], palette='RdYlGn')
    axes[2].set_title('BBB Classification vs Synthetic Accessibility', fontweight='bold')
    axes[2].set_xlabel('BBB Permeability')
    axes[2].set_ylabel('Synthetic Accessibility')

    plt.tight_layout()
    plt.savefig(output_dir / "bbb_performance_relationship.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_experiment_summary_report(df: pd.DataFrame, output_dir: Path):
    """创建实验总结报告"""

    # 计算关键统计信息
    summary_stats = {
        'experiment_info': {
            'total_molecules': len(df),
            'target': df['target'].iloc[0] if len(df) > 0 else 'N/A',
            'date': '2026-01-11',
            'method': 'GBM-enhanced Llamole with CoT reasoning'
        },
        'performance_metrics': {
            'composite_score': {
                'mean': df['composite_score'].mean(),
                'std': df['composite_score'].std(),
                'min': df['composite_score'].min(),
                'max': df['composite_score'].max(),
                'median': df['composite_score'].median()
            },
            'gbm_activity': {
                'mean': df['gbm_activity'].mean(),
                'std': df['gbm_activity'].std(),
                'range': f"{df['gbm_activity'].min():.2f} - {df['gbm_activity'].max():.2f}"
            },
            'bbb_distribution': {
                'high': int((df['bbb_classification'] == 'high').sum()),
                'medium': int((df['bbb_classification'] == 'medium').sum()),
                'low': int((df['bbb_classification'] == 'low').sum()),
                'high_percentage': (df['bbb_classification'] == 'high').mean() * 100,
                'low_percentage': (df['bbb_classification'] == 'low').mean() * 100
            },
            'molecular_properties': {
                'avg_molecular_weight': df['molecular_weight'].mean(),
                'avg_logp': df['logp'].mean(),
                'avg_tpsa': df['tpsa'].mean(),
                'avg_qed': df['qed'].mean()
            }
        },
        'top_candidates': []
    }

    # 找出顶级候选分子
    top_3 = df.nlargest(3, 'composite_score')
    for _, row in top_3.iterrows():
        candidate = {
            'id': int(row['id']),
            'composite_score': row['composite_score'],
            'bbb_classification': row['bbb_classification'],
            'gbm_activity': row['gbm_activity'],
            'selectivity_index': row['selectivity_index'],
            'molecular_weight': row['molecular_weight'],
            'smiles': row['smiles'][:50] + '...' if len(row['smiles']) > 50 else row['smiles']
        }
        summary_stats['top_candidates'].append(candidate)

    # 保存详细报告
    with open(output_dir / "experiment_summary.json", 'w', encoding='utf-8') as f:
        # 处理numpy类型
        def convert_types(obj):
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        json.dump(convert_types(summary_stats), f, indent=2, ensure_ascii=False)

    # 创建可读性强的文本报告
    with open(output_dir / "experiment_summary_report.txt", 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("GBM EXPERIMENT SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")

        # 实验基本信息
        f.write("EXPERIMENT OVERVIEW\n")
        f.write("-" * 30 + "\n")
        for key, value in summary_stats['experiment_info'].items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")

        # 性能指标
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 30 + "\n")

        comp_score = summary_stats['performance_metrics']['composite_score']
        f.write(f"Composite Score:\n")
        f.write(f"  Mean: {comp_score['mean']:.3f} ± {comp_score['std']:.3f}\n")
        f.write(f"  Range: [{comp_score['min']:.3f}, {comp_score['max']:.3f}]\n")
        f.write(f"  Median: {comp_score['median']:.3f}\n\n")

        gbm_act = summary_stats['performance_metrics']['gbm_activity']
        f.write(f"GBM Activity: {gbm_act['mean']:.2f} ± {gbm_act['std']:.2f} {gbm_act['range']}\n\n")

        bbb_dist = summary_stats['performance_metrics']['bbb_distribution']
        f.write(f"BBB Permeability Distribution:\n")
        f.write(f"  High: {bbb_dist['high']} ({bbb_dist['high_percentage']:.1f}%)\n")
        f.write(f"  Medium: {bbb_dist['medium']}\n")
        f.write(f"  Low: {bbb_dist['low']} ({bbb_dist['low_percentage']:.1f}%)\n\n")

        mol_props = summary_stats['performance_metrics']['molecular_properties']
        f.write(f"Molecular Properties (Average):\n")
        f.write(f"  Molecular Weight: {mol_props['avg_molecular_weight']:.1f} Da\n")
        f.write(f"  LogP: {mol_props['avg_logp']:.2f}\n")
        f.write(f"  TPSA: {mol_props['avg_tpsa']:.1f}\n")
        f.write(f"  QED: {mol_props['avg_qed']:.3f}\n\n")

        # 顶级候选分子
        f.write("TOP CANDIDATE MOLECULES\n")
        f.write("-" * 30 + "\n")
        for i, candidate in enumerate(summary_stats['top_candidates'], 1):
            f.write(f"Molecule {i}:\n")
            f.write(f"  Composite Score: {candidate['composite_score']:.3f}\n")
            f.write(f"  BBB Classification: {candidate['bbb_classification']}\n")
            f.write(f"  GBM Activity: {candidate['gbm_activity']:.2f}\n")
            f.write(f"  Selectivity Index: {candidate['selectivity_index']:.2f}\n")
            f.write(f"  Molecular Weight: {candidate['molecular_weight']:.1f} Da\n")
            f.write(f"  SMILES: {candidate['smiles']}\n\n")

        f.write("=" * 80 + "\n")

def main():
    """主函数"""
    # 实验结果文件路径
    results_file = "experiments/test_results/gbm_gpu_test/generated_molecules.json"

    # 输出目录
    output_dir = Path("experiments/test_results/gbm_gpu_test/visualization")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading experiment results...")
    df = load_experiment_data(results_file)

    if len(df) == 0:
        print("No valid molecules found in results!")
        return

    print(f"Loaded {len(df)} valid molecules")
    print("Creating comprehensive visualizations...")

    # 创建综合可视化
    create_comprehensive_visualization(df, output_dir)

    # 创建详细指标图表
    create_detailed_metrics_plots(df, output_dir)

    # 创建实验总结报告
    create_experiment_summary_report(df, output_dir)

    print(f"\nVisualization complete! Results saved to: {output_dir}")
    print("\nGenerated files:")
    for file in output_dir.glob("*.png"):
        print(f"  - {file.name}")
    for file in output_dir.glob("*.json"):
        print(f"  - {file.name}")
    for file in output_dir.glob("*.txt"):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()
