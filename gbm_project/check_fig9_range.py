#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

targets = ['EGFR', 'IDH1', 'VEGF']
order = [
    'ablation_1_chain_of_thought',
    'ablation_2_no_feedback',
    'ablation_3_no_physical_eval',
    'ablation_4_no_domain_knowledge',
    'ablation_5_no_gbm_lora',
    'ablation_6_base_model',
]

dc_col = 'DrugClip'

# Full Model Top-15 平均值
full_top15 = {}
df_model = pd.concat([pd.read_csv(f) for f in Path('scored_results').glob('model_test_*_ranked.csv')], ignore_index=True)
df_model['DrugClip'] = pd.to_numeric(df_model[dc_col], errors='coerce')
df_model['TotalScore'] = pd.to_numeric(df_model['TotalScore'], errors='coerce')
for t in targets:
    sub = df_model[df_model['DrugClip'] > 0]
    sub = sub[sub['target'] == t]
    valid_scores = sub['TotalScore'].dropna()
    top15 = valid_scores.nlargest(min(15, len(valid_scores)))
    full_top15[t] = top15.mean() if len(top15) > 0 else 0

# 计算下降比例
all_drops = []
for exp in order:
    drops = []
    for t in targets:
        f = Path(f'scored_results/{exp}_{t}_ranked.csv')
        if f.exists():
            df = pd.read_csv(f)
            df['DrugClip'] = pd.to_numeric(df[dc_col], errors='coerce')
            df['TotalScore'] = pd.to_numeric(df['TotalScore'], errors='coerce')
            valid = df[df['DrugClip'] > 0]
            top10 = valid.nlargest(min(10, len(valid)), 'TotalScore')
            abl_top10 = top10['TotalScore'].mean() if len(top10) > 0 else 0
            if full_top15[t] > 0 and abl_top10 > 0:
                drop = (full_top15[t] - abl_top10) / full_top15[t] * 100
            else:
                drop = 0
            drops.append(drop)
            all_drops.append(drop)
    print(f'{exp}: EGFR={drops[0]:.1f}%, IDH1={drops[1]:.1f}%, VEGF={drops[2]:.1f}%')

print(f'\nMin drop: {min(all_drops):.1f}%, Max drop: {max(all_drops):.1f}%')
print(f'Suggested xlim: ({min(all_drops)-5:.0f}, {max(all_drops)+10:.0f})')
