#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

targets = ['EGFR', 'IDH1', 'VEGF']
exps = [
    'model_test',
    'ablation_1_chain_of_thought',
    'ablation_2_no_feedback',
    'ablation_3_no_physical_eval',
    'ablation_4_no_domain_knowledge',
    'ablation_5_no_gbm_lora',
    'ablation_6_base_model',
]

for exp in exps:
    print(f'=== {exp} ===')
    for t in targets:
        f = Path(f'scored_results/{exp}_{t}_ranked.csv')
        if f.exists():
            df = pd.read_csv(f)
            dc = pd.to_numeric(df['DrugClip'], errors='coerce')
            valid = df[dc > 0]
            if len(valid) >= 10:
                top10_mean = valid.nlargest(10, 'TotalScore')['TotalScore'].mean()
            elif len(valid) > 0:
                top10_mean = valid['TotalScore'].mean()
            else:
                top10_mean = 'N/A'
            print(f'  {t}: total={len(df)}, DrugClip>0={len(valid)}, Top10_mean={top10_mean}')
        else:
            print(f'  {t}: FILE NOT FOUND')
    print()
