#!/usr/bin/env python3
"""
重画 fig5_bbb_pass_rate.zh.png
堆叠条形图：BBB穿透通过率（DrugClip>0分子）
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import numpy as np
import os

# ============================================================
# 中文字体设置（macOS STHeiti Medium）
# ============================================================
FONT_PATH = '/System/Library/Fonts/STHeiti Medium.ttc'
if os.path.exists(FONT_PATH):
    PROP = fm.FontProperties(fname=FONT_PATH)
    print(f'使用字体: {FONT_PATH}')
else:
    PROP = fm.FontProperties(size=12)
    print('Warning: STHeiti Medium not found')

# ============================================================
# 数据（DrugClip>0分子）
# EGFR: n=99, pass=47, 47.5%
# IDH1_IDH2: n=100, pass=45, 45.0%
# VEGF_VEGFR: n=99, pass=55, 55.6%
# ============================================================
targets   = ['EGFR', 'IDH1_IDH2', 'VEGF_VEGFR']
n_total   = [99, 100, 99]
n_pass    = [47, 45, 55]
pct_pass  = [47.5, 45.0, 55.6]
n_fail    = [n - p for n, p in zip(n_total, n_pass)]

# 颜色：绿色=BBB通过，浅灰=BBB未通过
COLOR_PASS = '#2ecc71'
COLOR_FAIL = '#d0d0d0'   # 浅灰色
COLOR_TEXT = '#2c3e50'

# ============================================================
# 图表
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

x = np.arange(len(targets))
bar_width = 0.55

# 堆叠条形
ax.bar(x, pct_pass, bar_width,
       color=COLOR_PASS, alpha=0.9,
       edgecolor='white', linewidth=0.8,
       label='BBB通过')
ax.bar(x, [100 - p for p in pct_pass], bar_width,
       bottom=pct_pass,
       color=COLOR_FAIL, alpha=0.9,
       edgecolor='white', linewidth=0.8,
       label='BBB未通过')

# 顶部百分比标签
for i, pct in enumerate(pct_pass):
    ax.text(x[i], pct + 1.8, f'{pct:.1f}%',
            ha='center', va='bottom',
            fontsize=13, fontweight='bold', color=COLOR_PASS)

# 条形内标签
for i, (np_, nf_, pct_) in enumerate(zip(n_pass, n_fail, pct_pass)):
    ax.text(x[i], np_ / 2,
            f'{np_}\n({pct_:.1f}%)',
            ha='center', va='center',
            fontsize=11, fontweight='bold',
            color='white')
    ax.text(x[i], pct_pass[i] + (100 - pct_pass[i]) / 2,
            f'{nf_}\n({100-pct_pass[i]:.1f}%)',
            ha='center', va='center',
            fontsize=10, color='#555555')

# X轴标签（中文）
ax.set_xticks(x)
label_texts = [f'{t}\nn={n}' for t, n in zip(targets, n_total)]
ax.set_xticklabels(label_texts, fontsize=13, fontweight='bold', color=COLOR_TEXT)
ax.set_xlabel('靶点', fontsize=14, color=COLOR_TEXT, labelpad=10, fontproperties=PROP)

# Y轴
ax.set_ylim(0, 108)
ax.set_ylabel('比例 (%)', fontsize=14, color=COLOR_TEXT, labelpad=10, fontproperties=PROP)
ax.yaxis.grid(True, linestyle='--', alpha=0.6, color='gray')
ax.set_axisbelow(True)
ax.tick_params(axis='y', labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 图例
legend = ax.legend(
    [mpatches.Patch(color=COLOR_PASS),
     mpatches.Patch(color=COLOR_FAIL)],
    ['BBB通过', 'BBB未通过'],
    loc='upper right',
    fontsize=12,
    frameon=True,
    framealpha=0.85,
)
for text in legend.get_texts():
    text.set_fontproperties(PROP)

# 标题（中文）
ax.set_title('不同GBM靶点生成分子的BBB渗透性分布',
             fontsize=16, fontweight='bold', color=COLOR_TEXT, pad=16,
             fontproperties=PROP)

plt.subplots_adjust(left=0.10, right=0.95, top=0.88, bottom=0.15)

out_path = '/Users/yaoyuxuan/Desktop/毕设/毕业论文/TongjiThesis-1.4.1/图片/实验结果可视化图/fig5_bbb_pass_rate.zh.png'
plt.savefig(out_path, dpi=150, facecolor='white')
plt.close()
print(f'已保存: {out_path}')
