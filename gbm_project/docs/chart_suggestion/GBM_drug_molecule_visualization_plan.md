# GBM 靶点药物分子生成结果数据可视化绘制思路

## 1. 文档目的

本文档用于整理针对不同 GBM 靶点生成药物分子结果的数据可视化方案。当前数据包括三个靶点 EGFR、IDH1、VEGF 的生成分子性质指标，以及六组消融实验结果。可视化目标不是单纯展示指标，而是围绕以下核心问题构建分析逻辑：

1. 生成模型是否能够在不同 GBM 靶点上产生高质量候选分子？
2. 高质量分子的 DrugClip、Vina Dock、BBB、理化性质和结构警报表现是否合理？
3. 不同模块或策略在消融实验中分别贡献了哪些能力？
4. 最终筛选出的候选分子是否具有较好的成药潜力和结构多样性？

本文档中的图表标题、坐标轴、图例和注释建议使用中文；DrugClip、Vina Dock、BBB、PAINS、Brenk、Synthetic Accessibility、UMAP、t-SNE、Top-K 等专业术语可保留英文。

---

## 2. 数据组织建议

在绘图前，统计有药效（评分不为0）的分子占比，并画出表格（行为不同的实验，列为不同的靶点），并剔除评分为0的分子，然后将不同靶点和消融实验的数据合并为统一格式，方便后续批量绘图和分组统计。

### 2.1 主实验数据字段

建议统一字段名如下：

| 字段名 | 含义 | 说明 |
|---|---|---|
| `target` | 靶点 | EGFR、IDH1、VEGF |
| `smiles` | 分子 SMILES | 用于结构图、fingerprint 和 scaffold 分析 |
| `drugclip` | DrugClip Score | 越高越好，若小于 0 则直接判为低可信分子 |
| `vina` | Vina Dock Score | 通常越负越好 |
| `bbb` | BBB permeant | 是否可通过血脑屏障 |
| `mw` | Molecular Weight | 分子量 |
| `tpsa` | TPSA | 极性表面积，影响 BBB 和透膜性 |
| `logp` | Consensus LogP | 疏水性指标 |
| `pgp` | Pgp substrate | 是否可能被 P-gp 外排 |
| `pains` | PAINS Alerts | 结构警报数量 |
| `brenk` | Brenk Alerts | 结构警报数量 |
| `sa` | Synthetic Accessibility | 合成可及性，通常越低越容易合成 |
| `score` | 综合 Score | 根据打分函数计算得到 |

### 2.2 消融实验数据字段

对于消融实验，建议增加字段：

| 字段名 | 含义 |
|---|---|
| `experiment` | 实验设置 |
| `target` | 靶点 |
| `score` | 综合分数 |
| `drugclip` | DrugClip Score |
| `vina` | Vina Dock Score |
| `bbb` | BBB permeant |
| `mw` / `tpsa` / `logp` | 理化性质 |
| `pains` / `brenk` | 结构警报 |
| `sa` | 合成可及性 |

建议将六组消融实验命名为：

| 文件夹名 | 图中显示名称 |
|---|---|
| `ablation_1_chain_of_thought` | 去除 Chain-of-Thought |
| `ablation_2_no_feedback` | 去除 Feedback |
| `ablation_3_no_physical_eval` | 去除 Physical Eval |
| `ablation_4_no_in_knowledge` | 去除 In-domain Knowledge |
| `ablation_5_no_gbm_lora` | 去除 GBM LoRA |
| `ablation_6_base_model` | Base Model |

若有完整模型结果，建议统一命名为：

```text
Full Model
```

---

## 3. 主结果可视化：整体生成质量

### 3.1 三靶点综合 Score 分布图

**推荐图形：**箱线图、violin plot、箱线图叠加散点图  
**图表标题：**不同 GBM 靶点生成分子的综合 Score 分布  
**横轴：**靶点  
**纵轴：**综合 Score  
**分组：**EGFR、IDH1、VEGF

**绘制目的：**  
展示不同靶点下生成分子的总体质量差异。

**分析重点：**

- 哪个靶点的中位数综合 Score 更高？
- 哪个靶点分布更集中，说明生成稳定性更好？
- 哪个靶点高分分子更多？
- 是否存在明显低分或异常离群分子？

**推荐图注：**  
“箱线图展示 EGFR、IDH1 和 VEGF 三个靶点下生成分子的综合评分分布，综合 Score 越高表示分子在靶点相关性、docking、BBB 和药物样性质等方面表现越优。”

---

### 3.2 Top-K 分子比例柱状图

**推荐图形：**堆叠柱状图、百分比堆叠柱状图  
**图表标题：**不同靶点下高分候选分子的比例  
**横轴：**靶点  
**纵轴：**分子比例  
**颜色分组：**Score ≥ 80、70 ≤ Score < 80、60 ≤ Score < 70、Score < 60

**绘制目的：**  
评估不同靶点下有多少分子值得进入下一轮筛选。

**推荐分层：**

| 综合 Score 区间 | 筛选建议 |
|---|---|
| Score ≥ 80 | 高优先级候选 |
| 70 ≤ Score < 80 | 建议进一步验证 |
| 60 ≤ Score < 70 | 保留观察 |
| Score < 60 | 优先淘汰 |

**分析重点：**

- 哪个靶点高优先级分子比例最高？
- 哪个靶点主要集中在中低分区间？
- 生成模型是否能够稳定地产生可用候选分子？

---

### 3.3 每个靶点 Top 10 分子横向条形图

**推荐图形：**横向 bar chart、facet bar chart  
**图表标题：**各靶点 Top 10 候选分子综合 Score 排名  
**横轴：**综合 Score  
**纵轴：**分子 ID 或 SMILES 简写  
**颜色：**BBB permeant 状态或 DrugClip 是否大于 0

**绘制目的：**  
直接展示每个靶点下最值得关注的候选分子。

**建议标注信息：**

- DrugClip Score
- Vina Dock Score
- BBB permeant
- PAINS/Brenk Alerts
- Synthetic Accessibility

**分析重点：**

- Top 分子是否同时具备较高 DrugClip 和较好 Vina Dock？
- 高分分子是否大多 BBB permeant？
- Top 分子是否存在明显结构警报或合成难度问题？

---

## 4. 核心指标可视化：DrugClip、Vina Dock 和 BBB

### 4.1 DrugClip Score 分布图

**推荐图形：**直方图、KDE 密度曲线、violin plot  
**图表标题：**不同靶点生成分子的 DrugClip Score 分布  
**横轴：**DrugClip Score  
**纵轴：**分子数量或密度  
**颜色：**靶点

**关键绘图要求：**  
在 DrugClip Score = 0 的位置画一条竖线，表示硬性筛选阈值。

**绘制目的：**  
展示生成分子与目标靶点或药物语义空间的匹配程度。

**分析重点：**

- 哪个靶点的 DrugClip Score 整体更高？
- 各靶点 DrugClip < 0 的比例是多少？
- DrugClip 负值比例是否提示该靶点生成难度较高？

**推荐图注：**  
“DrugClip Score 小于 0 的分子被认为缺乏可靠的靶点相关性，在后续综合打分中直接赋 0 分。”

---

### 4.2 Vina Dock Score 分布图

**推荐图形：**箱线图、violin plot、ridge plot  
**图表标题：**不同靶点生成分子的 Vina Dock Score 分布  
**横轴：**靶点  
**纵轴：**Vina Dock Score

**绘制目的：**  
展示不同靶点生成分子的 docking 亲和力表现。

**注意事项：**  
Vina Dock Score 通常越负表示结合越好。因此绘图时可以采用两种方式：

1. 直接绘制原始 Vina Dock Score，图中说明“越低越好”。
2. 将 Vina Dock Score 转换为 Docking Affinity Score，使其变成“越高越好”。

推荐在综合对比图中使用转换后的 Docking Affinity Score，避免解释混乱。

**分析重点：**

- 哪个靶点的 docking 表现更强？
- 是否存在极端异常 docking 值？
- 高 docking 分子是否同时具有较好的 DrugClip？

---

### 4.3 DrugClip vs Vina Dock 二维散点图

**推荐图形：**scatter plot、density scatter、hexbin plot  
**图表标题：**DrugClip Score 与 Vina Dock Score 的关系  
**横轴：**DrugClip Score  
**纵轴：**Vina Dock Score 或 Docking Affinity Score  
**颜色：**靶点  
**点大小：**综合 Score  
**点形状：**BBB permeant 状态

**绘制目的：**  
同时观察两个最重要指标之间的关系，识别真正有价值的候选分子。

**建议划分区域：**

| 区域 | 含义 |
|---|---|
| 高 DrugClip + 好 docking | 最优候选分子 |
| 高 DrugClip + 差 docking | 靶点相关性较好，但结合力不足 |
| 低 DrugClip + 好 docking | docking 偶然较好，可信度需谨慎 |
| 低 DrugClip + 差 docking | 优先淘汰 |

**分析重点：**

- 高分分子是否集中在高 DrugClip 和好 docking 区域？
- 不同靶点是否呈现不同分布模式？
- BBB permeant 分子是否更集中在理想区域？

---

### 4.4 BBB 通过率柱状图

**推荐图形：**分组柱状图、百分比堆叠柱状图  
**图表标题：**不同靶点生成分子的 BBB permeant 比例  
**横轴：**靶点  
**纵轴：**分子数量或比例  
**颜色：**BBB permeant / non-BBB permeant

**绘制目的：**  
强调 GBM 药物设计中血脑屏障穿透能力的重要性。

**分析重点：**

- 哪个靶点下 BBB permeant 分子比例更高？
- 高 DrugClip 和好 docking 分子中 BBB 通过率是否仍然较高？
- 是否存在靶点相关性好但 BBB 不通过的分子，需要降低优先级？

---

### 4.5 BBB 分组下的综合 Score 分布

**推荐图形：**boxplot、violin plot  
**图表标题：**BBB permeant 状态下的综合 Score 分布  
**横轴：**BBB permeant  
**纵轴：**综合 Score  
**颜色或分面：**靶点

**绘制目的：**  
分析 BBB permeant 是否与综合 Score 提升相关。

**分析重点：**

- BBB permeant 分子的综合 Score 是否整体更高？
- BBB 不通过但综合 Score 高的分子是否过多？
- 是否需要进一步提高 BBB 权重或设置更强惩罚？

---

## 5. 药物样性质可视化

### 5.1 MW、TPSA、LogP 分布图

**推荐图形：**箱线图、直方图、KDE 曲线  
**图表标题：**不同靶点生成分子的理化性质分布  
**横轴：**靶点或指标值  
**纵轴：**指标值、分子数量或密度  
**指标：**MW、TPSA、Consensus LogP

**绘制目的：**  
评估生成分子是否位于合理的药物样性质空间。

**分析重点：**

- 分子量是否整体偏大？
- TPSA 是否过高，从而不利于 BBB 穿透？
- LogP 是否过高或过低，影响溶解性和膜通透性？

**建议图中加入参考区间：**

- MW：可参考 150–500
- TPSA：中枢药物通常偏低更有利于 BBB
- LogP：适中更优，过高可能带来溶解性或非特异结合问题

---

### 5.2 TPSA vs LogP 散点图

**推荐图形：**scatter plot  
**图表标题：**TPSA 与 LogP 的 BBB 相关理化空间分布  
**横轴：**TPSA  
**纵轴：**Consensus LogP  
**颜色：**BBB permeant  
**点大小：**综合 Score  
**分面：**靶点

**绘制目的：**  
展示生成分子是否落在更有利于 BBB 穿透的理化性质区域。

**分析重点：**

- BBB permeant 分子是否集中在低 TPSA 和适中 LogP 区域？
- 高分分子是否同时具有合理 TPSA 和 LogP？
- 哪个靶点生成的分子更偏离 BBB-friendly 区域？

---

### 5.3 MW vs TPSA / MW vs LogP 散点图

**推荐图形：**scatter plot、facet scatter plot  
**图表标题：**分子量与极性/疏水性关系分析  
**横轴：**MW  
**纵轴：**TPSA 或 Consensus LogP  
**颜色：**靶点  
**点大小：**综合 Score  
**点形状：**BBB permeant

**绘制目的：**  
判断生成模型是否倾向产生过大、过极性或过疏水的分子。

**分析重点：**

- docking 表现好的分子是否同时 MW 偏大？
- 高分分子是否位于合理理化性质范围？
- 不同靶点生成分子的物化空间是否存在差异？

---

### 5.4 PAINS / Brenk 结构警报统计图

**推荐图形：**柱状图、堆叠柱状图、热图  
**图表标题：**不同靶点生成分子的结构警报统计  
**横轴：**靶点  
**纵轴：**分子数量或比例  
**颜色：**无警报、PAINS Alerts、Brenk Alerts、两者均有

**绘制目的：**  
评估生成分子中潜在假阳性、反应性或不稳定结构的比例。

**分析重点：**

- 哪个靶点的结构警报比例更高？
- 高分候选分子是否仍存在较多 PAINS 或 Brenk Alerts？
- 结构警报是否应作为后续筛选的强惩罚项？

---

### 5.5 Synthetic Accessibility 分布图

**推荐图形：**箱线图、直方图、violin plot  
**图表标题：**不同靶点生成分子的 Synthetic Accessibility 分布  
**横轴：**靶点  
**纵轴：**Synthetic Accessibility

**绘制目的：**  
评估生成分子的实际合成可行性。

**分析重点：**

- 高分分子是否合成难度过高？
- 哪个靶点生成分子的 SA 分布更合理？
- 是否存在 docking 好但合成不可行的候选分子？

---

## 6. 多指标综合可视化

### 6.1 多指标相关性热图

**推荐图形：**correlation heatmap  
**图表标题：**生成分子各评价指标的相关性分析  
**指标：**DrugClip Score、Vina Dock Score、Docking Affinity Score、MW、TPSA、LogP、BBB、Pgp、PAINS、Brenk、SA、综合 Score

**绘制目的：**  
分析不同指标之间的关系，并验证综合打分函数是否过度依赖单一指标。

**数据处理建议：**

- BBB permeant：Yes = 1，No = 0
- Pgp substrate：Yes = 1，No = 0
- Vina Dock Score 可转换为 Docking Affinity Score
- PAINS/Brenk 可使用警报数量，或转换为是否有警报

**分析重点：**

- DrugClip 与 Vina Dock 是否相关？
- Vina Dock 是否与 MW 相关，是否存在“大分子 docking 更好”的偏差？
- BBB 是否与 TPSA、LogP 有明显关系？
- 综合 Score 是否主要由 DrugClip 和 Vina Dock 驱动，同时兼顾 BBB 和药物样性质？

---

### 6.2 多指标雷达图

**推荐图形：**radar chart  
**图表标题：**不同靶点生成分子的多指标性能画像  
**雷达图轴：**DrugClip、Docking、BBB、PhysChem、Safety、SA、Final Score  
**线条：**EGFR、IDH1、VEGF

**绘制目的：**  
展示不同靶点生成结果的整体性能画像。

**注意事项：**  
所有指标必须先归一化到 0–1 或 0–100。Vina Dock Score 需要转换为越高越好的 Docking Affinity Score。

**分析重点：**

- 哪个靶点整体能力更均衡？
- 哪个靶点在 DrugClip 或 Docking 上更强？
- 哪个靶点在 BBB 或 Safety 上存在短板？

---

### 6.3 平行坐标图

**推荐图形：**parallel coordinates plot  
**图表标题：**高分候选分子的多指标表现  
**每条线：**一个分子  
**每个轴：**一个指标  
**颜色：**综合 Score 或靶点

**建议只绘制：**

- Top 20 或 Top 30 分子
- 或 Score ≥ 70 的分子

**绘制目的：**  
观察高分候选分子是否在多个关键指标上同时表现优秀。

**分析重点：**

- 高分分子是否同时满足 DrugClip 高、Docking 好、BBB 通过？
- 是否存在某些分子依赖单一指标获得高分？
- 哪些指标是高分分子的共同特征？

---

### 6.4 Top candidate 多指标热图

**推荐图形：**heatmap  
**图表标题：**Top 候选分子的多指标表现热图  
**行：**Top-K 分子  
**列：**DrugClip、Docking、BBB、MW、TPSA、LogP、Pgp、PAINS、Brenk、SA、Final Score  
**颜色：**归一化后的指标值

**绘制目的：**  
集中展示最终候选分子的优势和短板。

**分析重点：**

- 哪些 Top 分子表现最均衡？
- 哪些分子虽然 Final Score 高，但存在 BBB 或结构警报问题？
- 是否存在值得优先进入后续 MD 模拟或实验验证的候选？

---

## 7. 消融实验可视化

### 7.1 不同消融设置下综合 Score 对比图

**推荐图形：**boxplot、violin plot、bar plot with error bar  
**图表标题：**不同消融设置下生成分子的综合 Score 对比  
**横轴：**实验设置  
**纵轴：**综合 Score  
**颜色或分面：**靶点

**推荐横轴顺序：**

1. Full Model
2. 去除 Chain-of-Thought
3. 去除 Feedback
4. 去除 Physical Eval
5. 去除 In-domain Knowledge
6. 去除 GBM LoRA
7. Base Model

**绘制目的：**  
评估不同模块对最终生成质量的贡献。

**分析重点：**

- 哪个消融设置导致综合 Score 下降最多？
- Base Model 与 Full Model 差距有多大？
- GBM LoRA 是否显著提升 GBM 场景下的生成质量？
- Physical Eval 是否主要影响 BBB 和理化性质？

---

### 7.2 消融实验 Top-K hit rate 图

**推荐图形：**分组柱状图、折线图、点图  
**图表标题：**不同消融设置下 Top-K hit rate 对比  
**横轴：**实验设置  
**纵轴：**Top-K hit rate  
**颜色：**靶点

**推荐定义：**

```text
Top-K hit rate = 达到筛选阈值的分子数量 / 总生成分子数量
```

可以设置以下阈值：

- Score ≥ 70
- Score ≥ 80
- DrugClip > 0 且 Vina Dock 达标
- DrugClip > 0 且 BBB permeant 且 PAINS = 0

**绘制目的：**  
比均值更贴近实际药物筛选目标，衡量每个实验设置产生可用候选分子的能力。

**分析重点：**

- Full Model 是否产生更多高分候选？
- 去除哪个模块后 hit rate 下降最明显？
- 不同靶点对模块的依赖是否不同？

---

### 7.3 消融实验相对下降百分比图

**推荐图形：**horizontal bar chart、lollipop chart  
**图表标题：**消融实验相对于 Full Model 的性能下降比例  
**横轴：**相对下降百分比  
**纵轴：**实验设置  
**颜色：**指标或靶点

**计算方式：**

```text
Relative Drop = (Full Model 指标 - Ablation 指标) / Full Model 指标 × 100%
```

可分别计算：

- 综合 Score 下降
- DrugClip Score 下降
- Docking Affinity Score 下降
- BBB 通过率下降
- Top-K hit rate 下降

**绘制目的：**  
直观展示各模块的重要性。

**分析重点：**

- 哪个模块对综合质量贡献最大？
- 哪个模块主要影响 DrugClip？
- 哪个模块主要影响 BBB 或理化性质？
- Base Model 与 Full Model 的相对差距有多大？

---

### 7.4 消融实验 × 靶点二维热图

**推荐图形：**heatmap  
**图表标题：**不同靶点对各模块的依赖程度  
**行：**消融设置  
**列：**靶点  
**颜色：**平均综合 Score 或相对下降百分比

**可绘制多张热图：**

- 综合 Score 热图
- DrugClip Score 热图
- Docking Affinity Score 热图
- BBB 通过率热图
- Top-K hit rate 热图

**绘制目的：**  
分析不同靶点是否依赖不同模块。

**分析重点：**

- EGFR 是否更依赖 Feedback？
- IDH1 是否更依赖 In-domain Knowledge？
- VEGF 是否更依赖 Physical Eval？
- 哪个靶点在 Base Model 下退化最严重？

---

### 7.5 消融实验多指标雷达图

**推荐图形：**radar chart  
**图表标题：**消融实验下的多指标性能画像  
**雷达图轴：**DrugClip、Docking、BBB、PhysChem、Safety、SA、Final Score

**推荐绘制方式：**

1. 每个靶点单独绘制一张雷达图。
2. 每张图只对比 Full Model、某个关键消融和 Base Model。
3. 不建议将所有消融实验同时放在一张雷达图上，否则可读性较差。

**绘制目的：**  
分析每个模块主要提升哪类能力。

**可能结论示例：**

- 去除 Physical Eval 后，MW、TPSA、LogP、BBB 等相关指标下降。
- 去除 In-domain Knowledge 后，DrugClip Score 下降。
- 去除 GBM LoRA 后，GBM 靶点相关性和 BBB 相关性能下降。
- Base Model 在多个指标上整体偏弱。

---

### 7.6 Base Model vs Full Model 的 DrugClip-Docking 对比图

**推荐图形：**二维散点图、分面散点图  
**图表标题：**Full Model 与 Base Model 在 DrugClip-Docking 空间中的分布对比  
**横轴：**DrugClip Score  
**纵轴：**Vina Dock Score 或 Docking Affinity Score  
**颜色：**实验设置  
**点形状：**BBB permeant

**绘制目的：**  
直观展示完整模型是否将生成分子推向更理想的 DrugClip-Docking 区域。

**分析重点：**

- Full Model 是否产生更多高 DrugClip 和好 docking 的分子？
- Base Model 是否存在更多 DrugClip < 0 的分子？
- Full Model 是否提高了 BBB permeant 高质量候选的比例？

---

## 8. 筛选流程可视化

### 8.1 分子筛选漏斗图

**推荐图形：**funnel chart、step bar chart  
**图表标题：**GBM 候选分子筛选流程  
**横轴或纵轴：**剩余分子数量  
**分组：**靶点或实验设置

**推荐筛选步骤：**

```text
全部生成分子
↓
DrugClip ≥ 0
↓
Vina Dock 达标
↓
BBB permeant
↓
PAINS / Brenk 可接受
↓
Synthetic Accessibility 可接受
↓
Top-K 候选分子
```

**绘制目的：**  
展示从全部生成分子到最终候选分子的筛选过程。

**分析重点：**

- 哪一步淘汰分子最多？
- 不同靶点的主要瓶颈是否不同？
- 消融实验中哪些设置在早期筛选阶段损失较多？

---

### 8.2 分子淘汰原因统计图

**推荐图形：**堆叠柱状图  
**图表标题：**不同靶点生成分子的主要淘汰原因  
**横轴：**靶点  
**纵轴：**淘汰分子数量或比例  
**颜色：**淘汰原因

**推荐淘汰原因类别：**

- DrugClip < 0
- Vina Dock 不达标
- BBB 不通过
- PAINS/Brenk Alerts 过多
- Synthetic Accessibility 过差
- MW/TPSA/LogP 不合理

**绘制目的：**  
识别生成模型当前主要短板。

**分析重点：**

- 如果 DrugClip < 0 比例高，说明靶点相关性不足。
- 如果 BBB 不通过比例高，说明需要增强 BBB-aware 约束。
- 如果结构警报比例高，说明需要优化生成分子的安全性和稳定性。

---

## 9. 分子结构和化学空间可视化

### 9.1 Top 分子 2D 结构网格图

**推荐图形：**RDKit molecule grid image  
**图表标题：**各靶点 Top 候选分子的二维结构  
**每个分子标注：**Score、DrugClip、Vina Dock、BBB、SA

**绘制目的：**  
直观展示最终候选分子的化学结构。

**建议：**

- 每个靶点展示 Top 6 或 Top 9。
- 分子结构下方用简短中文标签说明关键指标。
- 避免一次展示过多分子，影响可读性。

---

### 9.2 Scaffold 分布图

**推荐图形：**scaffold frequency bar chart、scaffold network  
**图表标题：**生成分子的 Scaffold 分布  
**横轴：**Scaffold 类型  
**纵轴：**出现频次  
**颜色：**靶点或实验设置

**绘制目的：**  
分析生成分子的结构多样性。

**分析重点：**

- Top 分子是否集中在少数 scaffold？
- 不同靶点是否产生不同 scaffold 家族？
- 消融实验是否导致 scaffold 多样性下降？

---

### 9.3 分子相似性 UMAP / t-SNE 图

**推荐图形：**UMAP scatter plot、t-SNE scatter plot  
**图表标题：**生成分子的化学空间分布  
**横轴：**UMAP-1 或 t-SNE-1  
**纵轴：**UMAP-2 或 t-SNE-2  
**颜色：**靶点、实验设置或综合 Score  
**点大小：**DrugClip 或 Docking Affinity  
**点形状：**BBB permeant

**推荐方法：**

1. 用 RDKit 计算 Morgan fingerprint。
2. 计算分子间 Tanimoto similarity。
3. 使用 UMAP 或 t-SNE 降维。
4. 绘制二维化学空间图。

**绘制目的：**  
展示不同靶点和不同模型设置生成分子的化学空间分布。

**分析重点：**

- 三个靶点生成分子是否处于不同化学空间？
- 高分分子是否集中在某些区域？
- Base Model 是否生成更分散但质量较低的分子？
- Full Model 是否在保持多样性的同时提高高分分子密度？

---

### 9.4 Top 分子相似性热图

**推荐图形：**similarity matrix heatmap、hierarchical clustering heatmap  
**图表标题：**Top 候选分子的 Tanimoto Similarity 热图  
**行和列：**Top-K 分子  
**颜色：**Tanimoto similarity

**绘制目的：**  
判断最终候选分子是否具有结构多样性。

**分析重点：**

- Top 分子是否高度相似？
- 是否需要加入 diversity penalty？
- 是否可以从不同 scaffold 中选择代表性候选进入后续验证？

---

## 10. 推荐 Figure 组织方式

如果用于论文、组会或答辩，建议将图表组织为四组 Figure。

### Figure 1：三靶点主结果表现

建议包含：

- A. 三靶点综合 Score 分布
- B. DrugClip Score 分布，并标出 0 阈值线
- C. Vina Dock Score 分布
- D. BBB 通过率柱状图
- E. DrugClip vs Vina Dock 散点图
- F. Top-K hit rate 柱状图

**核心信息：**  
完整模型能够在不同 GBM 靶点上生成具有较好靶点相关性、结合能力和 BBB 潜力的分子。

---

### Figure 2：药物样性质与筛选合理性

建议包含：

- A. MW、TPSA、LogP 分布
- B. TPSA vs LogP，颜色表示 BBB
- C. PAINS/Brenk Alerts 统计
- D. Synthetic Accessibility 分布
- E. 多指标相关性热图
- F. Top candidate 多指标热图

**核心信息：**  
生成分子不仅在 DrugClip 和 Vina Dock 上表现较好，也在 BBB、理化性质、安全警报和合成可及性方面具备一定合理性。

---

### Figure 3：消融实验分析

建议包含：

- A. 不同消融设置综合 Score 分布
- B. 各消融设置 Top-K hit rate
- C. 消融相对下降百分比
- D. 消融实验 × 靶点热图
- E. 消融实验多指标雷达图
- F. Base Model vs Full Model 的 DrugClip-Docking 散点对比

**核心信息：**  
各模块对生成质量有不同贡献，其中某些模块主要提升靶点相关性，某些模块主要改善 BBB、理化性质或综合筛选质量。

---

### Figure 4：候选分子展示与化学空间分析

建议包含：

- A. Top 分子 2D 结构图
- B. UMAP 化学空间分布
- C. Scaffold 频率图
- D. Top 分子 Tanimoto Similarity 热图
- E. 分子筛选漏斗图

**核心信息：**  
最终候选分子具有较高综合 Score，并且在结构上具有一定多样性，不是单一 scaffold 的重复生成。

---

## 11. 最优先绘制的 10 张图

如果时间有限，建议优先完成以下 10 张图：

| 优先级 | 图表 | 作用 |
|---|---|---|
| 1 | 三靶点综合 Score 分布图 | 展示整体生成质量 |
| 2 | DrugClip Score 分布图 | 展示靶点相关性，并体现 DrugClip < 0 硬阈值 |
| 3 | Vina Dock Score 分布图 | 展示 docking 表现 |
| 4 | DrugClip vs Vina Dock 散点图 | 同时展示两个核心指标 |
| 5 | BBB 通过率柱状图 | 突出 GBM 场景需求 |
| 6 | TPSA vs LogP 散点图 | 解释 BBB 相关理化空间 |
| 7 | Top-K hit rate 图 | 展示可用候选分子比例 |
| 8 | 消融实验综合 Score 对比图 | 分析模块贡献 |
| 9 | 消融实验相对下降百分比图 | 量化各模块重要性 |
| 10 | Top candidate 多指标热图 | 汇总最终候选分子优缺点 |

---

## 12. 推荐分析叙事主线

建议按照以下逻辑组织汇报或论文结果部分：

1. **总体质量：**先比较 EGFR、IDH1、VEGF 三个靶点下生成分子的综合 Score 和 Top-K hit rate。
2. **核心指标：**重点展示 DrugClip 和 Vina Dock，说明分子既具有靶点相关性，又具有潜在结合能力。
3. **GBM 特异性：**强调 BBB permeant、TPSA 和 LogP，说明候选分子更符合脑部药物筛选需求。
4. **药物样性质：**分析 MW、SA、PAINS 和 Brenk，说明候选分子的可开发性。
5. **消融实验：**比较不同模块去除后的性能下降，解释 Chain-of-Thought、Feedback、Physical Eval、In-domain Knowledge 和 GBM LoRA 的作用。
6. **候选分子展示：**展示 Top 分子结构、scaffold 和化学空间分布，说明最终候选具有一定结构多样性。

---

## 13. 图表中文命名建议

| 英文术语 | 图中推荐写法 |
|---|---|
| DrugClip Score | DrugClip Score |
| Vina Dock Score | Vina Dock Score |
| BBB permeant | BBB permeant |
| Molecular Weight | MW |
| Topological Polar Surface Area | TPSA |
| Consensus LogP | Consensus LogP |
| Pgp substrate | Pgp substrate |
| PAINS Alerts | PAINS Alerts |
| Brenk Alerts | Brenk Alerts |
| Synthetic Accessibility | Synthetic Accessibility |
| Final Score | 综合 Score |
| Top-K hit rate | Top-K hit rate |
| Full Model | Full Model |
| Base Model | Base Model |
| Chain-of-Thought | Chain-of-Thought |
| Feedback | Feedback |
| Physical Eval | Physical Eval |
| In-domain Knowledge | In-domain Knowledge |
| GBM LoRA | GBM LoRA |
| UMAP | UMAP |
| t-SNE | t-SNE |
| Tanimoto similarity | Tanimoto similarity |

建议图表主标题、坐标轴说明和图注使用中文，例如：

```text
不同 GBM 靶点生成分子的综合 Score 分布
DrugClip Score 与 Vina Dock Score 的关系
不同消融设置下 Top-K hit rate 对比
Top 候选分子的多指标表现热图
```

---

## 14. 总结

整体可视化应围绕“GBM 靶点药物分子生成是否有效”这一核心问题展开。DrugClip 和 Vina Dock 是最关键的质量指标，BBB 是 GBM 药物设计中的重要场景约束，MW、TPSA、LogP、PAINS、Brenk 和 Synthetic Accessibility 则用于评估候选分子的药物样性质、安全性和可开发性。

推荐优先构建以下四类图：

1. **主结果图：**证明模型能生成高质量候选分子。
2. **指标解释图：**说明高分来源于 DrugClip、Vina Dock、BBB 和理化性质的综合表现。
3. **消融实验图：**证明各模块确实对生成质量有贡献。
4. **候选分子图：**展示最终分子的结构和化学空间多样性。

通过这套可视化体系，可以形成从“生成质量评估”到“模块贡献解释”，再到“候选分子筛选”的完整证据链。
