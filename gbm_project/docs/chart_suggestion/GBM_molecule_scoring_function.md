# GBM 靶点生成分子筛选打分函数设计

## 1. 适用场景

本打分函数用于对针对不同 GBM 相关靶点（如 EGFR、IDH1、VEGF/VEGFR）的生成分子进行初筛排序。输入指标来自当前结果表中的以下字段：

- 生成模型/语义匹配：`DrugClip Score`
- 靶点结合能力：`Vina Dock Score` 或 `vina_docking_score`
- 成药性与理化性质：`MW`、`TPSA`、`Consensus Log P`
- 中枢暴露相关指标：`BBB permeant`、`Pgp substrate`
- 结构风险：`PAINS Alerts` / `PAINS #alerts`、`Brenk Alerts` / `Brenk #alerts`
- 可合成性：`Synthetic Accessibility`

当前附件数据概况如下：

| 靶点 | 分子数 | DrugClip 范围 | Vina Dock 范围 | Vina P5–P95 | BBB permeant=Yes |
|---|---:|---:|---:|---:|---:|
| EGFR | 92 | -2.25 ~ 1.61 | -8.06 ~ 7.47 | -7.93 ~ -2.96 | 40 |
| IDH1 | 100 | -0.75 ~ 2.47 | -10.11 ~ -1.37 | -9.29 ~ -4.47 | 45 |
| VEGF/VEGFR | 100 | -1.78 ~ 1.74 | -9.08 ~ 16.75 | -7.70 ~ 4.01 | 56 |

> 注：VEGF/VEGFR 的 docking 分数存在明显坏值离群点（最高 16.75），因此 docking 分数不建议直接使用全局 min-max 归一化，推荐使用每个靶点内部的 P5–P95 稳健归一化。

---

## 2. 总体原则

### 2.1 DrugClip 小于 0 时直接记 0 分

DrugClip 代表生成分子与目标语义/结构空间的一致性。若 DrugClip < 0，说明该分子与目标条件的匹配程度较差，即使 docking 或部分 ADMET 指标较好，也不建议进入优先筛选池。

**硬规则：**

```text
if DrugClip < 0:
    TotalScore = 0
```

### 2.2 DrugClip 与 Vina Dock 权重最高

生成分子筛选首先需要同时满足两个核心要求：

1. **生成方向正确**：DrugClip 分数较高，说明分子与目标任务更匹配。
2. **靶点结合潜力较好**：Vina Dock 分数越低，通常代表预测结合越有利。

因此二者设置为并列最高权重。

### 2.3 BBB 对 GBM 场景非常关键

GBM 药物开发需要考虑血脑屏障穿透能力。即使分子在靶点结合上表现较好，如果 BBB 不可渗透，后续体内有效暴露风险较高。因此 BBB 设置为第三高权重，但不作为绝对硬门槛，以避免过早淘汰可能通过结构优化、递送系统或前药策略改善的分子。

---

## 3. 推荐权重

总分范围为 0–100 分。

| 模块 | 权重 | 说明 |
|---|---:|---|
| DrugClip 匹配分 | 30% | 反映生成分子与目标条件的一致性；小于 0 直接总分为 0 |
| Vina Dock 结合分 | 30% | 反映靶点结合潜力；分数越低越好 |
| BBB 渗透分 | 15% | GBM 场景关键指标，影响中枢暴露 |
| 理化性质分 | 8% | 综合 MW、TPSA、LogP，保证基础 drug-likeness |
| P-gp 外排风险分 | 5% | P-gp substrate=Yes 可能降低脑内暴露 |
| 结构警报分 | 7% | PAINS/Brenk alerts 代表假阳性或毒性/反应性风险 |
| 可合成性分 | 5% | SA 越低越容易合成，影响后续实验可行性 |

**总分公式：**

```text
TotalScore = 100 × (
    0.30 × S_drugclip
  + 0.30 × S_vina
  + 0.15 × S_bbb
  + 0.08 × S_physchem
  + 0.05 × S_pgp
  + 0.07 × S_alert
  + 0.05 × S_sa
)
```

并加入硬性条件：

```text
if DrugClip < 0:
    TotalScore = 0
```

---

## 4. 各子分数定义

### 4.1 DrugClip 子分数

由于不同靶点的 DrugClip 分布不同，建议在每个靶点内部进行归一化。

```text
S_drugclip = clip(DrugClip / P95_positive_DrugClip, 0, 1)
```

其中：

- `P95_positive_DrugClip` 为当前靶点中 DrugClip > 0 分子的第 95 百分位。
- 若样本量较小，也可以使用当前靶点 DrugClip 正值最大值。
- `clip(x, 0, 1)` 表示小于 0 记为 0，大于 1 记为 1。

设计理由：

- DrugClip 代表生成方向是否正确。
- 使用 P95 而非最大值，可以避免单个极端高值压缩其他分子的得分。
- DrugClip < 0 已经由硬规则直接归零。

---

### 4.2 Vina Dock 子分数

Vina Dock 分数越低越好，因此采用反向归一化。考虑到 docking 分数可能存在坏值或离群点，建议使用每个靶点内部的 P5–P95 稳健范围。

```text
V_good = P5_vina
V_bad  = P95_vina

S_vina = clip((V_bad - Vina) / (V_bad - V_good), 0, 1)
```

解释：

- `V_good`：当前靶点 docking 分数的第 5 百分位，代表较优结合水平。
- `V_bad`：当前靶点 docking 分数的第 95 百分位，代表较差结合水平。
- 分子 docking 分数越接近或优于 `V_good`，`S_vina` 越接近 1。
- 分子 docking 分数越接近或差于 `V_bad`，`S_vina` 越接近 0。

设计理由：

- Vina Dock 是靶点结合潜力的核心指标。
- 不同靶点 docking 分数分布不同，不宜跨靶点直接比较原始值。
- 当前 VEGF/VEGFR 数据中存在 docking 坏值离群点，使用 P5–P95 更稳健。

---

### 4.3 BBB 渗透子分数

```text
S_bbb =
    1.00, if BBB permeant == "Yes"
    0.00, if BBB permeant == "No"
    0.50, if missing/unknown
```

设计理由：

- GBM 药物需要跨越血脑屏障，因此 BBB permeant=Yes 应获得明显加分。
- BBB permeant=No 不直接总分归零，是因为部分分子可能通过结构优化、递送系统或前药策略改善中枢暴露。
- 缺失值给中间分，避免因数据缺失而过度惩罚，但建议后续补算。

---

### 4.4 理化性质子分数

理化性质分由 MW、TPSA、Consensus LogP 三部分平均得到：

```text
S_physchem = mean(S_mw, S_tpsa, S_logp)
```

#### MW 子分数

```text
S_mw =
    1.0, if 250 <= MW <= 500
    linear from 0 to 1, if 150 <= MW < 250
    linear from 1 to 0, if 500 < MW <= 650
    0.0, otherwise
```

理由：

- 过小分子可能结合特异性不足。
- 过大分子通常渗透性与合成可行性变差。
- 对 GBM 场景，MW 不宜过高。

#### TPSA 子分数

```text
S_tpsa =
    1.0, if 20 <= TPSA <= 90
    linear from 0.7 to 1.0, if 0 <= TPSA < 20
    linear from 1.0 to 0.0, if 90 < TPSA <= 140
    0.0, if TPSA > 140
```

理由：

- 较低 TPSA 通常有利于 BBB 渗透。
- TPSA 过高会显著增加中枢暴露风险。
- TPSA 过低虽然有利渗透，但可能牺牲结合位点相互作用，因此不给满分下限以下的极小 TPSA。

#### LogP 子分数

```text
S_logp =
    1.0, if 1.0 <= LogP <= 4.0
    linear from 0.5 to 1.0, if 0.0 <= LogP < 1.0
    linear from 1.0 to 0.0, if 4.0 < LogP <= 6.0
    0.0, otherwise
```

理由：

- LogP 过低可能不利于膜渗透。
- LogP 过高可能增加非特异结合、溶解度和毒性风险。
- GBM 场景需要在渗透性与成药性之间平衡。

---

### 4.5 P-gp 外排风险子分数

```text
S_pgp =
    1.00, if Pgp substrate == "No"
    0.40, if Pgp substrate == "Yes"
    0.50, if missing/unknown
```

设计理由：

- P-gp substrate=Yes 可能导致药物被外排，降低脑内有效浓度。
- 但 P-gp 预测并非绝对准确，因此设置为中等惩罚，而非硬性淘汰。

---

### 4.6 结构警报子分数

结构警报分由 PAINS 与 Brenk 两部分构成：

```text
S_alert = 0.60 × S_pains + 0.40 × S_brenk
```

PAINS 子分数：

```text
S_pains =
    1.0, if PAINS alerts == 0
    0.2, if PAINS alerts == 1
    0.0, if PAINS alerts >= 2
```

Brenk 子分数：

```text
S_brenk = max(0, 1 - Brenk alerts / 5)
```

设计理由：

- PAINS alerts 更容易带来筛选假阳性，因此惩罚更重。
- Brenk alerts 代表潜在反应性、毒性或结构风险，但需要结合具体结构判断，因此采用线性扣分。

---

### 4.7 可合成性子分数

Synthetic Accessibility 越低越容易合成，通常 1 附近较优，6 以上较差。

```text
S_sa = clip((6 - SA) / 5, 0, 1)
```

解释：

- SA = 1 时，`S_sa = 1`
- SA = 6 时，`S_sa = 0`
- SA > 6 时，仍记为 0

设计理由：

- 生成分子如果难以合成，即使计算指标较好，后续实验推进成本也较高。
- 当前数据中 SA 大多数在 1–6.6 之间，因此该函数能较好地区分可合成性。

---

## 5. 推荐筛选分层

| 总分区间 | 建议 |
|---:|---|
| ≥ 75 | 优先候选分子，可进入人工结构复核、重对接或 MD 验证 |
| 60–75 | 次优候选，可作为结构优化或 scaffold hopping 起点 |
| 45–60 | 保留观察，除非具有特殊新颖骨架或某单项指标极优 |
| < 45 | 暂不推荐进入下一轮 |
| DrugClip < 0 | 直接淘汰，总分记 0 |

---

## 6. 可选硬性过滤规则

除 DrugClip < 0 外，可根据筛选严格程度加入以下硬过滤：

```text
Reject if PAINS alerts >= 2
Reject if MW > 700
Reject if TPSA > 160
Reject if Synthetic Accessibility > 7
Reject if Vina Dock Score is missing
```

建议默认阶段不要设置过多硬过滤，以免过早丢失具有优化潜力的新颖分子。初筛阶段可以先用总分排序，再对 Top 20–50 分子进行人工检查。

---

## 7. Python 伪代码

```python
import numpy as np

def clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def linear_score(x, x0, x1, y0, y1):
    if x1 == x0:
        return y1
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

def score_mw(mw):
    if 250 <= mw <= 500:
        return 1.0
    if 150 <= mw < 250:
        return clip(linear_score(mw, 150, 250, 0, 1))
    if 500 < mw <= 650:
        return clip(linear_score(mw, 500, 650, 1, 0))
    return 0.0

def score_tpsa(tpsa):
    if 20 <= tpsa <= 90:
        return 1.0
    if 0 <= tpsa < 20:
        return clip(linear_score(tpsa, 0, 20, 0.7, 1.0))
    if 90 < tpsa <= 140:
        return clip(linear_score(tpsa, 90, 140, 1.0, 0.0))
    return 0.0

def score_logp(logp):
    if 1.0 <= logp <= 4.0:
        return 1.0
    if 0.0 <= logp < 1.0:
        return clip(linear_score(logp, 0.0, 1.0, 0.5, 1.0))
    if 4.0 < logp <= 6.0:
        return clip(linear_score(logp, 4.0, 6.0, 1.0, 0.0))
    return 0.0

def score_bbb(value):
    if value == "Yes":
        return 1.0
    if value == "No":
        return 0.0
    return 0.5

def score_pgp(value):
    if value == "No":
        return 1.0
    if value == "Yes":
        return 0.4
    return 0.5

def score_alert(pains, brenk):
    if pains <= 0:
        s_pains = 1.0
    elif pains == 1:
        s_pains = 0.2
    else:
        s_pains = 0.0

    s_brenk = clip(1 - brenk / 5)
    return 0.60 * s_pains + 0.40 * s_brenk

def score_sa(sa):
    return clip((6 - sa) / 5)

def score_one_molecule(row, target_stats):
    drugclip = row["DrugClip"]
    vina = row["Vina"]
    mw = row["MW"]
    tpsa = row["TPSA"]
    logp = row["LogP"]
    bbb = row["BBB"]
    pgp = row["Pgp"]
    pains = row["PAINS"]
    brenk = row["Brenk"]
    sa = row["SA"]

    if drugclip < 0:
        return 0.0

    s_drugclip = clip(drugclip / target_stats["drugclip_positive_p95"])

    v_good = target_stats["vina_p5"]
    v_bad = target_stats["vina_p95"]
    s_vina = clip((v_bad - vina) / (v_bad - v_good))

    s_physchem = np.mean([
        score_mw(mw),
        score_tpsa(tpsa),
        score_logp(logp),
    ])

    total = 100 * (
        0.30 * s_drugclip
        + 0.30 * s_vina
        + 0.15 * score_bbb(bbb)
        + 0.08 * s_physchem
        + 0.05 * score_pgp(pgp)
        + 0.07 * score_alert(pains, brenk)
        + 0.05 * score_sa(sa)
    )

    return round(total, 2)
```

---

## 8. 推荐使用流程

1. 按靶点分别读取 EGFR、IDH1、VEGF/VEGFR 的分子结果。
2. 对每个靶点分别计算：
   - `drugclip_positive_p95`
   - `vina_p5`
   - `vina_p95`
3. 对每个分子计算总分。
4. DrugClip < 0 的分子直接记为 0 分。
5. 每个靶点内部按总分排序。
6. 选取每个靶点 Top 20–50 个分子进行人工结构复核。
7. 对优先分子进一步做：
   - docking pose 检查
   - scaffold novelty 检查
   - 关键相互作用分析
   - ADMET 二次评估
   - 必要时进行分子动力学模拟或自由能计算

---

## 9. 为什么不建议只看单一指标

- **只看 DrugClip**：可能得到生成方向合理但靶点结合较弱的分子。
- **只看 Vina Dock**：可能得到 docking 分数好但生成目标不匹配、结构风险高或 BBB 差的分子。
- **只看 BBB**：可能忽略靶点结合与分子新颖性。
- **只看 Lipinski/理化性质**：可能得到“像药”的分子，但不一定对目标有效。

因此推荐使用“DrugClip + Vina Dock 为核心，BBB 为 GBM 场景增强项，ADMET/SA/结构警报为风险修正项”的加权综合评分。

---

## 10. 最终推荐

本项目建议采用以下默认筛选函数：

```text
if DrugClip < 0:
    TotalScore = 0
else:
    TotalScore = 100 × (
        0.30 × S_drugclip
      + 0.30 × S_vina
      + 0.15 × S_bbb
      + 0.08 × S_physchem
      + 0.05 × S_pgp
      + 0.07 × S_alert
      + 0.05 × S_sa
    )
```

该函数符合当前任务需求：

- DrugClip < 0 的分子直接淘汰。
- DrugClip 与 Vina Dock 权重最高，分别占 30%。
- BBB 占 15%，突出 GBM 药物跨血脑屏障的重要性。
- P-gp、PAINS/Brenk、SA 和理化性质作为风险控制项，避免仅凭生成模型或 docking 分数选择不可推进的分子。
- 使用每个靶点内部归一化，避免不同靶点 docking 分布差异导致跨靶点评分失真。
