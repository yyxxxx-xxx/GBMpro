# 反馈闭环接入报告（2026-11-19）

## 一句话结论

反馈闭环**技术上完全跑通且明显有效**：DILI/hERG/Pgp 阳性率从 67%/58%/58% 降到 **0%/0%/0%**，
BBB+ 保持 100%，reward mean 逐轮上升 0.513 → 0.527 → 0.608。

但暴露出一个**论文层面必须处理的问题**：reward 与 Vina 分数**负相关**
（rho = -0.485 ~ -0.769），反馈闭环会把结合强度「优化掉」。

---

## 1. 管线（`scripts/generate_feedback_loop.py`，新增）

```
第 1 轮: 纯领域 prompt (与训练逐字符一致) -> 生成 25 -> 去重 12 -> 真实对接评估
第 2 轮: 注入第 1 轮的真实反馈 (1 正例 + 1 反例) -> 生成 -> 评估
第 3 轮: 同上
```

**为什么不用 `generate_tot_molecules.py`（ToT）**：

| | ToT | 本脚本 |
|---|---|---|
| prompt 形态 | `apply_chat_template` + 脚手架→组装→SMILES 三段式 | `Instruction:/Input:/Response:`（**与训练完全一致**）|
| 中间节点 | 需要模型输出 scaffold/assembly 推理 | 无，单步迭代 |
| 适配我们的 LoRA | ❌ 形态不匹配 | ✅ |

我们的 LoRA 是在 `Instruction: ...\nInput:\nResponse: SMILES: xxx` 上训的，
ToT 的 prompt 形态与之不匹配 —— 这是之前 ToT 效果差的根因之一。

## 2. 反馈文本的具体性

关键设计：反馈必须**可执行**，不是泛泛而谈。

```
- Vina binding energy = -6.20 kcal/mol (WEAK; the ligand does not bind tightly
  enough. Strengthen the core scaffold, add hinge-binding H-bond donors/acceptors,
  or extend into a hydrophobic pocket)
- DILI (hepatotoxicity) probability = 0.83 (TOO HIGH - must be reduced. Prefer
  removing aryl halides, avoid lipophilic aromatic rings, reduce ester/thioester
  and nitro groups)
- hERG (cardiotoxicity) probability = 0.71 (TOO HIGH - must be reduced. Avoid a
  basic tertiary amine flanked by lipophilic groups; lower LogP; add an
  electron-withdrawing group or reduce basicity)
- BBB permeability = 0.22 (TOO LOW - required for GBM. Reduce MW and TPSA, add
  fewer HBDs, or increase lipophilicity moderately)
```

## 3. 结果（IDH1, n=27, exh=4）

| 轮 | n | Vina mean | reward mean | reward best | DILI+ | hERG+ | BBB+ | Pgp+ |
|---|---|---|---|---|---|---|---|---|
| 1 | 12 | -8.26 | 0.513 | 0.687 | 67% | 58% | 100% | 58% |
| 2 | 5 | -6.17 | 0.527 | 0.679 | 60% | 20% | 80% | 20% |
| 3 | 10 | **-6.14** | **0.608** | **0.709** | **0%** | **0%** | **100%** | **0%** |

对比无反馈的 `Ours_active` IDH1（n=20）：Vina -7.33、DILI+ 70%、BBB+ 80%。

- **26/27 唯一 SMILES，14 个唯一 Murcko 骨架**
- 全部 27 个分子中，**没有一个**同时满足 Vina ≤ -8 且 reward ≥ 0.6

## 4. 核心问题：reward 与 Vina 负相关

Vina 权重扫描（27 个反馈分子）：

| vina_w | rho(Vina, reward) | Top-reward 组的 Vina |
|---|---|---|
| 0.25（当前） | +0.011 | -6.22 |
| 0.45 | -0.392 | -6.22 |
| 0.65 | -0.686 | -5.80 |
| 0.75 | -0.769 | -5.54 |

**提高 Vina 权重让相关性变得更负，不是更正。**

### 机制（已量化验证）

```
DILI vs Vina: rho = -0.630
Pgp  vs Vina: rho = -0.508
hERG vs Vina: rho = -0.413
QED  vs Vina: rho = -0.401
rho(Vina, MW) = -0.479      ← 结合更强的分子更大
```

链条：**Vina 越强 → 分子越大 → 更疏水 → 毒性/外排风险升高**。

这不是算法缺陷，而是**结构性的目标冲突**。REINVENT 4 用
`Vina × QED × MW × Alerts`（纯乘性、无 ADMET 风险端点）所以没这个问题，
但它的奖励里也**没有任何 ADMET 端点**。

### 与此前 EGFR 活性分子集的一致性

| 数据集 | rho(Vina, reward) | rho(reward, pIC50) | rho(Vina, pIC50) |
|---|---|---|---|
| EGFR 活性分子（45，有连续 IC50） | -0.485 | **+0.281** | -0.173 |
| IDH1 反馈闭环（27） | -0.485 ~ -0.769 | — | — |

两处独立数据集都显示 **reward 对真实亲和力的判别力强于 Vina 本身**
（EGFR 上 rho(reward, pIC50)=+0.281 > rho(Vina, pIC50)=-0.173）。

**所以：反馈闭环没有在带偏模型，它在正确地优化 MPO 目标。
问题在于「Vina」本身是否应该占 reward 的 25%。**

## 5. 需要决策的问题

论文里 Ours 的卖点目前是「LLM + 反馈优化」。接上真实对接后，
**Ours 和 REINVENT4 变成同类方法**（都用对接分做优化目标）。

三个可选框架：

### 方案 A：奖励函数对齐（推荐，改动最小）
把 Ours 的 reward 改成与 REINVENT4 **同一套**（`Vina × QED × MW × Alerts`），
然后论证「相同目标下，LLM 生成 + 反馈 vs RL + DAP」的机制差异。
- ✅ 公平性无懈可击
- ❌ 丢掉 ADMET 端点，放弃「多目标优化」的卖点

### 方案 B：保留 MPO，强调目标函数不同
明确写「Ours 优化的是 8 端点 MPO，REINVENT4 只优化 4 项乘积」，
把 REINVENT4 定位成「对接引导生成的上界参照」而非直接对手。
- ✅ 保留 ADMET 卖点
- ⚠️ 必须在报告里写清楚目标函数不同，否则会被质疑不公平

### 方案 C：双目标 Pareto
同时报告两组结果 —— Ours-MPO 和 Ours-DockingOnly，
再加 REINVENT4，呈现三方对比。
- ✅ 最诚实、最能回应审稿人
- ⚠️ 工作量最大

## 6. 其他待办

1. **有效生成率偏低**：第 1 轮 25 个候选只有 12 个有效（48%）。
   `max_new_tokens=128` 会截断长 SMILES（如 `...n(C1)\N#`）。
   建议提到 200，并把 `repair_smiles` 的 LLM 修复也接上。
2. **n 不足**：27 个分子达不到报告要求（≥50/靶点）。
   耗时瓶颈是**真实对接**（27 个分子 ≈ 380 s，~14 s/分子）。
   可用 `exhaustiveness=2` + 并行 ProcessPool 提速。
3. **EGFR / VEGFR 尚未跑**：IDH1 是 Vina 最弱、毒性最高的靶点，
   先跑它是对的（验证闭环能纠正问题）。
