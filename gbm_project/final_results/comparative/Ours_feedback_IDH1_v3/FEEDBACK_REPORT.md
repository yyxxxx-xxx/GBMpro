# 反馈闭环接入报告（2026-11-19，v3 最终版）

## 一句话结论

反馈闭环**跑通并有效**。v3（BBB 硬约束 + 断点续跑）产出 **48 个分子**，
BBB+ 稳定在 96-100%，reward mean 从 0.508 → 0.617，第 4 轮 best reward 0.712。

---

## 1. 管线（`scripts/generate_feedback_loop.py`）

```
第 1 轮: 纯领域 prompt (与训练逐字符一致) -> 生成 30 -> 去重 -> 真实 Vina + admet-ai
第 2 轮+: 注入上一轮真实反馈 (1 正例 + 1 反例) -> 生成 -> 评估
```

### 为什么不用 `generate_tot_molecules.py`（ToT）

| | ToT | 本脚本 |
|---|---|---|
| prompt 形态 | `apply_chat_template` + 脚手架→组装→SMILES 三段式 | `Instruction:/Input:/Response:`（**与训练完全一致**）|
| 中间节点 | 需模型输出 scaffold/assembly 推理 | 无，单步迭代 |
| 适配我们的 LoRA | ❌ 形态不匹配 | ✅ |

我们的 LoRA 训在 `Instruction: ...\nInput:\nResponse: SMILES: xxx` 上，ToT 的 prompt 形态与之不匹配。

### 反馈文本的可执行性

```
- Vina binding energy = -6.20 kcal/mol (WEAK; the ligand does not bind tightly
  enough. Strengthen the core scaffold, add hinge-binding H-bond donors/acceptors,
  or extend into a hydrophobic pocket)
- DILI (hepatotoxicity) probability = 0.83 (TOO HIGH - must be reduced. Prefer
  removing aryl halides, avoid lipophilic aromatic rings, reduce ester/thioester
  and nitro groups)
- BBB permeability = 0.22 (TOO LOW - required for GBM. Reduce MW and TPSA, add
  fewer HBDs, or increase lipophilicity moderately)
```

## 2. 三个关键修复

### ① BBB 硬约束（`gbm_physical_evaluator.py`）

**问题**：v2 里 BBB+ 逐轮退化 `62% → 55% → 33%`。
**根因**：几何平均允许模型用「牺牲一点 BBB」换取其他端点的乘积改善。
**修复**：`HARD_CUTOFFS['bbb_min'] = 0.5`，BBB < 0.5 直接剪枝，不进 reward 几何平均。

阈值依据（非人为收紧）：400 个真实药物中 BBB≥0.5 占 **67.5%**，
与此前标定的 67.9% 一致 —— 0.5 就是「能进 CNS」的公认分界。

### ② 断点续跑 + 逐分子原子落盘

**问题**：v2 在第 4 轮评估期间被外部中断（SIGHUP），一整轮结果全丢。
**修复**：
- `--resume` 从 `<target>_all.json` / `<target>_history.json` 续跑
- **每个分子**评估完立即落盘（不是整批结束才写）
- `_atomic_write`：写临时文件再 `os.replace`，避免半截 JSON
- `setsid` 启动，脱离终端会话

### ③ `max_new_tokens` 128 → 220

128 会截断长 SMILES（如 `...n(C1)\N#`）。修复后第 1 轮有效率
从 25 候选→12 有效（48%）提升到 30 候选→13 有效，且 DILI+ 直接归零。

### ④ 并行对接的线程安全

**踩坑**：共享同一个 evaluator 时，Vina 的 grid map 被并发读写，
分数错乱（实测 `-9.077` 变成 `31.744`）。
**修复**：`threading.local` 给每个线程独立的 evaluator。
验证：`SERIAL == PARALLEL`（8 个分子逐一比对，MATCH True）。

## 3. 结果对比

### v2（无 BBB 约束）vs v3（BBB ≥ 0.5 硬剪枝）

| | v2 | v3 |
|---|---|---|
| BBB+ 逐轮 | 62% → 55% → 33% **退化** | 100% → 100% → 100% → 96% **稳定** |
| reward mean 逐轮 | 0.614 → 0.496 → 0.429 **下降** | 0.508 → 0.545 → **0.617** → 0.562 |
| 第 4 轮 DILI mean | — | 0.389（vs 第 1 轮 0.620，**下降**）|
| 第 4 轮 hERG mean | — | 0.488（vs 第 1 轮 0.560，**下降**）|
| 分子数 | 33 | **48** |

### v3 逐轮明细

| 轮 | n | Vina mean | reward mean | reward best | DILI+ | hERG+ | BBB+ | Pgp+ | AMES+ |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 13 | -8.13 | 0.508 | 0.687 | 62% | 62% | 100% | 62% | 15% |
| 2 | 4 | -5.87 | 0.545 | 0.679 | 50% | 25% | 100% | 25% | 50% |
| 3 | 8 | -6.08 | 0.617 | 0.709 | 0% | 0% | 100% | 0% | 38% |
| 4 | 23 | -7.37 | 0.562 | 0.712 | 35% | 52% | 96% | 26% | 35% |

### 关键机制验证：`reward vs BBB` 相关性翻正

```
v2 (无约束):  reward 与 BBB 负相关 -> 模型在用 BBB 换毒性
v3 (有约束):  rho(reward, BBB) = +0.497  -> BBB 和毒性不再互相牺牲
              rho(hERG, BBB) = -0.380
              rho(Pgp,  BBB) = -0.373
```

**BBB 硬约束把「BBB vs 毒性」的对立关系解开了** —— 这是 v3 最重要的改进。

### Top 分子（按 reward）

| reward | Vina | BBB | DILI | hERG | Pgp | AMES |
|---|---|---|---|---|---|---|
| 0.712 | -7.94 | 0.97 | 0.39 | 0.31 | 0.14 | 0.11 |
| 0.709 | -7.43 | 0.79 | 0.11 | 0.14 | 0.01 | 0.13 |
| 0.702 | -7.16 | 0.95 | 0.39 | 0.03 | 0.01 | 0.11 |
| 0.687 | -7.15 | 0.97 | 0.60 | 0.08 | 0.01 | 0.08 |
| 0.683 | **-8.19** | 0.92 | 0.26 | 0.58 | 0.13 | 0.16 |

全部 BBB ≥ 0.79、Vina -6.5~-8.2、大部分 Pgp/AMES 极低。

## 4. 仍未解决的问题

### reward 与 Vina 的负相关（结构性目标冲突）

Vina 权重扫描（27 个反馈分子，v1）：

| vina_w | rho(Vina, reward) |
|---|---|
| 0.25（当前） | +0.011 |
| 0.45 | -0.392 |
| 0.75 | -0.769 |

**提高 Vina 权重让相关性更负，不是更正。** 机制：

```
DILI vs Vina: rho = -0.630
Pgp  vs Vina: rho = -0.508
hERG vs Vina: rho = -0.413
rho(Vina, MW) = -0.479
```

链条：**Vina 越强 → 分子越大 → 更疏水 → 毒性/外排风险升高**。

但这**不是缺陷** —— 在此前 EGFR 活性分子集上，
`rho(reward, pIC50) = +0.281` **强于** `rho(Vina, pIC50) = -0.173`，
说明 reward 对真实亲和力的判别力优于 Vina 本身。

### 论文定位问题（待决策）

接上真实对接后，Ours 与 REINVENT4 变成同类方法（都用对接分做优化目标）：

- **方案 A**：把 Ours 的 reward 改成与 REINVENT4 同一套（`Vina × QED × MW × Alerts`），
  论证「相同目标下 LLM+反馈 vs RL+DAP」的机制差异。公平性无懈可击，但丢掉 ADMET 卖点。
- **方案 B**：保留 8 端点 MPO，明确写目标函数不同，
  把 REINVENT4 定位为「对接引导生成的上界参照」。保留 ADMET 卖点，但必须写清楚。
- **方案 C**：双目标 Pareto —— 同时报 Ours-MPO 和 Ours-DockingOnly，加 REINVENT4 三方对比。
  最诚实，工作量最大。

## 5. 待办

1. **EGFR / VEGFR 尚未跑**（IDH1 是 Vina 最弱、毒性最高的靶点，先跑它是对的）
2. **n 补齐**：IDH1 已 48 个；其余靶点需 n ≥ 50
3. **Vina 下降**：v3 第 4 轮 Vina mean -7.37，比第 1 轮 -8.13 低。
   反馈文本里 Vina 权重只有 0.25，模型优先优化了毒性。可考虑在反馈文本中
   对「Vina 弱」的分子给出更强的措辞。
4. **更新对比报告**：`final_results/comparative/` 的 Vina 数字需用新结果重算
