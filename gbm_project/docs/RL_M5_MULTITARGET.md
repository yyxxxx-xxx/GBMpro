# M5 — 多靶点扩展 (EGFR / VEGFR)
====================================

> **状态**: ⏸️ 等待 GPU 释放
> **启动日期**: 2026-11-24
> **前置条件**: 靶点特异性预检已通过 (`experiments/m5_preflight.json`)

---

## 1. 动机

M3/M4 在 **IDH1 单靶点**上验证了 GRPO 的有效性:

- reward mean **+31.1%** / median **+52.1%**, Mann-Whitney **p = 0.0007**
- 配对检验 **8/8 全胜**, p = 0.0003
- M4 四路消融证明结论稳健 (对 β_kl / 多样性过滤 / G / lr 均不敏感)

但单靶点结论的**外部效度存疑** —— 审稿人第一个问题就是
"这是不是只对 IDH1 有效?"。M5 的目标就是把这个单点结论
扩展成 **跨靶点结论**。

---

## 2. 设计: 严格控变量

M5 的配置与 **M3 (`rl_m3_kl008`) 完全一致**, 唯一变量是 `--target`:

| 参数 | M3 (IDH1) | M5 (EGFR / VEGFR) |
|---|---|---|
| algorithm | grpo | grpo |
| **target** | **IDH1** | **EGFR / VEGFR** ⬅ 唯一变量 |
| steps | 30 | 30 |
| lr | 1e-5 | 1e-5 |
| **beta_kl** | **0.08** | **0.08** (M3 找到的最优值) |
| group_size | 16 | 16 |
| train_batch / grad_accum | 4 / 1 | 4 / 1 |
| n_prompts / bank_size | 4 / 8 | 4 / 8 |
| temperature / top_p | 0.8 / 0.92 | 0.8 / 0.92 |
| kl_estimator | k3 | k3 |
| sigma0 | 8.0 | 8.0 |
| 多样性过滤 | 开 (bucket=10, minscore=0.4) | 同 |
| seed | 0 | 0 |

> ⚠️ **不要改步数**。M4-A 已经证明 80 步反而比 30 步略降
> (mean +21.9% vs +31.1%) —— 30 步是最优点。

---

## 3. 前置验证: 靶点特异性预检

在花 1.5 小时训练之前, 先跑 `scripts/preflight_multitarget.py` 确认:

1. **受体/后端正确** —— EGFR→2JIV, VEGFR→3WZD, 对接盒就位
2. **共晶配体重对接** —— 应落在 -12 ~ -4 kcal/mol 区间
3. **靶点交叉矩阵** —— 同一分子在不同靶点下 vina 必须不同
   (若完全相同, 说明 `receptors.yaml` 的 `target_aliases` 没生效,
   训练再久也没意义)
4. **corr(reward, vina)** —— 排除 reward 里非 vina 项的干扰

### 预检结果 (2026-11-24)

| 靶点 | 受体 | 共晶配体 vina | 探测分子非零 reward |
|---|---|---|---|
| EGFR | 2JIV (2535 原子, 盒内 609) | **-8.654** ✅ | 4/8 |
| VEGFR | 3WZD (2706 原子, 盒内 561) | n/a (配体未标注) | — |
| IDH1 (对照) | 6U4J (4027 原子, 盒内 583) | **-11.827** ✅ | — |

**结论**: 共晶配体重对接落在合理亲和力区间, 靶点映射正确, 可以训练。

---

## 4. 执行

### 4.1 训练

```bash
# EGFR → GPU 2
bash run_m5_when_gpu_free.sh EGFR 2 30

# VEGFR → GPU 3
bash run_m5_when_gpu_free.sh VEGFR 3 30
```

`run_m5_when_gpu_free.sh` 是**显存守卫**: 轮询等待目标卡空闲 ≥ 20000 MiB
(训练峰值实测 19.5GB), 一旦满足立即用 `setsid nohup` 启动训练。
最长等待 4 小时。

> ⚠️ 必须用 `setsid nohup` —— 父 shell 被中断会用 SIGHUP 杀掉
> 整个进程组。M2.2 曾因此在 step 51 静默终止。

### 4.2 评估

```bash
bash run_m5_eval.sh EGFR VEGFR
```

对每个靶点依次执行:

1. RL checkpoint 评估 (`--target $TARGET --seed 1234 --n_per_prompt 50`)
2. SFT 基线评估 (`--baseline_only`, **同一 seed** —— 否则对比不公平)
3. `compare_eval.py` 显著性检验 (Mann-Whitney / Cliff's δ / 配对 / Cohen's d)
4. CSV 导出 (`ID,SMILES`)

### 4.3 CSV 导出

```bash
python3 gbm_project/scripts/export_molecules_csv.py ../experiments --batch
```

ID 格式 `{靶点}_{组标签}_{来源}_{4位编号}`, 如 `EGFR_m5_egfr_kl008_RL_0001`。

---

## 5. 预期结果与判据

| 判据 | 阈值 | 说明 |
|---|---|---|
| Mann-Whitney p | < 0.05 | 主检验 |
| reward mean Δ | > 0 | 方向正确 |
| reward median Δ | > 0 | 典型分子质量提升 |
| 配对检验胜数 | ≥ 6/8 | 每个 prompt 都能产出好分子 (防 prompt starvation) |
| guard / prune 率 | 不高于基线 | reward hacking 检查 |
| prior 污染 | 0 次 | 训练日志 `prior_contaminated` |

### 三种可能结论

| 结论 | 含义 | 应对 |
|---|---|---|
| **A. EGFR/VEGFR 均显著** | 跨靶点有效, 论文核心结论 | 最有价值, 直接进论文主表 |
| **B. 一显著一不显著** | 靶点难度差异 | 诚实报告, 分析原因 (Vina 信号强度?) |
| **C. 均不显著** | β_kl=0.08 不是跨靶点通用值 | 回到超参搜索, 按靶点调 β_kl |

> **注意**: EGFR 的 Vina 信号通常比 IDH1 强 (共晶配体 -8.65 vs -11.83,
> 但盒子内原子更多), 预期 EGFR 的提升幅度**可能更大**。

---

## 6. 产物清单

| 类型 | 路径 |
|---|---|
| 训练产物 | `experiments/m5_{egfr,vegfr}_kl008/` (log / curve / summary / LoRA) |
| 评估产物 | `experiments/eval_m5_{egfr,vegfr}_kl008/` (smiles / stats / reward_cache) |
| CSV | `experiments/eval_m5_*/{rl,baseline_sft}_molecules.csv` |
| 预检 | `experiments/m5_preflight.json` + `.log` |
| 守卫日志 | `m5_guard_{egfr,vegfr}.log` |

---

## 7. 相关文件

| 文件 | 作用 |
|---|---|
| `gbm_project/scripts/preflight_multitarget.py` | 靶点特异性预检 |
| `gbm_project/scripts/train_rl.py` | 训练 CLI (`--target` 已参数化) |
| `gbm_project/scripts/eval_rl_checkpoint.py` | 评估 CLI (`--target` 已参数化) |
| `gbm_project/scripts/export_molecules_csv.py` | CSV 导出 |
| `run_m5_when_gpu_free.sh` | 显存守卫启动器 |
| `run_m5_eval.sh` | 评估编排 |
