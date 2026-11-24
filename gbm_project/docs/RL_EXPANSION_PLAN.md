# 扩大实验方案（待批准）

> 依据: M3/M4 阶段性结果 ｜ 日期: 2026-11-24
> 状态: **已决策 —— 暂不执行**（2026-11-24 用户决定：先用于阶段汇报，
> 不加 top-k 加权，等审阅具体命令后再启动）

## 📌 决策记录（2026-11-24）

| 问题 | 决定 |
|---|---|
| 扩大范围 | **暂不扩大** —— 先把当前 M3/M4 结果用于阶段汇报 |
| top-k 加权 | **不做** —— 保持代码最小改动 |
| 何时启动 | **等用户审阅具体命令/配置后再开始** |

因此本方案**仅作为备选清单存档**，批 1–3 的命令尚未执行。
下方内容保留了完整的可行性审计结论（seed 隔离已正确、EGFR/VEGFR 受体就绪、
DAP Bug 已修复），将来批准时可直接照此执行，无需重新审计。

---

## 0. 当前结论摘要（决策依据）

| 结论 | 证据 |
|---|---|
| RL 链路正确且有效 | M3: median **+52.1%**, MW **p=0.0007**, 配对 **8/8** |
| `beta_kl=0.08` 是最优工作点 | 0.02→0.08 斜率 ×12; 0.15 退化 |
| 30 步优于 80 步 | M3(30步) mean +31.1% > A(80步) +21.9% |
| 多样性过滤有保护作用 | 关掉后配对检验 3/8, p=1.0 |
| 对 G/lr 不敏感 | D(G=8,lr=5e-6) 仍 p=0.0369 |
| 无 reward hacking | guard/prune 率**低于**基线 |

**因此扩大实验的核心目的不是"再刷一遍更大的数"，而是补齐
① 统计稳健性 ② 靶点/算法/对照的外部有效性 ③ 论文级可复现性。**

---

## 1. 建议的扩大实验（分 3 批，共约 12 GPU·小时）

### 批 1：统计稳健性（最高优先级，约 4 h）

**问题**：目前所有结论都建立在**单一 seed=1234** 上，无法排除采样运气。

**方案**：主配置（M3: β=.08, G=16, lr=1e-5, 30 步）跑 **3 个额外 seed**，
每个 seed 独立采样评估 + 独立基线。

| run | seed | 用途 |
|---|---|---|
| M3-s2345 / s3456 / s4567 | 2345 / 3456 / 4567 | 种子方差 |

**产出**：mean ± std、跨 seed 的一致性检验（若 4/4 seed 均 p<0.05 → 结论坐实）

**代码改动**：✅ **无需改动**。
已审计确认 seed 隔离**本来就是对的**（`trainer.py:591-602`）：

```python
rng = random.Random(cfg.seed * 1000 + self.step)   # prompt 抽样
prompt_idxs = rng.sample(...)
...
self.rollout.generate(..., seed=cfg.seed * 100 + self.step)   # rollout 采样
```

prompt 抽样与 rollout 采样各用**独立派生 seed**，换 `--seed` 即可干净地
隔离"采样运气"。`eval_rl_checkpoint.py` 的 `--seed` 也已存在且已验证
（同 seed 两次运行得到完全相同的 116 个分子）。

---

### 批 2：靶点与算法外部有效性（约 6 h）

| run | 靶点 | 算法 | 说明 |
|---|---|---|---|
| EGFR-1 | EGFR | GRPO | Vina 最强靶点，REINVENT 也最强 → 对比最严苛 |
| EGFR-2 | EGFR | GRPO | 换 seed |
| IDH1-dap | IDH1 | **DAP-Lite** | 自适应 σ，REINVENT 原始路线对照 |
| IDH1-grpo-nodiv | IDH1 | GRPO | 关多样性过滤（更大样本量复现 M4-C）|

**靶点前置条件：✅ 已审计确认就绪**

```
$ python3 -c "from gbm_target_evaluator import *"
  IDH1  -> key=IDH1_IDH2  pdbqt=data/receptors/IDH1_IDH2_6U4J_chainA.pdbqt
  EGFR  -> key=EGFR      pdbqt=data/receptors/EGFR_2JIV_chainA.pdbqt
  VEGFR  -> key=VEGF_VEGFR pdbqt=data/receptors/VEGF_VEGFR_3WZD_chainA.pdbqt
```

`configs/receptors.yaml` 已配 3 个靶点 + 硬截断阈值
（`vina_max=-5.0, mw_max=600, tpsa_max=120, logp_max=5`），
`target_aliases` 已含 `IDH1 / EGFR / VEGFR` → **无需任何配置改动**。
（可顺带把 VEGFR 作为第 4 个靶点加入批 2，GPU 还有余量。）

**DAP-Lite 代码审计：⚠️ 发现并修复了一个潜伏 Bug**

`src/rl/losses.py` 同时含 `grpo_loss`（第 61 行）与 `dap_loss`（第 171 行），
`compute_loss` 按 `--algorithm` 分发。审计时**实测跑了一遍**才发现：

> **DAP 路径从 M0 到 M4 阶段从未被真实跑通过。**
> `compute_loss` 盲目 `**kwargs` 透传 `kl_estimator` / `length_normalize`
> （GRPO 专有参数）→ `TypeError: dap_loss() got an unexpected keyword
> argument 'kl_estimator'`。单元测试也没覆盖到这个分支。

已修复（commit `4e0b01c`）：
- `compute_loss` 按算法**显式筛选**参数
- `dap_loss` 新增 `length_normalize` 并在序列级目标上应用
  （否则长 SMILES 的 |seq_logprob| 更大 → 平方误差被长分子主导 →
  模型系统性偏好短分子）

**修复后实跑验证通过**：
```
[step 0] loss=+7.2610 kl=0.00000 rew=0.2434 gnorm=13.35
[step 1] loss=+2.2549 kl=0.00000 rew=0.3785 gnorm=2.59   ← 收敛
```

DAP 的核心特征（审计确认）：
- **无显式 KL 项**（`kl` 恒为 0）——靠 `sigma` 自适应标定代替
- `augmented = seq_prior + sigma * r`，`r` 为 batch 级 z-score
- `adaptive_sigma` 按 prior logprob 与 reward 的尺度自动定 σ
  （修正 REINVENT 原始 σ=128 的量纲问题）

⚠️ **DAP 路径下 `beta_kl` 不生效**。跑 DAP 时应调 `--sigma0`（默认 8.0）
而非 `beta_kl`。批 2 的 DAP run 固定 `sigma0=8.0`。

> 💡 **这暴露了一个方法论问题**：M0–M4 的所有结论都建立在 GRPO 一条路径上，
> 而 DAP 这条"备选对照"路径连跑都没跑过。**批 2 之前的任何对照实验都是
> 不可靠的**——这本身就是把 DAP 纳入批 2 的最强理由。

---

### 批 3：针对性消融（回答评审最可能问的问题，约 2 h）

| 消融 | 配置 | 回答的问题 |
|---|---|---|
| E1 | β=.05 | 最优 KL 工作点的精度扫描（现在只有 .08/.15 两个点）|
| E2 | β=.08, G=32 | 组大小敏感性（REINVENT 论文用 G=16–32）|
| E3 | β=.08, 30 步, **lr=2e-5** | 学习率是否还能更高（30 步时还没收敛？）|
| E4 | β=.08, 60 步（**从 M3 checkpoint resume**）| 30→60 步到底涨还是跌？|

**E4 特别说明**：M3 只存了 30 步的 checkpoint，可用 `--resume` 直接续训，
**零额外代码改动**，且能干净回答"训练更久是否有害"。

---

## 2. 需要的代码改动（小而可审计）

### 改动 1：`--seed` 贯通 —— ✅ **审计后确认无需改动**

`trainer.py:591-602` 已经正确解耦：

```python
rng = random.Random(cfg.seed * 1000 + self.step)   # prompt 抽样
prompt_idxs = rng.sample(range(len(self.prompt_bank.prompts)), ...)
...
self.rollout.generate(..., seed=cfg.seed * 100 + self.step)   # rollout 采样
```

prompt 抽样与 rollout 采样各用独立派生 seed。`eval_rl_checkpoint.py --seed`
已存在且已验证确定性（同 seed 两次运行得到完全相同的 116 个分子）。

→ **批 1/2/3 全部可直接开跑，无需改训练代码。**

### 改动 2：reward 上限自检（可选，低风险）

**现状**：`reward_max` 在 30 步时为 0.7065，但组内最优的提升不显著。
REINVENT 的做法是对 top-k 分子**额外加权**。

**改法（可选）**：在 `advantage.py` 增加 `topk_boost` 参数
（对组内前 k% 的 reward 乘以 1+w），默认关闭，单独作为一个消融 run。
**风险**：低（默认值 0 = 当前行为不变）。

### 改动 3：评估脚本支持多 seed 汇总（必须）

新增 `scripts/multi_seed_compare.py`：读入多个 `eval_*/` 目录，
输出跨 seed 的 mean ± std、逐 seed 显著性、汇总检验。

**不改**：`src/rl/` 下的核心训练逻辑、reward 实现、已有脚本行为。

---

## 3. 时间与资源预算

| 批次 | GPU·小时 | 4 卡并行墙钟 | 前置 |
|---|---|---|---|
| 批 1（3 seed） | 3 | ~1.2 h | 无需改代码 |
| 批 2（4 run） | 4 | ~1.6 h | EGFR reward 审计 |
| 批 3（4 run） | 4 | ~1.6 h | 无需改代码 |
| 评估（每 run ~25 min × 11） | — | ~1.5 h（并行） | — |
| **合计** | **11** | **~4 h** | |

**风险与缓解**：
- 4 卡并行时 reward 阶段是 CPU 瓶颈（60 核）→ `n_workers=8` × 4 run = 32 进程，可接受
- PEFT 加载权重硬编码 `cuda:0` → 已用 `CUDA_VISIBLE_DEVICES` 隔离解决（上一轮已修）

---

## 4. 明确**不做**的事（避免范围蔓延）

- ❌ 不改动 `src/gbm/` 原有 reward / 分子设计代码
- ❌ 不引入新的第三方 RL 框架（保持零依赖，便于复现）
- ❌ 不做 RL 训练侧的模型结构改动（LoRA r/alpha 已验证可用）
- ❌ 不改评估的评分口径（reward 同源性是结论可信的前提）

---

## 5. 交付物

1. `docs/RL_STAGE_REPORT.md` 更新（跨 seed 汇总表）
2. `docs/RL_INTEGRATION_DESIGN.md` 追加批 1–3 结果
3. `results/multi_seed_summary.md` + 汇总曲线图
4. 全部产物推送到 `https://github.com/yyxxxx-xxx/GBMpro.git`

---

## 6. 请您决策的 3 个点

| # | 问题 | 选项 |
|---|---|---|
| Q1 | 扩大范围 | A) 只做批 1（seed 稳健性，最快 ~1.5h）<br>B) 批 1+3（稳健性 + 消融，~3h）<br>C) 全部 1+2+3（~4h，含 EGFR）|
| Q2 | 是否做 top-k 加权（改动 2） | A) 做，作为独立消融 run<br>B) 不做，先补统计稳健性 |
| Q3 | 是否立即启动 | A) 批准后立即开始<br>B) 等我进一步审阅代码改动 |

---

## 附：当前 Git 状态

- `master` 比 `origin/master` **领先 1 个 commit**（`a082bac`，含本汇报文档与 M4 结果）
  —— 网络原因 push 较慢，已在后台重试
- 工作区干净，无未提交的代码改动
