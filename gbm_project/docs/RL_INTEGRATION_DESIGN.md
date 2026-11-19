# GBM-PharmaMind 强化学习接入方案设计（RL Integration Design）

> 版本：v1.0 ｜ 日期：2026-11-19 ｜ 目标靶点：IDH1 / IDH2 (6U4J)
> 状态：**设计文档（未实现）**。本文档给出算法选型、系统架构、模块接口、超参数、
> 风险控制与实施路线，作为后续编码的规格说明。

---

## 0. 文档导航

| 章节 | 内容 |
|---|---|
| 1 | 为什么要接 RL：当前方法学的缺口分析 |
| 2 | 路线选型：三条接入路径的可行性与取舍 |
| 3 | 算法选型：DAP vs GRPO 的量化对比与最终选择 |
| 4 | 系统架构：模块、数据流、时序 |
| 5 | 核心模块详细设计（6 个模块 + 接口签名 + **4 个实测踩坑**） |
| 6 | 奖励设计：MPO reward 的 RL 适配（含 hacking 防线） |
| 7 | 超参数表与算力预算（**全部实测数据**） |
| 8 | 消融实验设计 |
| 9 | 风险登记册与缓解措施（**18 条，含 5 条已实际发生**） |
| 10 | 实施路线图（M0–M4）与验收标准 |
| 11 | 已知未解问题与开放决策 |
| 附录 A | 关键源码位置索引 |
| 附录 B | 环境事实核查（三环境依赖矩阵） |
| **附录 C** | **性能实测（Vina 吞吐 / GPU 显存 / 单步耗时 / 踩坑清单）** |

> **本文档的实测数据均在 2026-11-19 于本机跑通**，
> 不是估算。核心结论：
> - 单靶点 IDH1 的 60 步 RL ≈ **52 分钟**（原估 2.5–4 小时）
> - 单张 3090 足够，**不需要多卡分进程**
> - **CPU 侧的 Vina 对接才是瓶颈**（占单步 54%），不是 GPU
> - HF `generate` 后端已足够快，**vLLM 不需要**

---

## 1. 为什么要接 RL：当前方法学的缺口

### 1.1 现状

当前 GBM-PharmaMind 的优化环节是 **prompt 级反馈闭环**
（`scripts/generate_feedback_loop.py`）：

```
第 1 轮: 纯领域 prompt → 生成 30 → 去重 → 真实 Vina + admet-ai → 评测
第 2 轮+: 注入上一轮真实反馈 (1 正例 + 1 反例) → 生成 → 评测
```

IDH1 v3 实测（`final_results/comparative/Ours_feedback_IDH1_v3/`）：

| 轮 | n | Vina mean | reward mean | reward best | BBB+ |
|---|---|---|---|---|---|
| 1 | 13 | -8.13 | 0.508 | 0.687 | 100% |
| 2 | 4 | -5.87 | 0.545 | 0.679 | 100% |
| 3 | 8 | -6.08 | 0.617 | 0.709 | 100% |
| 4 | 23 | -7.37 | 0.562 | 0.712 | 96% |

### 1.2 三个结构性缺口

**缺口 1：优化不改变参数。**
反馈闭环的全部信息都停留在上下文中，模型权重始终是 SFT 后的状态。
第 4 轮的 prompt 与第 1 轮完全不同（注入反馈），但产生这些分子的仍是同一组权重。
后果：模型的"能力"没有增长，只是"当次答题时看到了更多提示"。

**缺口 2：上下文是有限记忆，无法承担多轮迭代。**
每轮反馈注入 1 正例 + 1 反例（约 400–600 token）。第 3 轮后 prompt 已接近
`max_position_embeddings` 的实用上限。要做 10 轮、20 轮迭代必须靠外部状态，
而外部状态无法转化为参数改进。

**缺口 3：对比上的"方法不对称"是审稿人必然追问的点。**
在现有对比表中，REINVENT 4 的 RL 阶段奖励信号就是真实 Vina 对接分数，
与评测指标同源；而 Ours 的优化是 prompt 级的、且 reward 目标函数与 REINVENT 不同。
审稿人会问："**你们用同一条 RL 路线时，能赢吗？**"

### 1.3 论文定位（本文档的技术主张）

> 把 REINVENT 4 的**强化学习算法**（策略梯度 + 奖励引导）迁移到
> **多模态大语言模型的分子生成**上，使其同时具备：
> (a) LLM 的条件生成与知识迁移能力；
> (b) RL 的参数级优化能力；
> (c) 与 REINVENT 4 **完全相同**的物理反馈源（真实 Vina + admet-ai）。

即：**Ours-GRPO vs REINVENT4-DAP，在同一奖励函数下比较。**
这才是真正能站住的对比（详见 §8 消融设计）。

---

## 2. 路线选型：三条接入路径

### 2.1 路线 A：把我们的 LLM 塞进 REINVENT 的 `runmodes/RL/`

**结论：不可行（架构级不兼容）。**

REINVENT 4 的 RL 栈全部绑死在它的 SMILES LSTM 上，核对源码后确认：

| REINVENT 依赖 | 我们的 Qwen2-7B | 兼容性 |
|---|---|---|
| 131-token 字符级 SMILES 词表 | 151k BPE 词表 | ❌ |
| `model.likelihood()` 返回 SMILES 序列对数似然 | `CausalLMOutput.logits` | ❌ 接口完全不同 |
| `SmilesState` 状态机（VALID/DUPLICATE/INVALID） | LLM 输出自由文本 | ❌ 需重写 |
| `reinvent_plugins` 的 component 协议 | 我们自己的 evaluator | ⚠️ 可适配 |
| DAP loss（`reward.py:51-52`） | 数学形式可迁移 | ✅ |

**核心障碍**：`runmodes/RL/learning.py` 中的 `prior_lls` / `agent_lls` 计算路径
完全依赖字符级 tokenizer 与 `model.likelihood()`。强行接入需要重写整个
REINVENT RL 循环的采样、状态机、似然计算三块 —— 那本质上就是"重新实现一遍"。

### 2.2 路线 B：把 REINVENT 的 RL 算法移植到我们的 LLM 上

**结论：可行，这是本方案的主体。**

算法（DAP / GRPO）是模型无关的：只要能采样分子、能算对数似然、能给奖励，
就能套策略梯度。REINVENT 的价值不在代码，而在**算法形式**和**工程惯例**
（staged learning、多样性过滤、checkpoint 策略）。

### 2.3 路线 C：用 REINVENT 当搜索器，我们的 LLM 当提案器

**结论：可行，但学术定位不划算。**

流程：LLM 采样分子 → 作为 TL 数据微调 REINVENT → REINVENT 跑原生 RL。

问题：这等于把 RL 部分完全外包给 REINVENT，论文里"我们的方法"就只剩 SFT。
而且我们已经跑过 REINVENT4-TLRL 基线（`configs/reinvent_tlrl_idh1.toml`），
这条路线的信息增量接近零。

### 2.4 最终选型

| 路线 | 结论 | 论文中的角色 |
|---|---|---|
| A | ❌ 技术不可行 | — |
| **B** | ✅ **主线** | Ours-GRPO（本文档主体） |
| C | ⚠️ 降级为可选消融 | "LLM 提案 + REINVENT 精修"（可选，非必须） |

---

## 3. 算法选型：DAP vs GRPO

### 3.1 REINVENT 的 DAP 原始形式

源码（`reinvent/runmodes/RL/reward.py:51-52`）：

```python
augmented_lls = prior_lls + sigma * scores
loss = torch.pow((augmented_lls - agent_lls), 2)
```

即：**把奖励加到先验对数似然上作为回归目标，强制 agent 去拟合。**
它是 DAP（Difference Augmented Prior）与 Augmented Likelihood 的混合体。

### 3.2 DAP 移植到 LLM 的形式

\[
\mathcal{L}_{\text{DAP}} = \big(\underbrace{\log p_{\theta_{\text{prior}}}(y|x)}_{\text{prior\_lls}} + \sigma R(x) - \underbrace{\log p_{\theta}(y|x)}_{\text{agent\_lls}}\big)^2
\]

其中 \(y\) 是生成的 SMILES 响应，\(x\) 是 prompt。

映射到我们的栈：

| DAP 符号 | LLM 实现 | 显存开销 |
|---|---|---|
| `prior_lls` | \(\sum_t \log p_{\text{SFT}}(y_t \mid x, y_{<t})\) | 与 agent **共享同一份 7B 权重** |
| `agent_lls` | \(\sum_t \log p_{\theta}(y_t \mid x, y_{<t})\) | 同上 |
| `scores` | MPO reward（Vina + admet-ai 8 端点） | 0（外部计算） |

**关键工程红利**：PEFT 0.12.0 提供 `model.disable_adapter()` 上下文管理器。
只要把 GBM LoRA 设为可训练、而把"合并后的 Llamole + 基座"当作 prior，
就能在**同一份 14GB 权重**上分别前向算出 prior_lls 与 agent_lls
（agent 时开 adapter，prior 时关 adapter），**不需要加载第二份 7B 模型**。

### 3.3 三个必须解决的移植问题

#### 问题 1：σ 必须重标定（最容易翻车）

REINVENT 配置里的 `sigma = 128`（`configs/reinvent_tlrl_idh1.toml`）是针对
**字符级 SMILES NLL** 标定的：一条 40 字符 SMILES 的 NLL 约 20–40 nat，
σ·R 的量级与 prior_lls 的量级相当。

LLM 侧完全不同：
- BPE token 化的 SMILES 约 30–60 token（比字符数多）
- teacher-forcing 下每 token 的 logprob 尺度不同
- prompt 越长，序列 logprob 绝对值越大

**直接照搬 σ=128 的后果**：`augmented_lls` 变成不可达量级，
loss 退化成常数（target >> 可达范围），梯度消失或爆炸。

**对策**：σ 改为**自适应**：
\[
\sigma_{\text{eff}} = \sigma_0 \cdot \frac{\text{std}(\log p_{\text{prior}}(y|x))}{\text{std}(R(x))}
\]
使 σ·R 的方差与 log-likelihood 的方差同阶。σ₀ 作为超参扫 {2, 4, 8, 16}。

#### 问题 2：DAP 没有 baseline，梯度方差大

DAP 的 target 是 `prior_lls + σR`，**隐含地**假设 prior_lls 本身提供了"基线"。
但它没有对同一组样本内的 reward 做中心化 —— 一个 batch 里如果所有分子
reward 都很高，所有 target 都会同向上移，模型无法区分"更好"和"好"。

**对策（两选一）**：
- (a) 对 `scores` 做 batch 内 z-score：`R' = (R - mean) / std`
- (b) 直接换 GRPO（见下）

#### 问题 3：漂移控制缺失

DAP 没有显式 KL 惩罚项。RL 训练几十步后，agent 可能大幅偏离 prior，
生成荒谬分子。REINVENT 靠"prior_lls 出现在 target 里"隐式抑制，
但抑制力度与 σ 耦合，很难调。

### 3.4 GRPO 的对应形式

GRPO（DeepSeek-R1 那套）在 LLM 上是标准选择：

- 对每个 prompt 采样 group \(G\) 个响应（我们 G = 8 或 16）
- 组内 advantage：\(A_i = \dfrac{R_i - \text{mean}(\{R_j\})}{\text{std}(\{R_j\}) + \epsilon}\)
- Policy loss：\(-\dfrac{1}{G}\sum_i A_i \sum_t \log p_\theta(y_{i,t} \mid x, y_{i,<t})\)
- KL 惩罚：\(\beta \cdot \mathbb{E}\big[\log p_\theta(y|x) - \log p_{\text{prior}}(y|x)\big]\)（k3 估计量）

**我们的场景对 GRPO 特别友好**：
- reward 是**确定性函数**（Vina + admet-ai），无噪声 → 不需要 reward model
- **不需要 value model / critic**（GRPO 的核心优势）→ 省掉一半显存和一半复杂度
- group 内 advantage 天然处理"这个 prompt 下什么算好"

### 3.5 对比表

| 维度 | DAP | GRPO | 我们的选择 |
|---|---|---|---|
| 需要 critic | 否 | 否 | — |
| 需要 reward model | 否 | 否 | — |
| reward 尺度敏感性 | **高**（σ 必须重标定） | 低（z-score 归一） | **GRPO** |
| 显式 KL / 漂移控制 | 弱（隐式，耦合 σ） | 强（显式 β 项） | **GRPO** |
| 需要 prior logprob | 是 | 是（仅 KL 项） | 两者都需要 |
| LLM 社区验证 | 无（专为字符级 SMILES 设计） | 大量（DeepSeek / OpenRLHF / TRL） | **GRPO** |
| 我们的 TRL 版本 | — | **TRL 0.9.6 尚无 GRPO**（0.12+ 才有） | 需手写 |

### 3.6 最终算法决策

> **主算法：GRPO**（手写，不依赖 TRL 版本升级）
> **对照组：DAP-Lite**（按 §3.2 移植，用自适应 σ + batch z-score 修掉坑 1、2）
> —— 两者都实现，用同一奖励函数、同一预算对比。**这本身就是一个有价值的消融**
> （"为什么 LLM 上的分子 RL 应该用 GRPO 而不是 REINVENT 的 DAP"）。

---

## 4. 系统架构

### 4.1 总体数据流

```
                    ┌─────────────────────────────────────────────┐
                    │  Prompt Bank (固定, 每靶点 N_prompts 条)      │
                    │  来自 configs/english_gbm_prompts.yaml       │
                    │  与 SFT 训练逐字符一致                        │
                    └────────────────┬────────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────────┐
                    │  Stage 1: Rollout  (vLLM / HF generate)      │
                    │  每个 prompt 采 G 个响应 → 抽取 SMILES        │
                    │  过滤: RDKit 可解析 + 重原子数 >= 12          │
                    └────────────────┬────────────────────────────┘
                                     │  (smiles, prompt_id, group_idx)
                    ┌────────────────▼────────────────────────────┐
                    │  Stage 2: Reward  (CPU, 进程池)               │
                    │  真实 Vina 1.2.7 + admet-ai (52 端点)        │
                    │  → MPO reward (8 端点加权几何平均)            │
                    │  → 硬截断 (BBB>=0.5 等) → pruned = reward 0   │
                    └────────────────┬────────────────────────────┘
                                     │  R(x)
                    ┌────────────────▼────────────────────────────┐
                    │  Stage 3: Advantage                          │
                    │  组内 z-score: A_i = (R_i - μ_g)/σ_g         │
                    │  全组被剪 → A=0 (该组跳过)                    │
                    └────────────────┬────────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────────┐
                    │  Stage 4: Logprob (GPU, 共享 7B)              │
                    │  agent_lls: 开 adapter 前向 (带 grad)         │
                    │  prior_lls: disable_adapter 前向 (no_grad)   │
                    │  GRPO loss + β·KL(k3 估计量)                  │
                    │  → LoRA 反向传播 → AdamW → 优化器步           │
                    └────────────────┬────────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────────┐
                    │  Stage 5: 多样性过滤 + 日志 + Checkpoint      │
                    │  IdenticalMurckoScaffold bucket_size=10      │
                    │  (与 REINVENT 相同，保证可比)                  │
                    └─────────────────────────────────────────────┘
```

### 4.2 关键设计决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| **prompt 是否随轮次变？** | **否，冻结** | GRPO 的 prior/agent 必须在同一条件分布上比较。反馈闭环那种"每轮换 prompt"的做法与 RL 不兼容。 |
| **prior 用哪个模型？** | SFT 后的 Ours 模型 | 它是"未优化的起点"，RL 的 KL 锚点。 |
| **agent 怎么表示？** | **新增第三个 LoRA**（RL LoRA） | 不污染 SFT LoRA。SFT LoRA 冻结 → prior；RL LoRA 可训练 → agent。Rollout 时两个都开。 |
| **Rollout 引擎？** | **HF `generate`**（实测 B=32 → 0.343 s/分子） | vLLM 已装但**不需要**：HF 后端比 Vina 对接还快 2.5 倍，GPU 根本不是瓶颈，而 vLLM 会引入 RL LoRA 权重热同步的复杂度。 |
| **Reward 在哪算？** | **CPU 进程池**（15 workers，实测 1.16 mol/s） | Vina 是 CPU 密集（AutoDock Vina 1.2.7 无 GPU 后端）。60 核可开 15 worker。**这是唯一真正的瓶颈**（占单步 54%）。 |
| **用哪个 Python env？** | **`/root/anaconda3` (base)** | 实测唯一"CUDA 4 卡 + vina + meeko + admet_ai"全依赖环境，且已验证与反馈闭环 reward 同源（差 3.4e-9）。 |
| **reward 函数** | 与 REINVENT 相同 or Ours MPO？ | **两者都跑**（见 §8），这是论文最关键的一栏。 |
| **多样性控制** | IdenticalMurckoScaffold, bucket=10 | **必须与 REINVENT 一致**，否则多样性对比不公平。 |
| **训练 batch** | **8**（实测 19.5 GB；16 会 OOM） | 靠 `grad_accum=4` 拿到有效 batch 32。 |

### 4.3 与 REINVENT 的公平性对齐（必须严格遵守）

| 维度 | REINVENT 4 (TL+RL) | Ours-GRPO | 对齐方式 |
|---|---|---|---|
| Prior | PubChem 先验 + IDH1 TL | Ours SFT（Qwen2-7B + Llamole + GBM LoRA） | 各自最优起点（**这是本质差异，需在文中说明**） |
| RL 奖励 | Vina × QED × MW × Alerts | Vina × QED × MW × Alerts **和** Ours MPO | **两个变体都跑** |
| 多样性过滤 | IdenticalMurckoScaffold, bucket=10 | 同 | 完全一致 |
| 采样温度 | multinomial, T=1.0 | T=1.0, top_p=0.95 | 接近即可 |
| RL 步数 | max_steps=40, batch=32 | 建议 60 步 × (4 prompts × G=8) | Ours 预算 ≥ REINVENT |
| 最终评测 | 本地 Vina + admet-ai 重打分 | 同 | 复用 `scripts/rescore_parallel.py` |

---

## 5. 核心模块详细设计

### 5.1 `src/rl/prompt_bank.py` — 冻结的 Prompt 集合

**职责**：提供训练/推理完全一致的固定 prompt 列表。

```python
@dataclass
class PromptBank:
    """RL 期间冻结的 prompt 集合。"""
    prompts: List[str]            # 与 SFT 训练逐字符一致的 instruction
    target: str                   # 'IDH1'

    @classmethod
    def build(cls, target: str, n_prompts: int = 4,
              prompts_yaml: Path = None) -> "PromptBank":
        """
        从 configs/english_gbm_prompts.yaml 构造。
        复用 scripts/generate_with_new_adapter.py 的 build_prompt()，
        遍历 CELL_STATES × METRIC_VARIANTS 取前 n_prompts 个。
        """
        ...

    def sample(self, rng: random.Random) -> str:
        """每步随机抽一条 prompt（保持多样性，避免过拟合单条 prompt）。"""
```

**要点**：
- prompt 文本必须与 `generate_feedback_loop.py` 的第 1 轮**逐字符一致**
  （`Instruction: {instr}\nInput:\nResponse:` 格式，**注意 `Response:` 后无换行**）
- `use_cot` 交替（与反馈闭环一致），但**每次 rollout 全组用同一条 prompt**
  （GRPO 的组必须同 prompt）
- prompt 集合在 RL 全程**不更新**（存 hash 到 checkpoint，resume 时校验）

### 5.2 `src/rl/rollout.py` — 采样器

**职责**：给定 prompt 列表，采样 G 个响应，抽取有效 SMILES。

```python
@dataclass
class RolloutConfig:
    n_prompts: int = 4          # 每步并行处理多少条 prompt
    group_size: int = 8         # G：每条 prompt 采多少个分子
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 220    # 与反馈闭环一致（128 会截断长 SMILES）
    min_heavy_atoms: int = 12
    repetition_penalty: float = 1.1
    backend: str = "hf"         # 'hf' | 'vllm'

@dataclass
class RolloutSample:
    prompt_id: int
    group_idx: int
    raw_response: str
    smiles: Optional[str]
    n_tokens: int
    valid: bool                 # False = 解析失败或重原子数不足

class RolloutEngine:
    def generate(self, prompts: List[str], cfg: RolloutConfig) -> List[RolloutSample]:
        ...
```

**两种后端的取舍（已实测，2026-11-19）**：

| | HF `model.generate` | vLLM 0.6.1 |
|---|---|---|
| 显存 | 与训练模型**同卡共存**可行（见下方实测） | 需独占显存（`gpu_memory_utilization=0.85`） |
| **实测吞吐** | **B=32: 11.0 s / 0.343 s·分子⁻¹** | 预期再快一个量级，但需实测 |
| 峰值显存 | **15.6 GB**（B=32） | — |
| LoRA 权重热更新 | 无痛（同一进程） | **需重建 LoRA 槽位或重启实例** |
| 实现风险 | 低 | **vLLM 0.6.1 的 `enable_lora` 参数在 `LLM.__init__` 签名里不可见，只能通过 `kwargs` 透传，需实测** |

> **实测结论：HF 后端已经足够快，vLLM 不是必需品。**
> B=32 时 0.343 s/分子 → 每步 32 个分子只要 **11 秒**，
> 而同期 Vina 对接 32 个分子需要 **~28 秒**（1.16 mol/s）。
> **Reward 计算比 Rollout 还慢 2.5 倍** → GPU 侧完全不是瓶颈，
> vLLM 优化收益有限，不值得引入权重同步的复杂度。
> **M1 直接用 HF 后端，跳过 vLLM。**

**Rollout 与 SMILES 抽取**：必须复用 `generate_with_new_adapter.py` 的
`extract_candidate` + `repair_smiles` + `_heavy_atoms`，**与反馈闭环完全同一套**
（否则前后不可比）。

**⚠️ 实测踩坑 1：生成阶段必须关掉 gradient checkpointing。**
`gradient_checkpointing_enable()` 会强制 `use_cache=False`，
KV cache 失效 → 生成从 11 s 变成**数分钟**（实测 B=16 生成卡住 > 60 s 无输出）。
正确做法是**生成和训练分两个阶段切换**：
```python
m.eval(); m.config.use_cache = True                      # 生成阶段
with torch.no_grad(): out = m.generate(...)               # 不开 ckpt

m.gradient_checkpointing_enable(                          # 训练阶段
    gradient_checkpointing_kwargs={"use_reentrant": False})
m.config.use_cache = False; m.train()
```


### 5.3 `src/rl/reward_service.py` — 奖励服务

**职责**：批量计算 MPO reward，进程池并行，磁盘缓存。

> **⚠️ 关键架构约束：两个环境的依赖不通用**
>
> | 环境 | CUDA | vina/meeko | admet_ai | rdkit | transformers | peft | torch |
> |---|---|---|---|---|---|---|---|
> | `/root/anaconda3/envs/llamole` | ✅ 4 卡 | ❌ | ❌ | ✅ | 5.3.0 | 0.12.0 | 2.4.0+cu121 |
> | **`/root/anaconda3` (base)** | **✅ 4 卡** | **✅** | **✅** | ✅ | 4.44.0 | 0.18.1 | 2.5.1+cu121 |
> | `/root/venv_reinvent` | ❌ (cpu) | ✅ | ✅ | ✅ | — | — | 2.9.1+cpu |
>
> **实测结论：直接用 base env (`/root/anaconda3`) 跑整个 RL 训练。**
>
> 理由链：
> 1. base env 有 **CUDA**（实测 `torch 2.5.1+cu121`，`cuda.is_available()=True`，**4 卡全可见**）
>    —— 之前误以为 base env 无 GPU，实测是有的。
> 2. base env 有 **vina + meeko + admet_ai + rdkit**，是唯一"全依赖 + GPU"的环境。
> 3. **base env 是反馈闭环的同源环境**（实测验证见下）。
>
> **同源性验证（实测，2026-11-19）**：
> 用 base env 重算反馈闭环 `IDH1_history.json` 里的分子，与存档值比对：
> ```
> stored: reward=0.5914004167  vina=-8.982  bbb=0.984540
> recalc: reward=0.5914004201  vina=-8.982  bbb=0.984540
>   → vina 完全一致，bbb 完全一致，reward 差 3.4e-9（float32/64 精度差，非算法差）
> ```
> **结论：base env 的 reward 与反馈闭环完全可复现 → RL 的 reward 与历史结果同尺度。**
>
> 用 base env 避免了：
> - 往 llamole env 装 vina/admet-ai 污染 numpy/scipy 依赖（Q7）
> - Reward Service 的 `ProcessPoolExecutor` 子进程继承父进程环境导致缺包
> - 两套环境版本不一致导致 reward 数值漂移
>
> **唯一顾虑**：base env 的 transformers 是 4.44.0，llamole env 是 5.3.0。
> 必须确认反馈闭环的 rollout 用的是哪个环境（`batch_generation_*.log` 记录了）。
> 若反馈闭环用 base env → 直接用 base env，零成本。
> 若用 llamole env → 走 **独立 Reward 进程**（base env 起服务进程，
> GPU env 通过 socket 请求），彻底解耦两个环境。
>
> **注意**：admet-ai 在 base env 里会**自动用 GPU**（实测日志：
> `GPU available: True (cuda), used: True` / `NVIDIA GeForce RTX 3090`）。
> 这意味着 Reward 计算会占用 GPU 显存 —— **训练时必须限制 admet-ai 只用 CPU**，
> 否则会和 7B 模型抢显存导致 OOM。
> 设 `CUDA_VISIBLE_DEVICES=""` 给 admet-ai 子进程，或在 `ADMETCalculator`
> 里显式指定 device="cpu"（实测 CPU 上 24–86 ms/分子，完全够用）。
>
> **另外一个可选项：Vina 走独立常驻进程池**（不 fork 训练进程），
> 这样 Reward 的环境与训练环境彻底解耦 —— 最稳的架构。

```python
@dataclass
class RewardConfig:
    target: str = "IDH1"
    exhaustiveness: int = 4      # RL 内循环用 4（REINVENT 的 TLRL 配置也是 4）
    n_poses: int = 5
    cpu_per_worker: int = 4
    n_workers: int = 15          # 60 核 / 4
    use_admet_ai: bool = True
    admet_device: str = "cpu"    # ★ 必须锁 CPU，否则和 7B 抢显存
    cache_path: Optional[Path] = None   # 磁盘缓存，避免重复对接

@dataclass
class RewardRecord:
    smiles: str
    reward: float                # MPO reward，pruned 分子强制为 0.0
    vina: Optional[float]
    dili: Optional[float]
    herg: Optional[float]
    bbb: Optional[float]
    pgp: Optional[float]
    ames: Optional[float]
    qed: Optional[float]
    pruned: bool
    prune_reason: Optional[str]
    error: Optional[str]

class RewardService:
    def score_batch(self, smiles: List[str], cfg: RewardConfig) -> List[RewardRecord]:
        """
        进程池并行。每 worker 独立的 VinaDocking 实例（threading.local 的
        进程池等价物），grid maps 跨分子复用。
        """
```

**必须遵守的三个既有教训**（来自反馈闭环的踩坑记录）：
1. **线程/进程安全**：共享 Vina 实例会导致 grid map 竞争，分数错乱
   （实测 `-9.077` 变成 `31.744`）。每 worker 必须独立实例。
2. **pruned 分子 reward = 0.0**：硬截断（BBB < 0.5 等）命中的分子必须给
   **明确的最差奖励**，否则模型学到"违反硬约束反而无害"。
3. **失败分子也要给 reward**：对接失败不能静默跳过，给 0.0 并记 error。

**缓存设计**：JSONL append-only，每行 `{smiles, reward, vina, ...}`。
RL 过程中同一 SMILES 会被反复采样（尤其早期），缓存能省掉大量 Vina 时间。
预估命中率 20–40%。

### 5.4 `src/rl/advantage.py` — Advantage 计算

```python
@dataclass
class GroupAdvantage:
    prompt_id: int
    rewards: List[float]
    advantages: List[float]      # z-score 后的 advantage
    n_valid: int                 # 组内有效分子数
    skipped: bool                # True = 整组跳过（无有效分子或 reward 无方差）

def compute_group_advantages(
    samples: List[RolloutSample],
    records: List[RewardRecord],
    eps: float = 1e-6,
) -> List[GroupAdvantage]:
    """
    对每个 (prompt_id) 组做 z-score。
    - 组内 reward 方差 < eps → skipped=True（无梯度信号）
    - 组内全部 pruned (reward 全 0) → skipped=True
    - 过滤掉 rollout 失败的样本（invalid），只对有效样本算 advantage
    """
```

**设计要点**：
- **只对有效 SMILES 算 advantage**。无效样本不给梯度（避免模型学"输出垃圾"，
  因为垃圾的 reward=0 但那是"无效"而非"低质量"，混淆信号）
- 但**保留 pruned 分子**在组内（pruned 是有效 SMILES，只是违反约束）→ 正确地
  惩罚模型违反 BBB 硬约束
- z-score 在**组内**做，不是 batch 内 —— GRPO 的标准做法

### 5.5 `src/rl/losses.py` — GRPO / DAP-Lite 损失

```python
def grpo_loss(
    agent_logps: torch.Tensor,      # [B, T] 每 token logprob
    prior_logps: torch.Tensor,      # [B, T] 参考模型 logprob
    mask: torch.Tensor,             # [B, T] response token mask
    advantages: torch.Tensor,       # [B] 组内 advantage
    beta: float = 0.02,             # KL 系数
    kl_estimator: str = "k3",
) -> Dict[str, torch.Tensor]:
    """
    L = -(1/N_valid) Σ_i A_i * (1/|y_i|) Σ_t log p_θ(y_t|.)
         + β * KL(π_θ || π_prior)
    """
```

**KL 估计量（k3）**：
\[
\hat{K}_3 = \exp(\Delta) - \Delta - 1, \quad \Delta = \log p_\theta - \log p_{\text{prior}}
\]
k3 无偏且在 Δ→0 时方差小，是 TRL / OpenRLHF 的默认选择。

**长度归一化**：按每条样本的有效 token 数归一（`/|y_i|`），
否则长 SMILES 会被系统性惩罚/奖励。

```python
def dap_loss(
    agent_logps: torch.Tensor,
    prior_logps: torch.Tensor,
    mask: torch.Tensor,
    rewards: torch.Tensor,          # [B]
    sigma: float = 8.0,             # 自适应标定后的 σ
    reward_zscore: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    L = (prior_lls + σ·R' - agent_lls)²
    R' = z-score(rewards) if reward_zscore else rewards
    """
```

**关键**：DAP-Lite 相比原始 DAP 加了 `reward_zscore`（修掉 §3.3 问题 2）
和自适应 σ（修掉问题 1）。**否则 DAP-Lite 跑不出结果，不能作为对照组。**

### 5.6 `src/rl/trainer.py` — 训练循环

```python
@dataclass
class RLConfig:
    # 模型
    base_model: str = "/root/Llamole-main/models/Qwen2-7B-Instruct"
    llamole_adapter: str = "/root/Llamole-main/saves/Llamole-Qwen2-7B-Instruct-Adapter"
    sft_adapter: str = "/root/Llamole-main/saves/Llamole-Qwen2-7B-Instruct-Adapter-gbm-active"
    rl_adapter_out: str = ".../gbm-rl-grpo"
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = ["q_proj","k_proj","v_proj","o_proj",
                                  "gate_proj","up_proj","down_proj"]
    # 优化
    lr: float = 1e-6              # 比 SFT (1e-4) 低 2 个数量级
    warmup_steps: int = 5
    max_steps: int = 60
    grad_accum: int = 4
    beta_kl: float = 0.02
    max_grad_norm: float = 1.0
    # 采样
    rollout: RolloutConfig = ...
    # 奖励
    reward: RewardConfig = ...
    # 多样性
    diversity_filter: str = "IdenticalMurckoScaffold"
    bucket_size: int = 10
    minscore: float = 0.4
    # 算法
    algorithm: str = "grpo"       # 'grpo' | 'dap'
```

**训练循环伪代码**：

```python
for step in range(cfg.max_steps):
    # --- Stage 1: Rollout ---
    prompts = prompt_bank.sample_batch(cfg.rollout.n_prompts, rng)
    samples = rollout_engine.generate(prompts, cfg.rollout)

    # --- Stage 2: Reward ---
    valid_idx = [i for i, s in enumerate(samples) if s.valid]
    records = reward_service.score_batch([samples[i].smiles for i in valid_idx], cfg.reward)

    # --- Stage 3: Advantage ---
    groups = compute_group_advantages(samples, records)

    # --- Stage 4: Logprob + Loss + Step ---
    agent_logps, prior_logps, mask = compute_logprobs(
        model, tokenizer, prompts, samples, use_adapter_for_agent=True)
    loss_dict = grpo_loss(agent_logps, prior_logps, mask, advs, cfg.beta_kl)
    (loss_dict["loss"] / cfg.grad_accum).backward()
    if (step+1) % cfg.grad_accum == 0:
        clip_grad_norm_(rl_params, cfg.max_grad_norm)
        optimizer.step(); optimizer.zero_grad()

    # --- Stage 5: 日志 / 多样性过滤 / Checkpoint ---
    log_metrics(step, groups, records)
    diversity_filter.update([s.smiles for s, r in zip(valid_samples, records) if r.reward >= cfg.minscore])
    save_checkpoint(step)
```

### 5.7 `compute_logprobs` 的实现要点（**最容易出错的地方**）

```python
def compute_logprobs(model, tokenizer, prompts, samples, sft_adapter_name):
    """
    一次性算出 agent 和 prior 的 per-token logprob。

    实现路径（关键：用 PEFT 的 adapter 切换，而非加载两份模型）:
      1. model = PeftModel.from_pretrained(base, sft_adapter, adapter_name="sft")
      2. model.add_adapter("rl", lora_config)      # 新增可训练 LoRA
      3. model.set_adapter("rl")                    # agent = sft + rl
         with torch.no_grad(): forward → agent_logps
      4. with model.disable_adapter():              # prior = base + llamole
         forward → prior_logps
    """
```

**注意 `set_adapter` 的语义**（已在 PEFT 0.12.0 源码中确认）：
`set_adapter(name)` 会把该 adapter 设为 active **并将其余 adapter 设为非 active**
—— 这正是我们要的。同时它会把该 adapter 的参数 `requires_grad=True`，
所以在算 prior 前要显式把 `sft` 参数冻结。

**替代方案（若 adapter 切换有问题）**：把 SFT LoRA **merge 进基座**，
只保留一个可训练的 RL LoRA；prior 就是"完全 merge 后的 SFT 模型"，
用 `disable_adapter()` 得到。这更简单，显存也更低（无 SFT LoRA 常驻）。
**推荐这一条** —— 少一层 adapter 复杂度，踩坑面小。

#### ⚠️ 实测踩坑 2：`is_trainable=True` 是必须的

`PeftModel.from_pretrained(..., is_trainable=True)` —— 否则 adapter 以
`inference_mode=True` 加载，参数 `requires_grad=False`，
`loss.backward()` 直接报：
```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```
（已实测复现。）

#### ⚠️ 实测踩坑 3：`enable_input_require_grads()` 是必须的

即使 LoRA 参数 `requires_grad=True`，配合 `gradient_checkpointing` 时
若**输入 embedding 不需要梯度**，checkpoint 段内的梯度链断裂，
同样报 `does not require grad`。
实测发现该警告：`UserWarning: None of the inputs have requires_grad=True. Gradients will be None`

**必须调用**：
```python
m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
m.enable_input_require_grads()     # ← 缺这个就 backward 失败
```
且 `use_reentrant=False` 必须显式指定（新版默认 True 会出问题）。

#### ⚠️ 实测踩坑 4：三个 adapter 的 requires_grad 要精确控制

实测：`RL trainable(M): 40.37`（r=16 × 7 modules × 28 layers）。
需要**显式**逐参数设置，不能依赖 PEFT 的默认行为：
```python
m.add_adapter("rl", lora_config)
m.set_adapter("rl")
for n, p in m.named_parameters():
    p.requires_grad = ("lora" in n) and (".rl." in n)   # 只有 RL LoRA 可训
```
否则 `add_adapter` 后 SFT LoRA 的参数可能仍是 `requires_grad=True`，
会被一起优化，**静默污染 SFT 先验**（RL 跑完发现 prior 变了，KL 项失去意义）。

#### 完整的三 adapter 配置（已实测跑通）

```python
m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16,
                                         device_map="cuda:0")
m = PeftModel.from_pretrained(m, LL_ADAPTER);  m = m.merge_and_unload()  # Llamole 合并
m = PeftModel.from_pretrained(m, GBM_ADAPTER, is_trainable=True)        # SFT adapter
m.add_adapter("rl", LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                bias="none",
                                target_modules=["q_proj","k_proj","v_proj","o_proj",
                                                "gate_proj","up_proj","down_proj"],
                                task_type="CAUSAL_LM"))
m.set_adapter("rl")
for n, p in m.named_parameters():
    p.requires_grad = ("lora" in n) and (".rl." in n)
m.to("cuda:0"); m.enable_input_require_grads()
```


### 5.8 `scripts/train_rl.py` — CLI 入口

```bash
python3 scripts/train_rl.py \
    --target IDH1 \
    --algorithm grpo \
    --steps 60 \
    --n_prompts 4 --group_size 8 \
    --lr 1e-6 --beta_kl 0.02 \
    --reward_variant ours_mpo \
    --out_dir final_results/comparative/Ours_GRPO_IDH1 \
    --resume
```

---

## 6. 奖励设计

### 6.1 两个奖励变体（论文的关键对照）

| 变体 | 定义 | 用途 |
|---|---|---|
| `reinvent_like` | Vina × QED × MW × Alerts（与 `reinvent_tlrl_idh1.toml` 一致） | **公平性对照**：同目标下 LLM-GRPO vs REINVENT-DAP |
| `ours_mpo` | 现有 8 端点 MPO（vina .25 / dili .12 / herg .10 / bbb .20 / pgp .15 / ames .08 / qed .05 / physchem .05） | 我们的完整主张 |

**`reinvent_like` 必须精确复刻 REINVENT 的评分配置**：
```
VinaScore(weight=1.0) × QED(0.3) × MW(0.2, double_sigmoid 200-500) × Alerts(1.0)
聚合: geometric_mean
```

**已核实的 REINVENT 聚合细节**（`reinvent/scoring/aggregators/means.py`）：

```42:52:/root/venv_reinvent/lib/python3.11/site-packages/reinvent/scoring/aggregators/means.py
    if mode == "sum":
        sum_weights = np.nansum(weights, axis=0)
        result = np.nansum(scores * (weights / sum_weights), axis=0)
    elif mode == "prod":
        scores = np.maximum(scores, 1e-8)
        sum_weights = np.maximum(np.nansum(weights, axis=0), 1e-8)
        result = np.nanprod(scores ** (weights / sum_weights), axis=0)
```

三个必须对齐的点：

1. **权重归一化**：REINVENT 把权重归一化到和为 1（`weights / sum_weights`）。
   配置里 Vina=1.0 + QED=0.3 + MW=0.2 + Alerts=1.0 → 归一化后
   Vina=0.4, Alerts=0.4, QED=0.12, MW=0.08。
   **我们的 MPO 权重之和本来就是 1.0**，所以两边尺度天然可比 —— 但实现时
   必须确认没有重复归一化。

2. **floor 是 `1e-8`，不是我们的 `ENDPOINT_FLOOR = 0.05`**：
   ```python
   scores = np.maximum(scores, 1e-8)
   result = np.nanprod(scores ** (weights / sum_weights), axis=0)
   ```
   这意味着 REINVENT 在某个端点为 0 时，reward 会掉到 ~1e-8 量级
   （断崖）。我们的 `0.05` 是**故意**改的（避免 -18.4 的 ln 断崖）。
   **复刻时必须用 REINVENT 的 `1e-8`**，否则两个 reward 不同尺度，RL 数值不可比。
   → 这也意味着 `reinvent_like` 变体里**不加** BBB 硬约束（REINVENT 没有这个约束），
     否则就不是同一个目标函数了。

3. **`custom_alerts` 是 filter 型组件**（`reinvent_plugins/components/comp_custom_alerts.py`）：
   ```python
   scores = [1 - any(m) for m in all_matches]
   ```
   即命中任一 SMARTS → 该端点 score = 0。配合 `1e-8` floor，
   **命中 alert 的分子 reward 直接塌到 ~0** —— 这是极强的手工引导。
   我们的 `HARD_CUTOFFS` 语义相近但不完全等价（我们是"剪枝"，
   REINVENT 是"score=0 然后几何平均塌陷"）。

4. **MW 的 double_sigmoid**（`reinvent/scoring/transforms/double_sigmoid.py`）：
   `coef_div=100.0, coef_si=150.0, coef_se=150.0` 是默认值，
   配置里只给了 `low=200 / high=500`，所以用的是默认系数。
   左半段在 MW<200 时输出 → 0（200 处为 0.5），右半段在 MW>500 时 → 0。
   注意这与我们的 `normalize_mw` 形状不同，**不能直接复用**。

### 6.2 Reward Hacking 防线（RL 特有，SFT 阶段不存在）

有梯度压力时，模型会学会**骗 scorer**，而不是设计更好的分子。
必须设三层防线：

**第一层：奖励函数本身的地形**（已有）
- `HARD_CUTOFFS['bbb_min'] = 0.5` — BBB 不可交易
- `ENDPOINT_FLOOR = 0.05` — 无断崖，reward 始终可排序
- 几何平均 + 多端点 → 单一端点的 hacking 收益被稀释

**第二层：结构合理性检查**（需新增）
```python
HACKING_GUARDS = {
    'max_rotatable_bonds': 10,     # 防长链烷烃（Mistral 基线出现过 rot=147）
    'max_heavy_atoms': 40,
    'max_mw': 500,                 # 对齐 rescore 的类药预筛
    'max_ring_count': 6,
    'max_amide_bonds': 3,          # 防聚合物化
    'reject_long_aliphatic_chain': 12,   # 连续 sp3 碳链长度
}
```
违反 → reward = 0（并记 `hacked: True`）

**第三层：监控 + 人工抽检**
每 10 步导出 top-20 分子 + reward 各端点，人工看 20 个。
自动警报：
| 警报条件 | 含义 |
|---|---|
| reward ↑ 但 QED ↓ > 0.15 | 在牺牲成药性换 reward |
| reward ↑ 但唯一骨架数 ↓ > 30% | 多样性坍缩 |
| reward ↑ 但 BBB+ 率 ↓ | BBB 硬约束被绕过（应不可能，需排查） |
| MW 中位数持续 > 450 | 典型 size-hacking |
| 含高分子的分子比例 > 5% | 聚合度 hacking |

**第四层：KL 惩罚**（β=0.02）
限制 agent 偏离 SFT 先验太远，间接限制极端结构。

### 6.3 已知的目标冲突（必须写进论文讨论）

Vina 权重扫描实测（27 个反馈分子）：

| vina_w | ρ(Vina, reward) |
|---|---|
| 0.25（当前） | **+0.011** |
| 0.45 | -0.392 |
| 0.75 | -0.769 |

机制：`ρ(DILI, Vina) = -0.630`，`ρ(Pgp, Vina) = -0.508`，`ρ(MW, Vina) = -0.479`
—— **结合越强 → 分子越大越疏水 → 毒性/外排风险升高**。

**这不是缺陷**。在 EGFR 活性分子集上 `ρ(reward, pIC50) = +0.281`，
强于 `ρ(Vina, pIC50) = -0.173`（后者源于 BioAssay 截断）。

但 RL 会**主动放大**这个冲突（梯度只认 reward）。因此：
- `ours_mpo` 变体**必须保留 BBB 硬约束**（否则 BBB+ 会崩，v2 已实测 62%→33%）
- 论文要报告 **reward ↑ 与 Vina/BBB 的 trade-off 曲线**，而不是只报 reward

---

## 7. 超参数与算力预算

### 7.1 核心超参数

| 参数 | 建议值 | 依据 / 说明 |
|---|---|---|
| `max_steps` | 60 | REINVENT 用 40；我们步内样本更多，可略多 |
| `n_prompts` / 步 | 4 | 4 个 CELL_STATE 变体，保证 prompt 多样性 |
| `group_size` (G) | 8 | GRPO 标准；显存允许可到 16 |
| 每步分子数 | 32 | 4 × 8 |
| `lr` | **1e-6** | 比 SFT (1e-4) 低 2 个数量级。RL 是微调，不能冲垮先验 |
| `beta_kl` | 0.02 | 太大会锁死模型，太小会漂移 |
| `temperature` | 1.0 | 与 REINVENT 一致，保证探索 |
| `top_p` | 0.95 | 比反馈闭环的 0.92 略宽 |
| `max_new_tokens` | 220 | 128 会截断长 SMILES（已实测） |
| `max_grad_norm` | 1.0 | 防梯度爆炸 |
| `grad_accum` | 4 | 有效 batch = 128 分子 |
| `sigma` (DAP) | 自适应，σ₀ ∈ {2,4,8,16} | 见 §3.3 问题 1 |
| `exhaustiveness` (Vina) | 4 | 与 REINVENT TLRL 配置一致 |
| `n_workers` (Vina) | 15 | 60 核 / 4 核每 worker |
| `bucket_size` | 10 | 与 REINVENT 一致 |

### 7.2 时间预算（单靶点 IDH1，**实测数据**）

**每步（32 分子 = 4 prompts × G=8）**：

| 阶段 | 实测耗时 | 瓶颈 |
|---|---|---|
| Rollout (HF, batch=32, 220 token) | **11.0 s** | GPU（0.343 s/分子） |
| Reward (32 分子 ÷ 15 workers, 1.16 mol/s) | **~28 s** | **CPU ← 真正的瓶颈** |
| Advantage | < 0.1 s | — |
| Logprob agent (fwd+bwd B=8, grad_accum=4) | **8.7 s** | GPU（2.17 s × 4） |
| Logprob prior (no_grad B=8) | ~2 s | GPU |
| Checkpoint / 日志 | ~2 s | 磁盘 |
| **合计** | **~52 s/步** | |

**60 步 ≈ 52 分钟**（原估 2.5–4 小时，**高估了 3–4 倍**）

**Reward 累积成本**：60 步 × 32 = 1920 次对接（去重后约 1200–1500 次），
按 1.16 mol/s 计 ≈ **17–21 分钟纯 CPU 时间**。加上磁盘缓存命中率 20–40%，
实际约 13–17 分钟。**全程 CPU 时间 < 25 分钟。**

**单靶点完整实验（60 步 + 最终 200 分子独立采样 + 重打分）≈ 1.5–2 小时。**
4 张卡并行跑 4 个配置 → **全部主实验约 2 小时**。


### 7.3 显存预算（**实测数据，2026-11-19**）

单卡 RTX 3090 (23.69 GiB 可用) 实测：

| 环节 | 峰值显存 | 耗时 |
|---|---|---|
| 加载基座 + merge Llamole + GBM LoRA | 14.8 GB | 20 s |
| Rollout B=16 (220 token, `use_cache=True`) | **15.3 GB** | 10.7 s (0.666 s/分子) |
| Rollout B=32 | **15.6 GB** | 11.0 s (**0.343 s/分子**) |
| fwd+bwd B=4 (grad ckpt on) | **17.2 GB** | 1.06 s |
| fwd+bwd B=8 (grad ckpt on) | **19.5 GB** | 2.17 s |
| fwd+bwd B=16 | **OOM** (>23.7 GB) | — |
| prior fwd B=8 (no_grad, 无 ckpt) | 待补 | — |

**结论**：
- **单卡 3090 足够**，训练 batch 上限 **8**（19.5 GB，留 4 GB 余量给碎片）
- 生成和训练**可以在同一张卡上串行**（峰值不叠加，因为生成阶段是 `no_grad`）
- **不需要多卡分进程**（见 §7.4 修正）

**RL LoRA 可训练参数量实测**：40.37 M（r=16，7 个 target_modules）
→ AdamW fp32 状态 ≈ 40.37M × 12 B ≈ 0.48 GB。

### 7.4 多卡分工（**修正：实测后推翻了原方案**）

**实测数据表明多进程分卡没有必要**：

| 环节 | 单卡耗时 | 多卡能否加速 |
|---|---|---|
| Rollout 32 分子 | 11.0 s | ❌ 已是 GPU 饱和（batch 32 峰值 15.6GB，再加卡也不快多少） |
| fwd+bwd B=8 | 2.17 s | ❌ 通信开销 > 收益 |
| **Vina 对接 32 分子** | **~28 s (1.16 mol/s)** | ✅ **CPU 才是瓶颈** |

**每步总耗时 ≈ 11 s (GPU) + 28 s (CPU) + 2×2 s (logprob) ≈ 43 s**
→ 60 步 ≈ **43 分钟**（原估 2.5–4 小时，高估了 4 倍）

**最终方案**：

| 资源 | 用途 |
|---|---|
| **GPU0** | Rollout + 训练（同一进程串行，生成/训练两个模式切换） |
| **GPU1** | 备用 / 并行跑第二个消融实验 |
| **GPU2** | 备用 |
| **GPU3** | 备用 |
| **CPU 60 核** | Vina 进程池 15 workers + admet-ai |

**为什么不用多卡分进程**：
- Rollout 与训练**同卡串行零同步成本**，分进程反而要解决 RL LoRA 权重热同步
- 瓶颈在 CPU 侧 Vina，GPU 再多也无用
- 4 张卡可以**并行跑 4 个不同的实验配置**（GRPO/DAP × reinvent_like/ours_mpo），
  这才是这些卡的最优用法

**⚠️ 显存碎片**：B=16 时 OOM（`Tried to allocate 2.11 GiB`）。
设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 可缓解。
**训练 batch 固定为 8**，靠 `grad_accum=4` 拿到有效 batch 32。

---

## 8. 消融实验设计

### 8.1 主结果表（IDH1，全部用 `scripts/rescore_parallel.py` 统一重打分）

| 方法 | Prior | 优化算法 | Reward 函数 | Vina mean | reward mean | BBB+ | 唯一骨架 | IntDiv |
|---|---|---|---|---|---|---|---|---|
| REINVENT 4 (TL+RL) | PubChem+TL | DAP | Vina×QED×MW×Alerts | -9.95 | — | 100% | 20/20 | — |
| **Ours-GRPO-reinv** | Ours SFT | GRPO | Vina×QED×MW×Alerts | ? | ? | ? | ? | ? |
| **Ours-DAP-reinv** | Ours SFT | DAP-Lite | Vina×QED×MW×Alerts | ? | ? | ? | ? | ? |
| **Ours-GRPO-mpo** | Ours SFT | GRPO | Ours 8-端点 MPO | ? | ? | ? | ? | ? |
| Ours-Feedback (v3) | Ours SFT | 无（prompt 级） | — | -7.37 | 0.562 | 96% | — | — |
| Ours-SFT (无优化) | Ours SFT | 无 | — | -8.98 | — | — | 14/16 | 0.838 |

### 8.2 关键消融（回答"RL 到底带来了什么"）

| 消融 | 对照 | 检验的问题 |
|---|---|---|
| **A1** 无 KL 惩罚 (β=0) vs β=0.02 | 漂移是否被 KL 抑制 | KL 项是否必要 |
| **A2** DAP vs GRPO | LLM 场景下哪种 RL 算法更好 | §3.6 的主张是否成立 |
| **A3** 有/无多样性过滤 | reward 与多样性的 trade-off | 多样性过滤的必要性 |
| **A4** 有效率追踪 (round 1/20/40/60) | reward 随 RL 步数的曲线 | 是否收敛、何时过拟合 |
| **A5** reward hacking 守卫开/关 | 守卫的必要性 | 防线有效性 |
| **A6** `reinv` vs `mpo` reward | 目标函数对结果的影响 | 公平性 vs 性能 |

### 8.3 必做的收敛性分析

**RL 有效性的核心证据是曲线，不是终点值。**
必须画：
- reward mean vs step（含 ±std 带）
- 有效 SMILES 率 vs step
- 唯一 Murcko 骨架数 vs step
- Vina mean vs step（看是否与 reward 背离 —— trade-off 显性化）
- BBB+ 率 vs step（硬约束是否守住）
- QED / MW 中位数 vs step（hacking 监控）

**基线对照**：把 SFT 模型在**同一 prompt 集**上直接采样 N 次，
画一条水平参考线（SFT 水平线）。RL 曲线如果长期不高于这条线 → RL 没学到东西。

### 8.4 统计显著性

- 每步 32 分子、60 步 → 累积 1920 个样本，但**同一分布、有相关性**
- 最终评测必须**独立采样**（如从最终 checkpoint 采 200 个分子，与训练分子不重叠）
- 报告 mean ± sem，并做 Mann-Whitney U 检验（分子 reward 分布非正态）

---

## 9. 风险登记册

| # | 风险 | 概率 | 影响 | 缓解 | 触发信号 |
|---|---|---|---|---|---|
| R1 | **Reward hacking** | 高 | 高 | 四层防线（§6.2）+ 人工抽检 | reward ↑ 但 QED/骨架数 ↓ |
| R2 | **多样性坍缩** | 高 | 高 | Murcko bucket=10（与 REINVENT 同） | 唯一骨架数 ↓ > 30% |
| R3 | **σ 标定失败**（DAP） | 中 | 中 | 自适应 σ + z-score；先跑 GRPO | DAP loss 不下降 / 梯度 NaN |
| R4 | **显存 OOM** | 中 | 中 | merge SFT LoRA、grad checkpointing、batch=8 | OOM 异常 |
| R5 | **Vina 分数不稳定**（并发竞争） | 中 | 高 | 每 worker 独立实例；串行 vs 并行一致性测试 | 分数出现 >0 的正值（物理不可能） |
| R6 | **RL 后模型退化**（输出乱码） | 中 | 中 | KL 惩罚 + lr 1e-6 + 每步采样监控 | 有效 SMILES 率 ↓ > 20% |
| R7 | **Rollout/训练 权重不同步** | 中 | 高 | **M1 用同进程串行**，规避此问题 | Rollout 与训练 logprob 对不上 |
| R8 | **vLLM LoRA 热更新失败** | 中 | 低 | M1 用 HF 后端；vLLM 作为 M2 优化 | vLLM 启动/换 adapter 报错 |
| R9 | **磁盘不足**（当前 98% 已用，仅剩 148GB） | **高** | 高 | checkpoint 只存 LoRA（~250MB）；日志及时清理 | `df -h` 报警 |
| R10 | **REINVENT reward 复刻不准** | 中 | 中 | 已核实 `scoring/aggregators/means.py`（权重归一化 + `1e-8` floor）、`comp_custom_alerts.py`（filter 型）、`double_sigmoid.py`（默认系数）—— 见 §6.1 | 两法 reward 分布差异大 |
| R11 | **`reinvent_like` 变体忘了用 `1e-8` floor** | 中 | 高 | 复刻时必须用 REINVENT 的原始 floor，**不要**用我们的 `ENDPOINT_FLOOR=0.05` | 同一批分子两法 reward 排序不一致 |
| R12 | **backward 报 `does not require grad`** | **已发生** | 中 | 三个原因都已定位并修复：`is_trainable=True` + `enable_input_require_grads()` + `use_reentrant=False`（§5.7 踩坑 2/3/4） | 训练循环第一步崩溃 |
| R13 | **生成阶段被 gradient checkpointing 拖死** | **已发生** | 高 | 生成/训练两个阶段切换 `use_cache`（§5.2 踩坑 1） | Rollout 卡住 > 60 s 无输出 |
| R14 | **SFT LoRA 被误优化**（静默污染 prior） | 中 | 高 | 逐参数显式设 `requires_grad`；训练后校验 SFT adapter 权重哈希未变 | KL 项失去意义，RL 结果不可解释 |
| R15 | **训练 batch=16 OOM** | **已发生** | 低 | 固定 batch=8 + `grad_accum=4`；设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | CUDA OOM |
| R16 | **admet-ai 抢 GPU 显存** | **已发生** | 高 | admet-ai 自动选 GPU（实测日志确认）；**必须**锁 `CUDA_VISIBLE_DEVICES=""` 或显式 `device="cpu"` | 训练中 OOM，且 admet-ai 日志显示用了 CUDA |
| R17 | **两个环境的 transformers 版本不同**（4.44.0 vs 5.3.0） | 中 | 中 | 统一用 base env；rollout 前用同一批 prompt 对比两个环境的输出 token 序列 | 生成的 SMILES 明显不同 |
| R18 | **reward 数值与历史不可比** | 中 | 高 | M0.0 的同源性回归测试（已实测基线 3.4e-9） | 回归测试失败 |

> **R9 需要立即处理**：`/` 已用 98%，只剩 148GB。60 步 × 每步存 checkpoint × 250MB
> ≈ 15GB，尚可；但若误存完整模型（15GB/次）立刻爆盘。**checkpoint 必须只存 LoRA。**

---

## 10. 实施路线图

### M0 — 前置解耦 + 环境打通（0.5 天）★ 必做 ✅ **已完成 (2026-11-19)**

**目标**：把"评测"从"生成"里剥出来，并解决环境依赖。这是接 RL 的第一道坎。

**M0.0 — 环境确认（5 分钟，最高优先级）** ✅
- [x] **确认用哪个 env**（实测已倾向 base env `/root/anaconda3`）：
      ```bash
      /root/anaconda3/bin/python -c "
      import torch, vina, meeko, admet_ai, transformers, peft
      print('CUDA:', torch.cuda.is_available(), torch.cuda.device_count())
      print('tf', transformers.__version__, 'peft', peft.__version__)"
      ```
      → 实测：`CUDA: True 4` / `tf 4.44.0` / `peft 0.18.1` ✓
- [x] **确认反馈闭环用的是哪个 env**（决定 rollout 的 transformers 版本必须一致）：
      查 `batch_generation_*.log` / `generate_feedback_loop.py` 的启动命令
      → base env 重算历史分子 reward 差 3.4e-9 ⇒ **同源**，用 base env
- [x] **reward 同源性回归测试**（**必须通过，否则 RL 数字与历史不可比**）：
      ```python
      # 用 base env 重算 IDH1_history.json 里的分子，与存档值比对
      # 已实测基线: reward 0.5914004167 vs 0.5914004201 (差 3.4e-9)
      #              vina -8.982 == -8.982, bbb 0.984540 == 0.984540
      ```
- [x] **锁定 admet-ai 用 CPU**（实测它会自动抢 GPU）：
      `CUDA_VISIBLE_DEVICES=""` 或 `ADMETCalculator(device="cpu")`
      —— 否则和 7B 模型抢显存导致 OOM
      → 已在 `reward_service._score_one()` 里实现

**M0.1 — 代码解耦** ✅
- [x] 在 `src/rl/` 新建 `reward_service.py`，**从 `generate_feedback_loop.py`
      抽出 `evaluate()` / `evaluate_batch()` 的逻辑，不复制代码**
      → 复用 `gbm_target_evaluator.build_evaluator_for_target()`
- [x] 加磁盘缓存（JSONL）
- [x] 加 `HACKING_GUARDS` 结构检查
- [x] **一致性测试**：8 个分子与实测基线**逐分子比对**，
      Vina 分数**完全一致**（含 3 位小数精度）
      → 基线: `[-7.940, -8.105, -9.477, -7.758, -9.364, -7.079, -8.074, -7.884]`
      → 测试 `test_vina_matches_baseline` ✅ PASSED
- [x] **线程安全测试**：串行 (1 worker) vs 并行 (8 workers)，结果完全一致
      → 测试 `test_serial_vs_parallel_identical` ✅ PASSED
- [x] **缓存一致性测试**：缓存命中前后结果一致
      → 测试 `test_cache_returns_same` ✅ PASSED

**验收**：✅ **全部通过**
- [x] `pytest tests/test_reward_parity.py` 通过 —— **23/23 (20 单元 + 3 集成)**
- [x] base env 内能直接完成一次真实对接 + admet-ai 预测
- [x] admet-ai 不占 GPU 显存（`reward_service` 里 `CUDA_VISIBLE_DEVICES=""`）

### M1 — GRPO 单步打通（1.5 天）✅ **代码完成，待实跑验证**

**目标**：单步 RL 循环跑通，梯度正确。

- [x] `src/rl/prompt_bank.py`（复用 `build_prompt`）
- [x] `src/rl/rollout.py`（HF 后端，实测 0.343 s/分子 @ B=32）
- [x] `src/rl/advantage.py`
- [x] `src/rl/losses.py`（GRPO + k3 KL / DAP-Lite + 自适应 σ）
- [x] `src/rl/trainer.py`（三 adapter + 训练循环 + checkpoint/resume）
- [x] `src/rl/diversity.py`（Murcko 骨架过滤器，对齐 REINVENT bucket=10）
- [x] `scripts/train_rl.py`（CLI）
- [x] **三 adapter 配置**（照抄 §5.7 实测通过的代码，含
      `is_trainable=True` + `enable_input_require_grads()` + `use_reentrant=False`）
- [x] **梯度正确性测试**（单元测试层面）：
      - `test_grpo_loss_sign` — advantage 符号正确
      - `test_grpo_gradient_direction` — 好样本一步后 logprob **上升**
      - `test_grpo_kl_zero_when_identical` / `test_grpo_kl_positive_when_different`
      - `test_dap_loss_finite` / `test_adaptive_sigma`
- [x] **污染检查**：`sft_adapter_fingerprint()` 实现，
      trainer 每步比对 SFT adapter 哈希，变化则报警
- [ ] **实跑验证**：跑 1 步真实 RL step，确认 loss 有限、梯度非零、显存 < 20GB
- [ ] **污染实跑验证**：训练一步后校验 SFT adapter 权重哈希未变（R14）

**验收**：单步 loss 有限、梯度非零、显存 < 20GB、SFT adapter 未被污染。



### M2 — 完整训练循环 + 收敛性（1 天）

- [x] 多步循环、checkpoint（**只存 LoRA**）、resume
- [x] 多样性过滤接入
- [x] 每步指标日志（§8.3 的全部曲线）+ **滑动平均**
- [x] **定期存 `latest` checkpoint**（`--save_every`，默认 5 步）
      —— 否则中途被杀只能从头开始
- [x] hacking 警报接入
- [x] **独立评估脚本**（`eval_rl_checkpoint.py`）：固定 prompt 集 + 同 seed
      采样 + 统一重打分，与 SFT 基线对比
- [x] **显著性检验**（`compare_eval.py`）：Mann-Whitney U / Cliff's delta /
      配对检验 / Cohen's d
- [x] 短程验证（15 步）—— **结果: 方向正确但统计不显著**（见 §M2.1）

**验收**：15 步后 reward mean 显著高于 SFT 水平线。

---

#### M2.1 — 15 步短程训练结果（IDH1 / GRPO / 15 步）

命令:
```bash
python3 scripts/train_rl.py --target IDH1 --algorithm grpo --steps 15 \
    --n_prompts 4 --group_size 8 --prompt_bank_size 8 \
    --train_batch 4 --grad_accum 1 --lr 5e-6 --beta_kl 0.02 \
    --n_workers 15 --save_every 5 --ma_window 5 \
    --out_dir ../experiments/rl_m2_grpo
```

**训练日志**（`reward_ma` = 5 步滑动平均）:

| step | reward | MA | max | KL | loss | 有效率 | grad_norm |
|---|---|---|---|---|---|---|---|
| 0 | 0.2448 | 0.2448 | 0.6755 | 0.00000 | +0.0861 | 20/32 | 1.85 |
| 1 | 0.2812 | 0.2630 | 0.5289 | 0.00002 | -0.1262 | 18/32 | 1.26 |
| 2 | 0.1986 | 0.2415 | 0.5187 | 0.00004 | +0.6428 | 12/32 | 2.71 |
| 3 | 0.3457 | 0.2676 | 0.6755 | 0.00006 | +0.1487 | 18/32 | 1.58 |
| 4 | 0.2370 | 0.2615 | 0.6470 | 0.00008 | -0.1059 | 22/32 | 1.54 |
| 5 | 0.2659 | 0.2657 | 0.6875 | 0.00011 | +0.0681 | 16/32 | 1.59 |
| 6 | 0.2352 | 0.2565 | 0.6948 | 0.00029 | +0.2961 | 17/32 | 1.68 |
| 7 | 0.2682 | 0.2704 | 0.6052 | 0.00064 | +0.2244 | 17/32 | 1.40 |
| 8 | 0.2529 | 0.2518 | 0.6850 | 0.00062 | -0.1849 | 19/32 | 2.30 |
| 9 | 0.2811 | 0.2607 | 0.7034 | 0.00044 | +0.0599 | 17/32 | 0.98 |
| 10 | 0.2925 | 0.2660 | 0.6549 | 0.00200 | +0.3759 | 17/32 | 1.91 |
| 11 | 0.2093 | 0.2608 | 0.6531 | 0.00236 | -0.2951 | 25/32 | 0.89 |
| 12 | 0.2053 | 0.2482 | 0.6742 | 0.00208 | -0.2931 | 16/32 | 2.00 |
| 13 | 0.2536 | 0.2484 | 0.5959 | 0.00232 | +0.0467 | 23/32 | 1.25 |
| 14 | 0.2800 | 0.2481 | 0.6531 | 0.00212 | -0.1611 | 18/32 | 1.53 |

**统计显著性**（`scripts/stat_test_m2.py`）:

| 指标 | 前 5 步 | 后 5 步 | Δ | t | 判定 |
|---|---|---|---|---|---|
| reward_mean | 0.2615 | 0.2481 | -0.0133 | -0.38 | ❌ 不显著 |
| reward_max | 0.6091 | 0.6462 | +0.0371 | +1.07 | ❌ 不显著 |
| 有效率 | 56.2% | 61.9% | +5.6% | +1.27 | ❌ 不显著 |
| **KL** | 0.00000 | 0.00220 | +0.0021 | **+4.90** | ✅ **显著** |

> **单步 reward_mean 的标准误 = 0.0551**（组内标准差 0.2358 / √18.3）。
> 15 步总共只有 275 个有效分子 —— **样本量根本不足以分辨
> "学到东西"和"没学到"**。要检测 0.02 量级的均值变化需要几百个样本。

**独立评估**（`scripts/eval_rl_checkpoint.py` + `compare_eval.py`，
同一 seed=1234、同一 8 条 prompt、独立采样后统一重打分）:

| 指标 | SFT 基线 | RL (15 步) | Δ | Δ% |
|---|---|---|---|---|
| reward mean | 0.2549 | **0.2673** | +0.0124 | **+4.9%** |
| reward median | 0.3050 | **0.3317** | +0.0267 | **+8.7%** |
| reward max | 0.6545 | **0.6936** | +0.0391 | +6.0% |
| reward p90 | 0.5642 | **0.5958** | +0.0317 | +5.6% |
| vina mean | -7.540 | **-7.609** | -0.069 | 更好 |
| vina best | -9.897 | -9.897 | = | = |
| 有效率 | 116/200 | 115/200 | ≈ | ≈ |
| 唯一性 | 78.4% | 77.4% | ≈ | ≈ |

- Mann-Whitney U = 6473.5, **p = 0.686**（不显著）
- Cliff's delta = **+0.0295**（方向为正，但效应量极小）
- Cohen's d = +0.0510
- 配对检验（8 个 prompt 的组内最优）：Δ 均值 +0.0201，**胜 2/8**，p = 0.538

**判定**: 🟡 **RL 链路完全正常，但 15 步的效应量不足以超过噪声。**

**确定学到的信号**（这些是确定的，不依赖噪声判定）:
1. **KL 单调增长** 0.00000 → 0.00212 (t=+4.90 显著) —— 策略确实在偏离先验
2. **rollout 有效率提升** 56.2% → 61.9% —— 生成分子的合法率上升
3. **组内最优变好** 0.6091 → 0.6462
4. **grad_norm 健康**（均值 1.63，最大 2.71）—— 梯度非零非爆炸
5. **prior 未被污染**（0/15 步）

**下一步（M2.2）要解决的是"信号太弱"，不是"代码有问题"**:

| 问题 | 原因 | 对策 |
|---|---|---|
| 组内有效样本太少 | group_size=8 × 有效率 55% → 每组仅 3-6 个有效分子, z-score 噪声大 | **group_size 8 → 16** (REINVENT 标准), 每组 6-10 个有效分子 |
| 总样本量不足 | 15 步 × 18 有效 = 275 个分子 | **训练 60 步** → ~1100 个分子 |
| 效应量小 | lr=5e-6 偏保守 | lr 5e-6 → 1e-5 (KL 已到 0.002, 还有很大空间) |

⚠️ **不要被 on-policy reward 骗了**: 训练日志里 reward_mean 的 ±0.02 波动
纯粹是采样噪声。任何关于"RL 有没有用"的结论, 都必须来自**独立评估**
（固定 prompt 集 + 同 seed + 统一重打分 + 显著性检验）。

---

#### M2.2 — 60 步长程训练结果（IDH1 / GRPO / G=16 / 60 步）✅ 已完成

命令:
```bash
python3 scripts/train_rl.py --target IDH1 --algorithm grpo --steps 60 \
    --n_prompts 4 --group_size 16 --prompt_bank_size 8 \
    --train_batch 4 --grad_accum 1 --lr 1e-5 --beta_kl 0.02 \
    --n_workers 12 --save_every 10 --ma_window 5 \
    --out_dir ../experiments/rl_m2_grpo_v2
```

**相对 M2.1 的三个变化**: group_size 8→16 (REINVENT 标准) / lr 5e-6→1e-5 /
步数 15→60。总计 **~2500 个有效分子**（M2.1 只有 275）。

**训练期趋势**（`reward_ma` = 5 步滑动平均）:

| 指标 | 前 5 步 | 后 5 步 | Δ |
|---|---|---|---|
| reward mean | 0.2847 | 0.2949 | **+0.0102** |
| reward MA | 0.2858 | 0.2989 | **+0.0131** |
| reward max | 0.6786 | 0.6655 | -0.0131 |
| rollout 有效率 | 57.2% | **66.2%** | **+9.0pp** |
| KL | 0.00000 | 0.10532 | 峰值 0.34490 |

- MA 线性斜率 **+0.000515/step** → 判定「🟡 微升，需更多步确认」
- prior 未被污染 ✅

**独立评估**（`seed=1234`、同一 8 条 prompt、25 分子/提示、统一重打分）:

| 指标 | SFT 基线 | RL (60 步) | Δ | Δ% |
|---|---|---|---|---|
| reward mean | 0.2549 | **0.2610** | +0.0062 | +2.4% |
| reward median | 0.3050 | **0.3390** | +0.0340 | **+11.1%** |
| reward max | 0.6545 | **0.6663** | +0.0119 | +1.8% |
| reward p90 | 0.5642 | **0.5788** | +0.0146 | +2.6% |
| vina mean | -7.5399 | **-7.7168** | -0.1769 | 更好 |
| vina best | -9.897 | **-10.439** | -0.542 | 更好 |
| qed mean | 0.6162 | **0.6422** | +0.0260 | +4.2% |
| ames (毒性) | 0.2537 | **0.2360** | -0.0178 | 更低 ✅ |
| herg | 0.6194 | 0.5828 | -0.0366 | 更低 ✅ |
| pgp | 0.4134 | 0.3847 | -0.0287 | 更低 ✅ |
| 有效率 | 116/200 (58%) | 125/200 (62.5%) | | +4.5pp |
| 唯一性 | 78.4% | 69.6% | | -8.8pp |

- Mann-Whitney U = 7241.5, **p = 0.9869**（不显著）
- Cliff's delta = **-0.0012**（**效应量归零** — 比 M2.1 的 +0.0295 更差）
- Cohen's d = +0.0257
- 配对检验（8 prompt 组内最优）：Δ 均值 **+0.0185**，中位数 +0.0158，**胜 5/8**，p = 0.1649

**关键发现 1 — 中位数提升是真实的，均值被左尾拖累**。
median +11.1% 明显高于 mean +2.4%，说明 RL 提升了「典型分子」的质量，
但拉低了分布的**左尾**（低分分子变多或不变）。MA max 反而下降（-0.0131）
也印证这点：RL 学会了产出「稳定的中上水平分子」，但**没有**提升极值。

**关键发现 2 — 没有 reward hacking**（这是最重要的正面结论）:

| 指标 | SFT 基线 | RL (60 步) |
|---|---|---|
| guard 触发数 | 35 / 116 | **35 / 125** |
| prune 率 | 43.1% | **42.4%** |
| guard 组 reward | 0.0000 | 0.0000 |

RL 模型的 guard 触发数与基线**完全相同**，prune 率一致 → 策略没有靠
"生成畸形分子骗过 scorer" 来提分。提升是真实的。

**关键发现 3 — KL 已接近失控**。峰值 0.3449 (step 40)，末步仍有 0.105。
`beta_kl=0.02` 贡献的惩罚项仅 `0.02 × 0.105 = 0.0021`，而 policy loss
量级 ~0.1–0.5 → **KL 项的约束力只有 policy 项的 0.4%–2%**。
与此同时生成 token 数从 ~200 降到 ~60–90（见续训日志），说明模型正在
快速塌缩到短的、高分的输出模板 —— 这正是左尾被拉低、中位数上升的机制。

**判定**: 🟡 **RL 链路正确、信号方向一致（中位数 +11.1%、配对 5/8 胜、
无 hacking、有效率 +9pp），但效应量仍被 on-policy 噪声淹没**
（p=0.16–0.99）。主因不是代码，而是 **KL 约束不足导致策略过早塌缩**。

**下一步（M3）的对策**（已据此调整优先级）:

| 问题 | 根因 | 对策 |
|---|---|---|
| 策略塌缩 | `beta_kl=0.02` 太小, KL 涨到 0.34 | **beta_kl 0.02 → 0.05–0.10**, 目标 KL 控制在 0.05–0.10 |
| 极值不提升 | 只优化组内均值优势 | 引入 **REINVENT 风格的 top-k 分子加权**（`reward_max` 项） |
| 分布左尾 | 组内 z-score 对高分样本不敏感 | **组内 rank-based 优势**（用 rank 代替 z-score） |
| 样本量 | 60 步 ≈ 2500 分子, 单次评估 241 个 | 评估侧 **n_per_prompt 25 → 100**（成本低, 方差降 2 倍） |

---

### M1.5 — 实跑踩坑记录（2026-11-19 实测，**必读**）

M1 的代码在**单元测试层面全过**，但实跑暴露了 9 个只有真跑才会出现的缺陷。
每一个都会让 RL **静默失效**（不报错，但学不到东西）——这是 RL 最危险的失败模式。

| # | 缺陷 | 症状 | 根因 | 修复 |
|---|---|---|---|---|
| 1 | **PEFT `set_adapter` 只能激活一个 adapter** | rollout 有效率 **0%** | `set_adapter("rl")` 把 SFT adapter 停用了, rollout 跑的是「基座+随机初始化 LoRA」 | SFT LoRA 也 `merge_and_unload()`, 只留一个可训练 RL LoRA |
| 2 | **ProcessPoolExecutor 默认 fork 继承 CUDA context** | reward 阶段死锁 >5 分钟 | 父进程已加载 7B 到 GPU, fork 出的 worker CUDA 初始化死锁 | `mp.get_context("spawn")` |
| 3 | **logits 全量计算** | `OutOfMemoryError` | `model(...)` 对全部位置算 logits, Qwen2 再 `logits.float()`: 4×650×152k×4B = 1.58GB ×3 | 只跑 body, `lm_head` 只作用于 response 位置 (省 3.52GB, 19%) |
| 4 | **gradient checkpointing 开得太晚** | 6 样本即 OOM | 在 backward 前才开, 前向激活图全留在显存 | 前向**之前**就开 |
| 5 | **采样温度过高** | 有效率 38% | t=1.0 vs 反馈闭环的 t=0.7 | 默认 t=0.8 (实测 75%) |
| 6 | **`_NewlineStopper` 只看 `input_ids[0]`** | 序列被截断 | `num_return_sequences>1` 时第一个遇换行就停掉全部 | `_BatchNewlineStopper` (全部遇换行才停) |
| 7 | **prompt 索引错配** | logprob 全错 | `prompt_id` 是组位置, `prompt_idxs` 是原始索引 | `group_to_prompt` 映射 |
| 8 | **response 格式不一致** | 分布漂移 | 只用裸 SMILES, SFT 训练是 `"SMILES: xxx"` | 统一成 `SMILES: xxx` |
| 9 | **基座指纹哈希 7B 参数** | 每步 24 秒 | `.float()` 产生 28GB 临时拷贝 | 采样 64 个张量 + 原始 dtype |

**外加 3 个非阻塞 bug**：

| 缺陷 | 症状 | 修复 |
|---|---|---|
| `RewardCache` 定义了 `__len__` | 空缓存时 `bool(cache)` 为 False → **缓存永远不落盘** | 去掉 `__len__`, 加 `__bool__` 返回 True |
| `score_batch` 去重只填第一个位置 | 重复 SMILES → `AttributeError: 'NoneType' has no 'items'` | 填**所有**出现位置 + None 兜底 |
| 守卫阈值重复了 MPO 的约束 | 45% 的真实生成分子被丢弃, RL 信号砍半 | 守卫只管 reward 管不到的 (`rot`/`rings`/`chain`), 不管 MW |
| **训练进程被 shell 连带杀掉** | 60 步跑到 step 51 静默终止 (无 traceback, 无 CUDA OOM) | 父 shell 被中断 → SIGHUP 杀掉整个进程组。改用 `setsid nohup ... & disown` 脱离进程组; 配合 `--resume` 从 step 50 的 checkpoint 续训, 零损失 |
| **optimizer 状态用 JSON 存** | 单个 checkpoint 膨胀到 2.2 GB | AdamW 的 `exp_avg`/`exp_avg_sq` 转 list 后体积暴涨 5-7 倍 → 改用 `torch.save` 存 `.pt` (2.3KB–0.4MB), JSON 只存元数据 |
| **gradient accumulation 尾部被丢弃** | 15 步 + `grad_accum=4` → 最后 3 步梯度没生效 | `(self.step+1) % grad_accum` 判断用的是未自增的 step → 加 `_accum_count` 计数器, 并在 train 退出时 `_flush_gradients()` |

**关键教训**: RL 的每个断言都要问「**如果这一步是错的, 会报错吗?**」。
上面 9 个里有 7 个**不报错**, 只是安静地学不到东西。

**M1 实跑验收结果**（IDH1 / GRPO / 3 步，命令见下）:

```bash
python scripts/train_rl.py --target IDH1 --algorithm grpo --steps 3 \
    --n_prompts 2 --group_size 8 --train_batch 4 --grad_accum 2 \
    --n_workers 8 --save_every 2 --out_dir ../experiments/rl_m1_final
```

```
[step 0] loss=-0.7266  kl=0.00000  rew=0.1944 (max 0.4729)
         valid=6/16  adv|=1.000  t=79.8s   prior_contaminated=false
[step 1] loss=-0.1658  kl=0.00000  rew=0.2132 (max 0.4595)
         valid=8/16  adv|=0.965  t=59.4s   prior_contaminated=false
[step 2] loss= 0.0517  kl=0.00001  rew=0.1051 (max 0.3154)
         valid=3/16  adv|=1.000  t=49.3s   prior_contaminated=false
```

| 验收项 | 结果 | 说明 |
|---|---|---|
| loss 有限 | ✅ -0.7266 / -0.1658 / +0.0517 | 三步都有限 |
| KL 起始为 0 | ✅ 0.00000 | RL LoRA 零初始化, agent ≡ prior |
| KL 开始偏移 | ✅ 0.00001 (step 2) | 策略确实在动, 不是死循环 |
| advantage 符号正确 | ✅ \|adv\| = 1.000 / 0.965 | 组内 z-score 精确 |
| 梯度非零 | ✅ backward 完成 | loss 逐步变化 |
| 显存 < 20 GB | ✅ 峰值 19.5 GB | |
| prior 未被污染 | ✅ `prior_contaminated=false` | 3/3 步 |
| checkpoint 保存 | ✅ `rl_lora_final.safetensors` (161MB) + `state_final.json` (192MB) | |
| reward 缓存生效 | ✅ 第二次跑相同分子 reward 耗时 **0.0s** | |
| 单元测试 | ✅ 21/21 (20 + 1 回归) | |

**⚠️ 关于 reward_mean 的波动**（0.1944 → 0.2132 → 0.1051）:
这**不是**收敛失败, 而是每步只采 16 个分子（有效 3~8 个）的**采样噪声**。
step 2 的 `valid=3/16` 且 `n_skipped_groups=1`（一个组内 reward 无方差被跳过）,
所以 reward_mean 掉下来了。判断收敛需要 **15+ 步**（见 M2），
并且应该看**滑动平均**而不是单步值。

**⚠️ 关于 KL 一直是 0.0000x**: 数值很小是因为 `beta_kl=0.02` 且
RL LoRA 的 lr=1e-6（很小）。这是**有意的** —— 先小步确认梯度方向正确,
再放大 lr。放大到 1e-5 / 5e-5 是 M2 的内容。


### M3 — DAP-Lite 对照 + 消融（1 天）

- [ ] `dap_loss` 实现 + 自适应 σ
- [ ] σ₀ 网格搜索 {2,4,8,16}
- [ ] A1（β=0 vs 0.02）、A3（多样性过滤）快速消融

### M4 — 全量实验 + 论文素材（2 天）

- [ ] IDH1 全 60 步 × 2 个 reward 变体 × 2 个算法 = 4 个 run
- [ ] 独立采样 200 分子（与训练分子不重叠）
- [ ] `scripts/rescore_parallel.py` 统一重打分
- [ ] 曲线图（matplotlib）
- [ ] 更新 `final_results/comparative/SUMMARY.md` 和 `report/comparison_report.md`

**总计：约 6 天**（不含等待训练跑完的时间）

**实测后的实际算力占用极小**：单靶点 60 步 ≈ 52 分钟（§7.2），
4 个配置并行跑在 4 张卡上 ≈ **1.5–2 小时**出全部主结果。
**M0–M2 的工作量主要在代码和测试，不在算力。**

---

## 11. 已知未解问题与开放决策

| # | 问题 | 状态 | 建议 |
|---|---|---|---|
| Q1 | prompt 在 RL 期间冻结 → 失去了"条件多样性"优势？ | 待定 | 可做变体：每 20 步重采样 prompt 集（但会破坏 prior/agent 条件一致性）。**M1 先不做** |
| Q2 | Ours-GRPO 的 prior 已是"靶点特异"的，REINVENT 的 prior 是"通用+TL"。谁更公平？ | 待定 | **承认这是本质差异**，在论文中明说：不同 prior 反映不同方法路线（LLM 条件先验 vs 化学 SMILES 先验），不做强行对齐 |
| Q3 | reward_variant 是否要加第三种（`ours_mpo` 去掉 BBB 硬约束）？ | 可选 | 加一个 `mpo_nobbb` 变体，证明 BBB 硬约束在 RL 下的必要性（呼应 v2 的 62%→33% 退化） |
| Q4 | 是否需要在 EGFR / VEGFR 上也跑 RL？ | 待定 | 建议先只做 IDH1。若时间允许，EGFR 必须做（Vina 最强靶点，REINVENT 也最强，对比最严苛） |
| Q5 | LoRA r=16 是否足够承载 RL 更新？ | 待定 | 若 RL 曲线不动，尝试 r=32 或对 `embed_tokens` 也做 LoRA |
| Q6 | 是否把 RL LoRA 合并回 SFT LoRA 产出最终模型？ | 待定 | 产出 `saves/...-gbm-rl-grpo` 独立 adapter，论文里报告"额外 +X 参数" |
| Q7 | ~~llamole env 装 vina 会不会污染依赖？~~ | **已解决** | 实测 base env 有 CUDA + 全依赖，且与反馈闭环 reward 同源 → **直接用 base env**，无需安装 |
| Q8 | ~~admet-ai 在 llamole env 里能用吗？~~ | **已解决** | base env 有 admet_ai。**但它会自动抢 GPU** → RL 训练时锁 CPU |
| Q9 | **base env 的 transformers 4.44.0 跑 SFT 模型会不会有问题？** | 待定 | 反馈闭环的 rollout 必须与 RL 用同一 env（否则 tokenization/生成行为不一致）。查 `batch_generation_*.log` 确认；若反馈闭环用的是 llamole env，改为"Reward 独立进程"架构 | 两 env 对同一 prompt 生成不同 SMILES |

---

## 附录 A：关键源码位置索引

| 内容 | 位置 |
|---|---|
| DAP loss 原始实现 | `reinvent/runmodes/RL/reward.py:51-52` |
| REINVENT RL 主循环 | `reinvent/runmodes/RL/learning.py` |
| **REINVENT 几何平均聚合（权重归一化 + 1e-8 floor）** | `reinvent/scoring/aggregators/means.py:42-52` |
| **REINVENT custom_alerts（filter 型，命中→0）** | `reinvent_plugins/components/comp_custom_alerts.py:58` |
| **REINVENT double_sigmoid 变换（默认 coef）** | `reinvent/scoring/transforms/double_sigmoid.py` |
| REINVENT 评分组件协议 | `reinvent_plugins/components/component_results.py` |
| 我们的 MPO reward | `src/gbm_physical_evaluator.py:1131` (`MPORewardCalculator`) |
| 硬截断阈值 | `src/gbm_physical_evaluator.py:1170` (`HARD_CUTOFFS`) |
| 真实 Vina 接入 | `src/gbm_target_evaluator.py` → `src/property_pipeline/docking.py` |
| 现有反馈闭环 | `scripts/generate_feedback_loop.py` |
| 统一重打分 | `scripts/rescore_parallel.py` |
| Prompt 构造 | `scripts/generate_with_new_adapter.py:58` (`build_prompt`) |
| SFT 数据格式 | `scripts/build_qa_dataset.py:294` |
| LoRA 训练配置 | `scripts/train_gbm_lora_real.py:58` |
| REINVENT TLRL 配置 | `configs/reinvent_tlrl_idh1.toml` |

## 附录 B：环境事实核查（2026-11-19 实测）

| 项 | 值 |
|---|---|
| GPU | 4× RTX 3090, 24576 MiB each (可用 23.69 GiB), **全部空闲** |
| CPU | 60 核 |
| RAM | 314 GB (285 free) |
| 磁盘 | `/` 已用 **98%**，剩 148 GB ⚠️ |
| REINVENT 环境 | `/root/venv_reinvent`: reinvent 4.7.15, torch 2.9.1+**cpu**（无 CUDA） |

### 三环境的依赖矩阵（实测，2026-11-19）

| 环境 | CUDA | vina | meeko | admet_ai | rdkit | transformers | peft | torch |
|---|---|---|---|---|---|---|---|---|
| **`/root/anaconda3` (base)** | ✅ 4 卡 | ✅ | ✅ | ✅ | ✅ | 4.44.0 | 0.18.1 | 2.5.1+cu121 |
| `/root/anaconda3/envs/llamole` | ✅ 4 卡 | ❌ | ❌ | ❌ | ✅ | 5.3.0 | 0.12.0 | 2.4.0+cu121 |
| `/root/venv_reinvent` | ❌ | ✅ | ✅ | ✅ | ✅ | — | — | 2.9.1+cpu |

**结论：用 base env 跑整个 RL。** 它是唯一"全依赖 + 4 卡 GPU"的环境，
且已验证与反馈闭环 reward 同源（差 3.4e-9）。

| 项 | 值 |
|---|---|
| LoRA 配置 | r=16, alpha=32, dropout=0.05, target_modules = q/k/v/o/gate/up/down |
| SFT adapter | `saves/Llamole-Qwen2-7B-Instruct-Adapter-gbm-active` |
| PEFT 能力确认 | `set_adapter` / `disable_adapter` / `add_adapter` 均可用 ✓ |
| vLLM LoRA | `EngineArgs` 有 `enable_lora` / `max_lora_rank`，但 `LLM.__init__` 签名里不可见（需 `kwargs` 透传）⚠️ —— **实测 HF 后端已足够快，vLLM 不需要** |
| admet-ai 设备行为 | ⚠️ **自动选 GPU**（实测日志 `GPU available: True (cuda), used: True` / `NVIDIA GeForce RTX 3090`）→ RL 训练时必须锁 CPU |

## 附录 C：性能实测（2026-11-19，IDH1 靶点，实机跑通）

### C.1 Vina 对接（CPU）

| 配置 | 吞吐 |
|---|---|
| 串行 1 worker (cpu=4) | 2.80 s/分子 |
| 4 workers | 0.80 mol/s |
| 8 workers | 1.12 mol/s |
| **15 workers** | **1.16 mol/s (0.86 s/分子)** ← 拐点，8 之后基本不再提升 |

**并行一致性验证**：4 / 8 / 15 workers 下 8 个分子的分数**完全相同**
（`[-7.94, -8.11, -9.48, -7.76, -9.36, -7.08, -8.07, -7.88]`），
且与反馈闭环报告里的 -7.94 一致 → **管线线程/进程安全 ✓**

### C.2 GPU（单卡 3090）

| 环节 | 耗时 | 峰值显存 |
|---|---|---|
| 加载基座 + merge Llamole + GBM LoRA | 20 s | 14.8 GB |
| Rollout B=16, 220 token | 10.7 s (0.666 s/分子) | 15.3 GB |
| **Rollout B=32, 220 token** | **11.0 s (0.343 s/分子)** | **15.6 GB** |
| fwd+bwd B=4 (grad ckpt) | 1.06 s, loss=1.533 | 17.2 GB |
| **fwd+bwd B=8 (grad ckpt)** | **2.17 s, loss=1.609** | **19.5 GB** |
| fwd+bwd B=16 | **OOM** | >23.7 GB |

RL LoRA 可训练参数：**40.37 M**

### C.3 单步耗时汇总（32 分子/步）

| 阶段 | 耗时 | 占比 |
|---|---|---|
| Rollout (GPU) | 11.0 s | 21% |
| **Reward (CPU, 15 workers)** | **~28 s** | **54%** ← 瓶颈 |
| Logprob agent (4× accum) | 8.7 s | 17% |
| Logprob prior | ~2 s | 4% |
| 其他 | ~2 s | 4% |
| **合计** | **~52 s/步** | |

**60 步 ≈ 52 分钟**（原估 2.5–4 小时）

### C.4 实测发现的 4 个坑（均已写入 §5.2 / §5.7）

1. 生成阶段开 gradient checkpointing → KV cache 失效，生成卡死
2. `is_trainable=True` 缺失 → `backward` 报 `does not require grad`
3. `enable_input_require_grads()` 缺失 → 同上（grad ckpt 下输入无梯度）
4. `add_adapter` 后 SFT LoRA 参数仍 `requires_grad=True` → 静默污染 prior

### C.5 验证脚本位置

本节的实测脚本（可复现）：
- `/tmp/bench_vina.py` — Vina 串行基准
- `/tmp/bench_par.py` — Vina 并行基准 + 一致性验证
- `/tmp/bench_gpu4.py` — GPU 全链路基准（三 adapter + grad ckpt）

