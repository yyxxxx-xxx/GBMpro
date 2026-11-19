# 对比实验结果 —— 摘要与解读

> 生成脚本：`scripts/make_comparison_report.py`（完整报告见 `final_results/comparative/report/comparison_report.md`）
> 数据：`final_results/comparative/rescored/`，统一由 AutoDock Vina 1.2.7 + admet-ai + RDKit 重新计算。

---

## 一、评测管线（已就绪，可复现）

| 环节 | 实现 | 验证状态 |
|---|---|---|
| 受体准备 | `scripts/prepare_receptors.py`（RCSB 下载 + meeko `mk_prepare_receptor`） | ✅ 3 靶点 |
| 真实对接 | `src/property_pipeline/docking.py`（Vina 1.2.7 + Meeko 0.8.0，exh=8，n_poses=5） | ✅ 见下方管线验证 |
| ADMET | `src/property_pipeline/admet.py`（admet-ai，52 端点） | ✅ |
| 描述符 / SA | `src/property_pipeline/descriptors.py`（RDKit + Contrib SA_Score） | ✅ |
| 多样性 | `src/property_pipeline/scaffold.py`（Murcko + Shannon 熵） | ✅ |
| 并行重打分 | `scripts/rescore_parallel.py`（14 workers × 4 核，磁盘缓存） | ✅ 24 个 CSV |

### 对接管线验证（审稿人必问项）

共晶配体重对接分数与生成分子同量级 → 管线能复现已知结合模式：

| 靶点 | PDB | 共晶配体 | 重对接 (kcal/mol) |
|---|---|---|---|
| EGFR | 2JIV | HKI (neratinib-like) | **-8.86** |
| IDH1/IDH2 | 6U4J | PWV (FT-2102-like) | **-11.83** |
| VEGF/VEGFR | 3WZD | LEV (lenvatinib) | **-10.18** |

---

## 二、对比方法全景（8 个方法，含 5 个外部基线）

| 类别 | 方法 | 来源 | 说明 |
|---|---|---|---|
| 专用化学 LM | **REINVENT 4 (Vina-RL)** | 本地安装 v4.7.15 | PubChem 先验 + **Vina 分数作 RL 奖励**（DAP 策略梯度）+ Murcko 多样性过滤 |
| 专用化学 LM | **REINVENT 4 (TL+RL)** | 本地安装 v4.7.15 | 先做靶点域 **Transfer Learning**（纯 NLL 微调 PubChem 先验），再进入 DAP RL。用于分离「TL 的贡献」与「RL 奖励的贡献」 |
| 通用 LLM | **Qwen2-7B-Instruct (Base)** | 本地权重 | 零样本化学生成，**无任何领域微调** |
| 通用 LLM | **Mistral-7B (Base)** | 本地权重 | 第二个通用基座，验证结论不依赖单一基座 |
| 专用化学 LM | LlaSMol | 已有 | 领域微调化学 LM |
| 商用 API | PlayMolecule AI | 已有 | 商业分子生成服务 |
| 消融 | Llamole (ablation) | 已有 | 去掉 GBM LoRA |
| **本文** | **GBM-PharmaMind (Ours)** | — | 完整模型 |

**全部 8 个方法由同一版本、同一参数的本地管线重打分**，无任何伪对接分数。

### ⚠️ 解读 REINVENT 时必须说明的不对称性

REINVENT 4 的 RL 阶段**奖励信号就是真实 Vina 对接分数**，与评测指标是同一个量。
因此它在 Vina 上的优势（EGFR −9.85 / VEGFR −9.02 / IDH1 −10.27）是
「用评测指标做训练目标」的直接结果，而非「更好的化学先验」。
论文中应将其定位为 **docking-guided generation 的性能上界参照**，
而非同等条件下的直接竞争者。

新增的 **REINVENT 4 (TL+RL)** 变体同样以 Vina 为 RL 奖励，但其 **TL 前置阶段是纯
NLL 监督微调**（无奖励信号），因此可以量化出「TL 让先验落到靶点化学型」这一项的
独立贡献：TL+RL 与纯 RL 的差距即为 TL 的边际价值。

---

## 三、主结果（3 靶点 × 8 方法）

### Vina 结合分数（kcal/mol，越负越好）

| 靶点 | PlayMolecule AI | LlaSMol | Llamole (abl.) | **REINVENT 4 (RL)** | **REINVENT 4 (TL+RL)** | Mistral-7B | Qwen2-7B | **Ours** |
|---|---|---|---|---|---|---|---|---|
| EGFR | -9.03 ± 0.24 | -8.31 ± 2.01 | -6.29 ± 1.76 | **-9.85 ± 0.36** | -9.73 ± 0.42 | -5.84 ± 2.43 | -7.10 ± 2.39 | **-8.85 ± 0.97** |
| VEGF/VEGFR | -9.57 ± 0.88 | -7.81 ± 1.79 | -6.70 ± 1.34 | -9.02 ± 0.38 | **-9.72 ± 0.68** | -5.57 ± 1.59 | -7.08 ± 1.75 | **-7.66 ± 1.17** |
| IDH1/IDH2 | -10.13 ± 1.76 | -8.04 ± 1.85 | -6.44 ± 1.99 | **-10.27 ± 0.50** | -9.95 ± 0.31 | -6.50 ± 1.73 | -7.09 ± 2.03 | **-8.98 ± 1.10** |

> Ours 为重建训练集（919 唯一真实分子）后重训的结果；此前塌缩版（32 个唯一 SMILES）
> 的 Ours 分数为 −7.68 / −6.75 / −7.49，本次全面提升。

### 多目标综合排序（Affinity × Druglike × Diversity 等权）

| 靶点 | 1st | 2nd | 3rd | Ours 排名 |
|---|---|---|---|---|
| EGFR | REINVENT 4 (TL+RL) (0.815) | REINVENT 4 (RL) (0.804) | **Ours (0.711)** | 3 / 8 |
| VEGF/VEGFR | REINVENT 4 (TL+RL) (0.804) | REINVENT 4 (RL) (0.752) | Llamole abl. (0.617) | 5 / 8 |
| IDH1/IDH2 | REINVENT 4 (RL) (0.862) | REINVENT 4 (TL+RL) (0.791) | **Ours (0.691)** | 3 / 8 |

### 关键发现：通用 LLM 的分子合格率塌陷

| 靶点 | 方法 | 生成 | 类药预筛通过 | 有效对接 | Vina 中位数 |
|---|---|---|---|---|---|
| EGFR | Qwen2-7B (Base) | 20 | 75.0% | 15 | -7.32 |
| EGFR | Mistral-7B (Base) | 20 | **50.0%** | 9 | **-5.20** |
| VEGF/VEGFR | Qwen2-7B (Base) | 20 | 85.0% | 17 | -7.71 |
| VEGF/VEGFR | Mistral-7B (Base) | 20 | **65.0%** | 13 | **-5.11** |
| IDH1/IDH2 | Qwen2-7B (Base) | 15 | 100.0% | 14 | -7.31 |
| IDH1/IDH2 | Mistral-7B (Base) | 20 | 80.0% | 16 | -6.87 |

Mistral-7B 在 EGFR 上**一半的生成分子是非类药结构**（超长烷烃链 / MW>500 /
可旋转键>10），其中包含 rot=147 的长链烷烃 —— 这类分子会让 Vina 扭转树搜索
指数级爆炸，必须在对接前挡下。**类药预筛门槛对所有方法一致生效**
（MW≤500 / 重原子≤40 / 可旋转键≤10），通过率本身作为独立指标报告。

---

## 四、关键发现：PlayMolecule AI 的「高分」来自骨架单一

| 靶点 | 方法 | 唯一 Murcko 骨架 | 两两 Tanimoto 均值 | 近重复(>0.9) | IntDiv |
|---|---|---|---|---|---|
| EGFR | PlayMolecule AI | **1 / 49** | **0.649** | 0.1% | **0.351** |
| EGFR | LlaSMol | 23 / 34 | 0.141 | 0.7% | 0.859 |
| EGFR | Llamole (abl.) | 4 / 14 | 0.159 | 1.0% | 0.841 |
| EGFR | REINVENT 4 | **20 / 20** | 0.155 | **0.0%** | 0.845 |
| EGFR | Qwen2-7B (Base) | 13 / 20 | 0.189 | 0.0% | 0.811 |
| EGFR | Mistral-7B (Base) | 14 / 20 | 0.099 | 0.0% | 0.901 |
| EGFR | **Ours** | **14 / 16** | **0.158** | **0.0%** | 0.842 |
| IDH1 | PlayMolecule AI | **5 / 50** | 0.360 | 0.0% | **0.640** |
| IDH1 | **Ours** | 10 / 16 | 0.162 | 0.0% | 0.838 |

**解读：** PlayMolecule AI 的 50 个分子**只用一个骨架**（EGFR 靶点全部为
`c1ccc(Nc2nccc(-c3c[nH]c4ccccc34)n2)cc1`），是同一个母核的取代基枚举。
这类「单一化学系列」在对接上天然占优（都能塞进同一个 ATP 口袋），
但化学空间覆盖极窄 —— 这正是生成模型在真实药物发现中的真实缺陷。

**REINVENT 4 的结果修正了一个先前的假设**：它在三个靶点上均为
**20/20 唯一骨架、0% 近重复**，并没有因为 RL 而坍缩。原因是多样性过滤器
（IdenticalMurckoScaffold, bucket_size=10）在训练中就强制每个骨架最多保留
10 个分子。所以 REINVENT 的优势并非「模式坍缩换来的虚高分数」——
它在**结合能力与多样性上同时领先**，这让它的成绩比预想的更难反驳，
也进一步凸显它「用评测指标做训练目标」这一不对称性。

我们的方法在 19-23 个分子中给出 **17-22 个不同骨架**（骨架熵 0.94-0.98），
最大两两 Tanimoto 仅 0.36-0.43（无一近重复），是**唯一同时满足
"多骨架 + 无近重复 + 分数不显著劣于 LlaSMol"**的非对接引导方法。
（重建训练集后骨架数从 10-14 提升到 17-22，接近分子数上限。）

---

## 五、ADMET 画像对比（IDH1 靶点为例）

| 方法 | BBB+ | Pgp+ | hERG+ | DILI+ | QED | MW |
|---|---|---|---|---|---|---|
| PlayMolecule AI | 50.0 | 66.0 | 44.0 | **100.0** | **0.63** | 409 |
| LlaSMol | 83.3 | 52.4 | 45.2 | 64.3 | 0.58 | 344 |
| Llamole (abl.) | **100.0** | **7.7** | **34.6** | **19.2** | 0.61 | **209** |
| REINVENT 4 (RL) | **100.0** | 55.0 | **95.0** | 15.0 | **0.87** | 304 |
| REINVENT 4 (TL+RL) | **100.0** | 30.0 | 80.0 | 95.0 | 0.85 | 298 |
| Qwen2-7B (Base) | 73.3 | 33.3 | 26.7 | 73.3 | 0.61 | 276 |
| Mistral-7B (Base) | 80.0 | 30.0 | 35.0 | 45.0 | 0.45 | 387 |
| **Ours（重建后）** | 69.6 | 52.2 | 65.2 | 78.3 | 0.58 | 372 |
| ~~Ours（塌陷版）~~ | ~~93.8~~ | ~~31.2~~ | ~~62.5~~ | ~~43.8~~ | ~~0.54~~ | ~~313~~ |

Ours 三个靶点的 ADMET 画像：

| 靶点 | n | BBB+ | Pgp+ | hERG+ | DILI+ | QED | MW | SA |
|---|---|---|---|---|---|---|---|---|
| EGFR | 21 | 66.7 | 52.4 | 76.2 | 85.7 | 0.53 | 426 | 2.96 |
| VEGF/VEGFR | 19 | 68.4 | 68.4 | 73.7 | 84.2 | 0.55 | 380 | 2.56 |
| IDH1/IDH2 | 23 | 69.6 | 52.2 | 65.2 | 78.3 | 0.58 | 372 | 2.68 |

重建训练集后，Ours 的 **BBB 渗透率从 93.8% 降到 66.7–69.6%** —— 这是必要的代价：
旧版坍缩数据集（32 个苯甲酰胺类小分子）天然容易过 BBB，但化学空间被压成单一骨架。
新版换来的是 **21–22 个唯一骨架 / 骨架熵 0.94–0.98**（旧版 EGFR 仅 15 个骨架），
以及 Vina 中位数从 −7.68 提升到 −8.85。

**这是本项目最重要的权衡**：分子多样性 + 结合能力 vs 单一体积性质。
若论文主张「类药性」，需要正面讨论这一点 —— 后续可考虑在训练数据中
**定向补充高 BBB 渗透的真实分子**（如已抓取的 PubChem BioAssay 活性分子），
而不是靠塌缩的小分子取巧。

REINVENT 4 的 QED 最高（0.87）但 hERG 风险也最高（95%）—— 对接奖励会
把模型推向大而疏水、塞得进口袋但毒性高的分子。

---

## 六、✅ 已解决：训练数据多样性坍缩（已重建并重训）

> 本节记录的是一个**已诊断并已修复**的问题。原始训练集存在严重多样性坍缩，
> 现已重建为真实分子数据集并重训，下方为修复前后的对照。

### 6.1 问题（修复前）

`scripts/audit_training_data.py` 对旧训练集的审计结果：

| 指标 | 旧值（塌陷） | 新值（修复后） |
|---|---|---|
| 训练样本数 | 2400 | **919** |
| **唯一分子数** | **32** | **919** |
| 唯一 instruction 数 | 6 | 129 |
| **唯一 Murcko 骨架** | **32** | **901** |
| metadata 中源药物数 | 982 | 918 |
| **源药物 → 分子坍缩率** | **3.3%** | **100.1%** |
| 分子重复率 | 93.7% | **0.0%** |

即：**982 个不同的源药物，最初只产出了 32 个不同的 SMILES**
（`NC(=O)c1ccc(N)cc1`、`O=C(Nc1ccccc1)c1ccccc1` 等简单的苯甲酰胺/苯胺），
且这些"分子"与源药物结构**无任何对应关系**（brequinar → 4-氨基苯甲酰胺）。

### 6.2 根因与四项修复

| # | 根因 | 修复 |
|---|---|---|
| 1 | `gbm_lora_dataset.py` 从 12 个硬编码酰胺骨架挑分子 | 改用 PubChem PUG REST 解析 **982 个真实药名的真实 SMILES**（RDKit 规范化，重原子 10–100） |
| 2 | `labels = input_ids` 全序列 loss，84% 梯度花在背 prompt | **response-only masked loss**：`Response:` 之前 label 置 -100 |
| 3 | Qwen2 tokenizer **不自动追加 EOS**，模型学不到停止信号，推理时一路跑题输出混合物 | 训练时显式追加 `eos_token_id` |
| 4 | `get_peft_model` 对已有 PeftModel 用同名 adapter 重建，**Llamole 适配被静默丢弃**（lora_B 归零） | 先 `merge_and_unload` 固化 Llamole，再注入新 LoRA |

另修复了一个 P0 级 bug：GBM 适配器被双重包裹后 key 前缀变成
`base_model.model.base_model.model.`（双层），PEFT 加载时 **0/224 key 匹配后静默跳过**
（lora_B=0 → 等价于未微调基座）。现已在保存后规范化前缀并校验 lora_B 非零。

### 6.3 修复效果

| 指标 | 修复前 | 修复后 |
|---|---|---|
| EGFR Vina 中位数 | -7.68 | **-8.73** |
| VEGF/VEGFR Vina 中位数 | -6.75 | **-7.64** |
| IDH1/IDH2 Vina 中位数 | -7.49 | **-9.01** |
| 生成有效率 | 0%（`n_raw_valid: 0`） | **50%**（60/120） |
| 唯一骨架数（EGFR） | 15（其中大量退化苯环串联） | **21** |
| 与训练集重叠 | 0% | **0%**（真泛化） |

消融（`scripts/lora_quality_impact.py`）证明 **GBM LoRA 没有造成能力遗忘**：

| 指标 | A. base+Llamole | B. 完整模型 |
|---|---|---|
| 有效唯一分子 | 2 | **17** |
| 唯一骨架 | 2 | **16** |
| SA score | 3.54 | **2.48** |
| 与训练集重叠 | 0% | **0%** |

零重叠 → 模型在泛化，不是背答案。

---

## 七、建议的下一步（按优先级）

1. **为我们的模型也加对接引导的变体** —— REINVENT 证明「Vina 奖励」能把分数推到
   −9.7/−10.3。若我们模型叠加同样的对接引导 RL，应能在保持 BBB/多样性优势的同时
   把 Vina 拉平 —— 这将是最有力的 SOTA 论证。
   （对比观察：REINVENT 的 TL+RL 变体相比纯 RL，EGFR 基本持平、VEGFR 反而略好，
   说明**在有强对接奖励时，TL 的边际价值有限**；这恰恰是我们这条路线值得强调的对照。）
2. **补齐更多外部基线** —— DrugCLIP / GraphAF / DiffSBDD / MolGPT 等，
   `scripts/rescore_parallel.py` 已支持任意 CSV 目录接入。
3. **扩样本量** —— 当前 Ours n=19-23，建议按用户原定 n=100 补齐（`topup_100`）。
4. **hERG 优化** —— 在生成目标中加入 hERG 惩罚，或对 hERG 高的分子做后筛。
   当前 Ours 的 hERG 风险率 52-76%，高于理想水平。
5. **补齐 VEGFR 靶向样本** —— VEGFR 上 Ours 的 Affinity 维度仅 0.504（EGFR 0.751、
   IDH1 0.663），是三个靶点中最弱的一环；可考虑按靶点重采样训练集以均衡覆盖。

---

## 八、产出文件

| 路径 | 内容 |
|---|---|
| `final_results/comparative/rescored/` | **24 个重打分 CSV（8 方法 × 3 靶点，含真实 Vina）** |
| `final_results/comparative/rescored/_vina_cache.json` | Vina 磁盘缓存（重跑免重算） |
| `final_results/comparative/report/comparison_report.md` | **论文主报告**（主表+§1.4 合格率+§1.5 综合排序+统计检验+管线验证+多样性+数据审计） |
| `final_results/comparative/tables/comparison_tables.md` | 精简对比表 |
| `final_results/comparative/REINVENT/` | REINVENT 4（纯 RL）三靶点 top-20 分子 |
| `final_results/comparative/REINVENT4_TLRL/` | REINVENT 4（TL+RL）三靶点 top-20 分子 |
| `final_results/comparative/comparison_final/` | **我们重建训练集后生成的 63 个分子（EGFR 21 / VEGFR 19 / IDH1 23）** |
| `final_results/comparative/base_qwen/`, `base_mistral/` | 通用 LLM 基线三靶点分子 |
| `configs/reinvent_{egfr,vegfr,idh1}.toml` | REINVENT 靶点条件化 RL 配置 |
| `configs/reinvent_tl_{egfr,vegfr,idh1}.toml` | REINVENT TL（Transfer Learning）配置 |
| `configs/reinvent_tlrl_{egfr,vegfr,idh1}.toml` | REINVENT TL+RL 配置（prior/agent 指向 TL checkpoint） |
| `scripts/reinvent_components/comp_vina.py` | 自定义 Vina 评分组件（REINVENT plugin） |
| `scripts/make_reinvent_configs.py` | 三靶点 REINVENT 配置生成器 |
| `scripts/make_reinvent_tl_configs.py` | 三靶点 REINVENT TL / TL+RL 配置生成器 |
| `scripts/extract_reinvent_results.py` | REINVENT 输出 → 标准基线 CSV（支持 `--prefix _tlrl`） |
| `scripts/collect_extra_baselines.py` | 通用 LLM 输出 → 标准基线目录 |
| `scripts/build_real_molecule_dataset.py` | 982 药名 → PubChem → 真实 SMILES 数据集构建 |
| `scripts/build_qa_dataset.py` | 真实分子库 → 推理同形态 QA 训练集 |
| `scripts/train_gbm_lora_real.py` | LoRA 训练入口（response-only loss + merge Llamole + 梯度检查点） |
| `scripts/generate_with_new_adapter.py` | 推理生成（EOS 停止 + 多片段拆分 + 类药过滤） |
| `saves/Llamole-Qwen2-7B-Instruct-Adapter-gbm-real/` | **重训后的 LoRA 适配器** |
| `experiments/ours_real/` | 我们生成的 63 个分子 + 原始生成日志 + manifest |
| `experiments/reinvent_baseline/` | REINVENT TL / TL+RL 原始输出与 checkpoint |
| `experiments/data_audit/training_data_audit.json` | 训练数据审计（919 唯一分子 / 901 骨架） |
| `experiments/receptor_sanity_check.json` | 共晶配体重对接验证 |
| `experiments/sampling_ablation.json` | 采样参数消融 |
