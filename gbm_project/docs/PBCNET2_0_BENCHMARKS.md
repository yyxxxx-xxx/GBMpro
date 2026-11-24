# PBCNet2.0 借鉴点分析
=====================

> **参考文献**
> - Yu J, Sheng X. *Atomic-level protein–ligand recognition with PBCNet2.0 for probe discovery.*
>   Nature Chemical Biology (2026). [doi:10.1038/s41589-026-02241-x](https://www.nature.com/articles/s41589-026-02241-x)
> - 预印本: [bioRxiv 2025.06.04.657800](https://doi.org/10.1101/2025.06.04.657800)
> - 代码: [github.com/YuJie-0202/PBCNet2.0](https://github.com/YuJie-0202/PBCNet2.0) (MIT License)
> - 阅读日期: 2026-11-24 ｜ 仓库 commit: `master` (2026-09-18)

---

## 0. 先厘清定位: 它和我的系统是什么关系

| | **PBCNet2.0** | **我的系统 (Llamole + GRPO)** |
|---|---|---|
| 任务 | **打分器** (predict ΔpAct) | **生成器 + RL 优化器** |
| 输入 | 两个分子的**结合位姿** (pocket-ligand complex) | 文本 prompt |
| 输出 | 相对亲和力差值 | SMILES 分子 |
| 学习方式 | 监督学习 (8.6M 实验亲和力标签) | RL (物理反馈 + 冻结先验) |
| 是否需要对接 | 需要 (依赖输入位姿) | 需要 (Vina 在线对接) |
| 关系 | **可作为 reward 的替代/补充** | — |

> ⚠️ **关键结论**: PBCNet2.0 **不是生成模型**, 不能替代我的 GRPO。
> 但它的**方法论**有几处可以直接搬到我的 RL 框架里, 而且价值不小。
> 下文只讲**真正可落地**的点, 并标注实现成本。

---

## 1. 【最高价值】成对差分奖励: 用 Δ 而非绝对分数

### 论文/代码依据

`model_code/models/readout.py` — readout 只做**差分**, 不做拼接:

```python
emb1, att1 = self._readout(g1, bb)
emb2, att2 = self._readout(g2, bb)

emb = self.norm(emb1 - emb2)      # ← 只用差
out_put = self.FNN(emb)

emb_neg = self.norm(emb2 - emb1)   # ← 反向差 (符号翻转增强)
out_put_neg = self.FNN(emb_neg)
```

论文原话 (Methods, Readout module):

> "employing Δ as the pair-level representation allows the model to devote more
> attention to protein-ligand interactions. This is important because if a model
> solely focuses on ligand molecular structures, [g_ref] and [g_query] would be
> similar as they are structural analogues, making Δ close to a null vector.
> This scenario would create an unsolvable optimization problem."

### 为什么这对我的系统重要

我的 GRPO 目前用**绝对 reward** 做组内 z-score:

```103:110:gbm_project/src/rl/advantage.py
        mean = float(np.mean(valid_rewards))
        std = float(np.std(valid_rewards))

        if std < min_std:
            # 组内 reward 几乎一样 —— 没有相对好坏, 无梯度信号
            out[pid] = GroupAdvantage(
                prompt_id=pid, rewards=list(valid_rewards),
                advantages=[0.0] * n_valid, n_valid=n_valid,
                n_total=n_total, skipped=True,
                skip_reason="zero_variance")
            continue

        advs = [(r - mean) / (std + eps) for r in valid_rewards]
```

**问题**: 组内 z-score 隐含假设"绝对分数高 = 更好"。但 M2.2 已经暴露了这个假设的代价 ——
reward 分布左尾被拉低, mean 涨得比 median 慢。绝对分数里混着大量**与靶点无关**的
噪声项 (TPSA 归一化、MW 分段、ADMET 端点截断)。

**PBCNet2.0 的启示**: 把 reward 定义成**同一组内两个分子的差**,
天然消掉所有"与组内对比无关"的常数偏移。

### 可落地方案: 成对差分 advantage

不用改 reward 函数 (Vina/ADMET 照旧), 只改 advantage 计算:

```
# 当前 (绝对分数 + 组内 z-score)
A_i = (R_i - mean(R)) / std(R)

# 建议 (成对差分)
A_i = (R_i - R_ref) / (|R_i - R_ref| + eps)     # 相对组内最优/最差
```

具体可做成"**成对排序 + 差值幅度**"的混合:
- 排序部分: 保证方向正确 (谁好谁坏)
- 差值幅度: 保证"好多少"的信息不丢 (这正是 PBCNet2.0 的 Δ 保留的)

**实现成本**: 低 (只改 `advantage.py` 一个文件, ~30 行)
**预期收益**: 缓解左尾问题, 让 mean 提升幅度追上 median
**验证方法**: 加一组 M5-E 消融 (`--advantage_mode pairwise`), 与 M3 对比 mean/median 差距

---

## 2. 【高价值】符号翻转增强: 免费的对称性正则

### 代码依据

`model_code/train.py`:

```python
logits, logits_neg = model(graph1, graph2)
loss = loss_func(logits.squeeze().float(), label.float()) \
     + loss_func(logits_neg.squeeze().float(), label_neg.float())
```

`readout.py` 里 `out_put` 和 `out_put_neg` 共享**同一个 FNN** (不是两个网络),
所以符号翻转增强的成本只是**一次额外前向**, 参数为零。

论文:

> "To achieve anti-symmetry (Δf(-Δ) = -f(Δ)) during training, we also input
> -Δ = -Δ into the fully connected layer and calculate the combined loss,
> implementing a computationally efficient sign-flip augmentation strategy."

### 为什么这对我的系统重要

我的 GRPO 有一个**结构性缺陷**: `advantage.py` 里的 z-score 保证了
`Σ A_i = 0` (组内和为零), 但**没有保证交换对称性** ——
即"如果把组内两个分子交换, 它们的 advantage 应当互为相反数"。

更具体地说, 我的 `min_std` 阈值机制 (`std < 1e-4` 就跳过整组) 是一个
**硬性的信息损失**: 组内 reward 接近时, 直接丢掉整组梯度。

**PBCNet2.0 的启示**: 用**对称性约束代替硬阈值**。
当组内方差小时, 不是跳过, 而是让模型在"对称扰动"下保持一致 ——
这样即使组内差异微小, 仍有梯度。

### 可落地方案: 对称性正则项

```python
# 在 GRPO loss 里加一项
loss = policy_loss + beta_kl * kl
        + lambda_sym * symmetry_penalty
```

其中 `symmetry_penalty` 衡量: 对组内 reward 排序做微小扰动后,
advantage 分配是否保持一致。

或者更简单的落地版: **降低 min_std 阈值 + 用 rank-based advantage 兜底**
(论文提到 PBCNet1.0 用的是 rank, PBCNet2.0 改用 Δ)。

**实现成本**: 低到中
**预期收益**: 减少 `n_skipped_groups` (M4-D 已经出现 18 次), 提高有效样本利用率

---

## 3. 【高价值】几何感知的奖励: 从"标量打分"升级到"原子级交互"

### 论文依据

PBCNet2.0 的核心创新是 **Cartesian rank-2 tensor 等变网络**, 显式建模
距离、方向、**角度**。

论文的 F-Opt 基准 (氟正交多极相互作用) 展示了纯距离方法的失败:
- PBCNet (1.0) 用的是**经验原子对势函数**, 给合理的 F···C 距离分配低概率
- PBCNet2.0 能识别, 并在角度被破坏时 (79.2°→52.9°) 权重从 0.98 掉到 0.5

图构建 (`Graph2pickle.py`) 的关键设计:

```python
# type1: 配体内距离边 (<= 5A)
# type2: 口袋内共价边
# type3: 配体-口袋距离边 (<= 5A)   ← 蛋白-配体交互
# type4: 口袋内距离边 (<= 3A, 且与配体相连的原子优先)
```

距离边阈值 5Å, 口袋截取配体周围 8Å 的残基。

### 为什么这对我的系统重要

我的 reward 目前**完全是标量**: Vina 给一个 kcal/mol, ADMET 给 8 个概率,
加权几何平均成 1 个数。

这导致一个**我已经在 M4-C 里观察到的现象**: 多样性过滤只能看
Murcko 骨架这种粗粒度特征, 无法区分"同一个骨架上哪个基团在与蛋白形成关键氢键"。

**PBCNet2.0 的启示**: reward 应该带上**"哪个原子在与口袋形成关键交互"**的信息。

### 可落地方案: 交互感知的多样性过滤

不需要训等变网络 (成本太高), 可以用 **Vina 自带的姿态输出**做轻量版:

Vina 已经在产出对接姿态 (`property_pipeline/docking.py`)。从姿态里算:
- 配体原子与口袋残基的**氢键 / 疏水接触图**
- 把这个接触图作为 diversity 的 key, 替代/补充 Murcko 骨架

效果: 过滤从"骨架级"升级到"**结合模式级**" ——
同一个骨架但结合模式不同的分子不会互相挤掉。

**实现成本**: 中 (需要从 Vina 输出提取 pose, 再定义交互指纹)
**预期收益**: 缓解 M4-C 暴露的 prompt starvation, 提升配对检验胜数
**备注**: 这是我目前认为**最有论文价值**的借鉴点

---

## 4. 【中价值】主动学习循环: 参考分子驱动的迭代选优

### 代码依据

`model_code/Selection.py` 的完整循环:

```python
for ref in range(1, 1000, 3):
    # 1) 用当前参考集预测, 选出 top-3 新分子
    selected_mols = model_predict(file_name, ref, model, device, logger_writer)
    # 2) 把新分子加入参考集, 重新生成预测/微调数据
    input_generation(file_name, ref, newly_mols=selected_mols)
    # 3) 用参考集微调模型
    model2 = model_finetune(file_name, ref, model, device, logger_writer)
    # 4) 迭代
    selected_mols = model_predict(file_name, ref, model2, device, logger_writer)
```

论文的量化结果:
- **advantage ratio** (资源节省): PBCNet2.0 **40.51%** > PBCNet 30.18% > MM-GB/SA 14.93%
- **efficiency improvement ratio** (优化加速): PBCNet2.0 **717.81%** > PBCNet 473% > MM-GB/SA 281%

### 为什么这对我的系统重要

我的 RL 是**单向**的: SFT 模型 → GRPO 训练 → 结束。**没有迭代**。

PBCNet2.0 的循环是: 预测 → 选优 → 选中的分子成为新参考 → 再预测 → ...

这对应到我的系统就是一个自然的 **M6 方向**:
```
GRPO 训练 → 评估出 top-K 分子 → 把 top-K 作为"锚点"
  → 下一轮 rollout 时, prompt 里带上锚点分子 (条件生成)
  → 围绕锚点做局部优化
```

### 可落地方案: 锚点条件生成 (anchor-conditioned generation)

我的 prompt bank 现在是 (cell_state × metric) 组合, 与具体分子无关。
改成: 把当前 best-K 分子以 SMILES 形式注入 prompt, 让模型做**类似物优化**。

**实现成本**: 中 (改 `prompt_bank.py` + 动态更新 prompt 集)
**预期收益**: 直接对上 PBCNet2.0 的 "717.81% 加速" 叙事, 论文价值高
**风险**: prompt 分布漂移会破坏 prior/agent 的条件一致性 (设计文档 Q1 已经提到这个顾虑)

---

## 5. 【中价值】零样本 / 少样本评测协议

### 论文依据

三种评测范式:

| 范式 | 场景 | PBCNet2.0 表现 |
|---|---|---|
| **Zero-shot** | 早期优化, 无 SAR 数据 | ρ=0.67, R=0.66 (FEP set, 16 个化学系列) |
| **Few-shot** | 中后期优化, 有少量 SAR | 微调后**持续超过 FEP+** |
| **Prioritization** | 真实项目选优 | 8 个靶点, advantage ratio 40.51% |

关键对比: PBCNet2.0 与 Schrödinger **FEP+** 的 ρ 差异 p=0.39 (>0.05),
即**统计上无显著差异**, 首次让 AI 方法达到工业级 FEP 模拟水平。

### 为什么这对我的系统重要

我现在的评估是 **"base vs RL, 一次训练, 一次对比"** —— 这是
**single-shot 评测**, 没法回答"RL 迭代 3 轮会不会更好"。

而且我 M4 的 B 组 (β_kl=0.15) 已经出现"训练更久反而更差" ——
这正是需要**多轮迭代评测**才能看清的曲线形状。

### 可落地方案: 多轮迭代评测 + 少样本设定

**评测协议升级** (不改代码, 改实验设计):

| 协议 | 做法 | 回答什么问题 |
|---|---|---|
| Zero-shot | 靶点上**完全没有** RL 数据时的性能 | 冷启动能力 |
| Few-shot | 用 N=5/10/20 个已知活性分子微调 RL LoRA | 数据效率曲线 |
| 多轮迭代 | 训 3 轮 (每轮用上一轮 top-K 做锚点) | 收敛上限在哪 |

**实现成本**: 低 (主要是实验编排, 复用 `run_m5_eval.sh` 的模式)
**预期收益**: 论文说服力大幅提升; 少样本曲线是审稿人最爱的图

---

## 6. 【中价值】可解释性: 原子-原子权重热图

### 论文依据

论文用 **invariant scalar embedding** 作为原子对重要性权重:

> "The invariant terms [I], [A], and [S] obtained from the Equation 5 can be
> seen as a measure of importance between atoms and [j]. We extracted these
> invariant terms from the first layer of the message-passing model and
> computed their average as relevance weights."

用法: 标注配体某原子周围 5Å 内的蛋白原子权重 (0.00-1.00),
识别氢键、氟正交多极相互作用等。

配套的**扰动验证**:
- 移动 F 原子破坏角度 → 权重从 0.98 掉到 0.5
- 突变 ASN229→CYS (破坏氢键) → 氢键氧权重 0.99 → 完全消失

### 为什么这对我的系统重要

我的 RL 现在是**黑盒的**: reward 涨了 31%, 但**说不出是哪个结构变化起的作用**。

论文的"扰动验证"模式尤其值得学 —— 它把"模型学到了东西"从
相关性提升到了**因果证据**。

### 可落地方案: 结构归因分析 (不必训模型)

不需要训等变网络, 用**差分对接**就能做:

1. 取 RL 生成的高分分子 A 和 base 的同骨架分子 B
2. 对 A/B 分别做 Vina 对接, 提取 pose
3. 计算接触残基差异 → 生成"RL 改进了哪个口袋接触"的热图
4. **扰动验证**: 把关键氢键给体的 H 换成 C (破坏氢键), 重打分, 验证 reward 显著下降

第 4 步特别重要: 它能**直接反驳 reward hacking 质疑** ——
"你的分子真的是靠这个氢键提高亲和力的, 不是靠改个 MW 骗分吗?"

**实现成本**: 中 (需要写 pose 分析 + 扰动重打分脚本)
**预期收益**: 论文里"机制分析"章节的素材; 强化 reward hacking 论证

---

## 7. 【低价值但低成本】训练数据构建的细节

### 论文/代码依据

`Graph2pickle.py` 的数据处理有若干值得抄的工程细节:

1. **按 (DOI, Target, Source) 分组** —— 保证同一组内是同一靶点同一文献的活性数据
2. **Tanimoto ≥ 0.5 聚类成化学系列** —— 系列内分子是结构类似物
3. **多标签冲突处理**:
   ```
   if max_label < 3 * min_label:  用平均值
   else:                          丢弃该分子 (保证数据质量)
   ```
4. **共晶配体相似度匹配选 PDB**: 与系列内分子 Tanimoto 最高的共晶配体 → 用它的 PDB
5. **修正 modification bias**: 统计同 R-group 改造对的 Δ 分布, 确认均值≈0 (正态)
   —— 否则模型可能只是记住"这个改造通常提升活性"

### 可落地方案

- **第 3 条 (多标签冲突处理)** 直接可以用在我的 `active_qa_train.json` 上 ——
  现在同一个 CID 可能有多个 IC50 值, 我是直接取一个, 应该改成
  "差异 <3 倍取均值, 否则丢弃"
- **第 5 条 (modification bias 检查)** 可以加到我的评估里:
  在 SFT 训练集里统计"同类 R-group 改造"对 ADMET/reward 的平均影响,
  如果系统性偏向某个方向, 我的 RL 结论就可能被质疑"只是复现了这个偏置"

**实现成本**: 低 (纯数据分析脚本)
**预期收益**: 加固实验可信度, 审稿人可能会问这个

---

## 8. 【反面参考】PBCNet2.0 自己的局限

论文诚实地列了三条, 对我的系统同样是警示:

> "First, the model relies on static protein-ligand binding conformations as
> input, which prevents it from explicitly accounting for the entropic effects."

> "Second, PBCNet2.0 struggles to predict negative data, i.e., cases where
> R-group modifications result in complete loss of ligand activity. This stems
> primarily from the scarcity of well-characterized negative data."

> (第三条是诱饵数据问题)

**对我的启示**:
1. **静态对接的熵盲区** —— 我的 Vina 也是静态的, 完全忽略了结合熵和构象柔性。
   这解释了我 M4 中"训练更久反而略降"的部分原因: 模型在往**对接友好但柔性大**的
   分子上钻 (reward hacking 的软形式)。
   → 对策: 可以在 reward 里加**构象柔性惩罚** (目前只测了 rotatable bonds 上限 15,
   但那只是硬守卫, 不是惩罚项)
2. **负样本稀缺** —— 我的 reward 里大量分子被硬截断成 0 (46.4%!),
   这正是"负样本"占比过高的问题。PBCNet2.0 明确说这类数据"adding noise"。
   → 对策: 我已经在做 (把 0 分改成 floor), 但可以更系统地做**负样本质量分层**

---

## 9. 优先级建议

| 优先级 | 借鉴点 | 成本 | 预期收益 | 章节 |
|---|---|---|---|---|
| **P0** | §1 成对差分 advantage | 低 | 缓解左尾, mean 追上 median | M5-E 消融 |
| **P0** | §6 差分对接 + 扰动归因 | 中 | 机制分析素材, 反驳 hacking 质疑 | 论文机制章节 |
| **P1** | §5 多轮迭代 + 少样本评测 | 低 | 论文说服力 | M6 实验设计 |
| **P1** | §7 多标签冲突处理 | 低 | 数据质量 | 数据清洗 |
| **P2** | §2 对称性正则 | 中 | 减少 skipped group | M5-E 变体 |
| **P2** | §3 交互感知多样性过滤 | 中 | 缓解 prompt starvation | M6 |
| **P2** | §4 锚点条件生成 | 中 | 对上"加速 717%" 叙事 | M6 |
| **P3** | §8 构象柔性惩罚 | 中 | 抑制软 reward hacking | reward v3 |

---

## 10. 立即可做的三件事 (不依赖 GPU)

1. **多标签冲突处理** (§7.1) —— 清洗 `active_qa_train.json`, 30 行代码
2. **modification bias 检查** (§7.2) —— 统计 SFT 数据的改造偏置, 纯分析
3. **advantage pairwise 模式** (§1) —— 在 `advantage.py` 里加 `--advantage_mode`,
   默认保持 `zscore` 不变, 保证现有实验可复现

---

## 附: 关键代码位置索引

| 内容 | 位置 |
|---|---|
| **成对差分 readout** | `model_code/models/readout.py` (forward 方法) |
| **符号翻转增强** | `model_code/train.py` (train 函数 loss 计算) |
| **主动学习循环** | `model_code/Selection.py` (主循环) |
| **图构建 / 交互边定义** | `Graph2pickle.py` (Graph_Information) |
| **原子特征** | `Graph2pickle.py` (lig_atom_featurizer) |
| **多标签冲突处理** | `model_code/Dataloader/dataloader.py` (LeadOptDataset) |
| **等变张量网络** | `model_code/models/tensornet.py` |
| **损失函数集 (含 rank loss)** | `model_code/utilis/function.py` (get_loss_func) |
| **可解释性权重提取** | 论文 Methods "Interpretability analysis" |
