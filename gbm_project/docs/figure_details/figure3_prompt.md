# 图3绘图 Prompt：两阶段 LoRA 微调架构

## 绘图 Prompt

```
绘制一幅简洁、专业的科技论文插图，展示两阶段 LoRA 微调流水线，用于药物分子
生成大语言模型。风格参考 NeurIPS、ICML、Nature Machine Intelligence 等计算机
领域 SCI 顶级期刊的深度学习架构图——模块清晰、箭头明确、装饰最少、信息密度高。
整体素材应体现大模型时代的技术美学，借鉴 PyTorch 框架图和 HuggingFace 模型卡
的常用视觉元素。

**绘制要求：**
- 全图使用中文（英文术语保留原文，如 Qwen2-7B-Instruct、LoRA、PeftModel、
  Adapter、merge_and_unload、GBM、EGFR、SMILES 等）
- 仅保留主要模块和核心概念，删除所有代码片段、具体参数数值细节
- 所有图示、节点、连接箭头统一使用清晰的 SVG 矢量图格式
- 文字清晰锐利，所有字号不低于 10pt
- 像素尽可能高，确保打印和屏幕均清晰

---

**图3：两阶段 LoRA 微调架构（从左到右水平布局，共 5 个主要模块）**

**模块0 — 基座模型（最左侧）**
- 垂直圆角矩形，深靛蓝渐变背景（#1A237E），白色边框
- 标签："Qwen2-7B-Instruct"
- 副标题标签（略小）："基座模型"
- 底部信息标签（略小）："7B 参数 | FP16 | HuggingFace"
- 右上角小图标（SVG 矢量）：锁形图标 + "Frozen" 标签，表示权重冻结
- 右侧发出粗实线箭头，指向 Stage 1

**模块1 — Llamole Adapter（左侧偏中）**
- 垂直圆角矩形，紫/紫罗兰渐变背景（#7B1FA2），白色边框
- 标签："Stage 1 | Llamole Adapter"
- 路径标签（略小，灰色字体）："saves/Llamole-...-Adapter"
- 内部结构列表（简洁展示，无具体数字）：
  · 类型：LoRA (PEFT)
  · 目标模块：q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- 底部注释箭头（向下）："merge_and_unload() → 合并回基座"
- 右侧发出实线箭头，标注 "合并后" 指向下一模块

**模块1合并 — 中间过渡状态（居中偏左）**
- 垂直圆角矩形，靛蓝渐变背景（#3949AB），白色边框
- 标签："Qwen2-7B-Instruct + Llamole"
- 副标题（略小）："Stage 1 合并完成"
- 底部说明（略小）："LoRA Adapter 合并后不再独立存在"
- 右侧发出实线箭头，指向 Stage 2

**模块2 — GBM Domain Adapter（右侧偏中）**
- 垂直圆角矩形，青绿渐变背景（#00695C），白色边框
- 标签："Stage 2 | GBM Domain Adapter"
- 路径标签（略小，灰色字体）："saves/Llamole-...-gbm-with-graph-models"
- 内部结构列表（简洁，无具体数字）：
  · 类型：LoRA (PEFT)
  · 目标模块：同 Stage 1
- 特别高亮标注（强调与 Stage 1 的区别）：
  · 右上角图标：保留标签（SVG 矢量）+ "PeftModel（保留，未合并）"
- 底部注释："推理时可动态切换"
- 右侧发出粗实线箭头，标注 "PeftModel.load()" 指向最终模型

**最终模块 — 推理模型（最右侧）**
- 垂直圆角矩形，亮绿渐变背景（#2E7D32），白色粗边框，增加发光/阴影效果
- 标签："最终推理模型"
- 副标题："PeftModel (Qwen2-7B + Llamole + GBM)"
- 底部标签："Ready for ToT Search"
- 右上角 SVG 图标：火箭或播放图标

---

**附加元素（嵌入主图中，不破坏主体结构）**

1. **Fallback 回退切换机制（置于最终模型下方）**
   - 独立小区域，浅橙背景（#FFF3E0），橙色虚线边框
   - 标签（略小）："SMILES Fallback 回退机制"
   - 内容（简洁流程块）：
     · 条件标签："当微调模型无有效 SMILES 输出时："
     · 箭头指向流程块："切换至 Stage 1 合并模型"
     · 箭头指向流程块："添加 Few-shot 示例至 Prompt"
     · 箭头指向流程块："恢复 SMILES 生成"
   - 用虚线弧形箭头从最终模型指回 Stage 2 标注："Fallback 触发"

2. **Llamole 图模型组件（侧边面板，右上角小区域）**
   - 独立小区域，浅紫背景（#F3E5F5），紫色虚线边框
   - 标签（略小）："Llamole 图模型组件（可选）"
   - 内容（SVG 列表形式）：
     · GraphDecoder（SMILES → 图）
     · GraphEncoder（图 → 嵌入向量）
     · GraphPredictor（连接性预测）
     · 连接器：g2l, l2g_dec, l2g_pred
   - 标注："GraphLLMForCausalMLM = 语言模型 + 图模型"

3. **ToT 系统集成（底部区域）**
   - 水平括号结构，浅灰背景（#F5F5F5）
   - 标签："由 TreeOfThoughtsGenerator 调用"
   - 内部内容（无具体代码，简洁图块）：
     · 图块1："Level 0：生成骨架提议"
     · 图块2："Level 1：生成组装策略"
     · 图块3："Level 2：生成 SMILES"
     · 图块4："Fallback：base_merged + few-shot"

4. **训练数据流（底部小箭头，可选）**
   - 底部箭头从下方指向上方模块
   - 标注（略小）："训练数据"
   - 流程标签："格式：Instruction → Response"

---

**连接线样式**
- 粗实线（3px，深色）：主流程路径（模块间主要连接）
- 细实线（1px）：ToT 集成、训练数据等辅助连接
- 虚线（2px 虚线）：Fallback 回退路径
- 点划线（1px）：可选组件连接
- 弧形虚线：Fallback 触发时的回退路径

---

**整体设计要求：**
- 布局：从左到右水平流，5 个主模块 + 3 个附加区域
- 背景：白色或极浅灰（#FAFAFA），轻微对齐辅助线
- 盒式样式：圆角矩形，彩色渐变填充，白色边框，轻微投影
- 箭头：主流程用实线加粗箭头，可选/回退路径用虚线
- 配色编码：
  · 模块0（基座）：深靛蓝（#1A237E）
  · 模块1（Llamole）：紫色（#7B1FA2）
  · 模块1合并：靛蓝（#3949AB）
  · 模块2（GBM）：青绿色（#00695C）
  · 最终模型：亮绿色（#2E7D32）
  · Fallback：橙色（#F57C00）
- 字体：中文用黑体或无衬线，英文术语保留原文，字号 ≥ 10pt
- 层次强调：Stage 1（合并后消失）vs Stage 2（保留为 PeftModel）用不同的边框样式和背景色区分
- 底部图注（中文）："图 3：两阶段 LoRA 微调流水线。Stage 1（Llamole Adapter）提供通用指令遵循能力，通过 merge_and_unload() 合并至基座模型。Stage 2（GBM Domain Adapter）提供 GBM 领域知识和分子生成能力，以 PeftModel 形式保留用于推理时动态加载。当微调模型无法生成有效 SMILES 时，Fallback 机制切换至 Stage 1 合并模型并添加 Few-shot 示例。"
