"""
GBM LoRA微调实现
基于药物-细胞状态连接性数据进行LoRA微调
参考scFOCAL项目的药物语义化和连接性标注方法
集成完整的Llamole图模型支持
"""

import os
import torch
import json
import torch.nn as nn
from typing import Dict, List, Any, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training
)
from datasets import Dataset
import numpy as np

# Import Llamole components
try:
    from src.model.loader import (
        load_graph_decoder,
        load_graph_encoder,
        load_graph_predictor,
        load_language_model
    )
    from src.model.modeling_llamole import GraphLLMForCausalMLM
    from src.hparams.model_args import ModelArguments
    LLAMOLE_AVAILABLE = True
except ImportError as e:
    LLAMOLE_AVAILABLE = False
    print(f"Warning: Llamole components not available ({e}), falling back to standard PEFT")


class GBMLoRAFinetuner:
    """GBM LoRA微调器 - 集成完整Llamole图模型支持"""

    def __init__(self, base_model_path: str, lora_config: Dict[str, Any],
                 use_llamole_graph_models: bool = False,
                 graph_decoder_path: str = None,
                 graph_encoder_path: str = None,
                 graph_predictor_path: str = None,
                 graph_lm_connector_path: str = None):
        self.base_model_path = base_model_path
        self.lora_config = lora_config
        self.use_llamole_graph_models = use_llamole_graph_models
        self.graph_decoder_path = graph_decoder_path
        self.graph_encoder_path = graph_encoder_path
        self.graph_predictor_path = graph_predictor_path
        self.graph_lm_connector_path = graph_lm_connector_path
        self.tokenizer = None
        self.model = None
        self.graph_decoder = None
        self.graph_encoder = None
        self.graph_predictor = None
        self.connectors = None

    def load_model_and_tokenizer(self, llamole_adapter_path: str = None):
        """加载基础模型和tokenizer"""
        print(f"Loading model from {self.base_model_path}")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            padding_side="right",
            local_files_only=True
        )
        # 左侧截断：保证 "Response: <smiles>" 永远完整保留在序列尾部，
        # 否则最长样本（764 token）会从右侧截断掉 response 末尾，
        # 恰好把要学的化学内容切掉。
        self.tokenizer.truncation_side = "left"

        # 添加特殊token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )

        # 如果提供了Llamole适配器，加载它
        if llamole_adapter_path:
            print(f"Loading Llamole adapter from {llamole_adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, llamole_adapter_path)
            # 关键: 必须先 merge_and_unload 把 Llamole 适配固化进基座，
            # 否则后续 get_peft_model() 再注入同名 'default' adapter 时，
            # PEFT 会用全新的 LoRA 层替换掉 target_modules 覆盖的模块，
            # 导致 Llamole 在这些模块上的化学适配被静默丢弃
            # （实测：self_attn 的 lora_B 均值从 0.00554 掉到 0.00110）。
            if self.lora_config.get("merge_llamole_before_lora", True):
                print("Merging Llamole adapter into base weights ...")
                self.model = self.model.merge_and_unload()
                print("✓ Llamole adapter merged into base model")

        print(f"Model loaded: {self.model.__class__.__name__}")
        print(f"Model parameters: {self.model.num_parameters():,}")

        # 加载Llamole图模型（如果启用）
        self.load_llamole_graph_models()

        # 设置完整的Llamole模型（如果图模型加载成功）
        self.setup_llamole_model()

    def load_llamole_graph_models(self):
        """加载Llamole图模型组件"""
        if not self.use_llamole_graph_models or not LLAMOLE_AVAILABLE:
            print("Skipping Llamole graph model loading (not enabled or not available)")
            return

        print("Loading Llamole graph models...")

        # 创建ModelArguments对象 - 只使用必要的参数
        model_args = ModelArguments(
            model_name_or_path=self.base_model_path,
            graph_decoder_path=self.graph_decoder_path,
            graph_encoder_path=self.graph_encoder_path,
            graph_predictor_path=self.graph_predictor_path,
            graph_lm_connector_path=self.graph_lm_connector_path
        )

        # 设置额外的属性（如果需要）
        if hasattr(model_args, 'compute_dtype'):
            model_args.compute_dtype = torch.float16
        if hasattr(model_args, 'disable_graph_model_gradient'):
            model_args.disable_graph_model_gradient = True
        if hasattr(model_args, 'print_param_status'):
            model_args.print_param_status = False

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 加载图模型
        try:
            self.graph_decoder = load_graph_decoder(model_args, self.graph_decoder_path, device)
            print("✓ Graph decoder loaded")

            self.graph_encoder = load_graph_encoder(model_args, self.graph_encoder_path, device)
            print("✓ Graph encoder loaded")

            self.graph_predictor = load_graph_predictor(model_args, self.graph_predictor_path, device)
            print("✓ Graph predictor loaded")

            # 加载连接器权重
            if self.graph_lm_connector_path and os.path.exists(self.graph_lm_connector_path):
                self.connectors = {}
                connector_files = ['graph_to_lm_connector.pt', 'lm_to_graph_decoder.pt', 'lm_to_graph_predictor.pt']

                for connector_file in connector_files:
                    connector_path = os.path.join(self.graph_lm_connector_path, connector_file)
                    if os.path.exists(connector_path):
                        self.connectors[connector_file] = torch.load(connector_path, map_location=device)
                        print(f"✓ Connector {connector_file} loaded")
                    else:
                        print(f"⚠ Connector {connector_file} not found")

            print("✓ All Llamole graph models loaded successfully")

        except Exception as e:
            print(f"❌ Failed to load Llamole graph models: {e}")
            print("Falling back to language model only")
            self.use_llamole_graph_models = False

    def setup_llamole_model(self):
        """设置完整的Llamole模型"""
        if not self.use_llamole_graph_models:
            return

        print("Setting up complete Llamole model with graph components...")

        # 创建token_id_dict（用于特殊token）
        token_id_dict = {}
        new_special_tokens = ["<design_start>", "<design_end>", "<design_body>", "<molecule>", "<retro_start>", "<retro_end>", "<retro_body>", "<rollback_start>", "<rollback_end>"]
        for token in new_special_tokens:
            if token in self.tokenizer.get_vocab():
                token_id_dict[token] = self.tokenizer.get_vocab()[token]

        # 创建Llamole模型
        try:
            llamole_model = GraphLLMForCausalMLM(
                model_args=None,  # 我们不需要完整的model_args
                finetuning_args=None,
                data_args=None,
                language_model=self.model,
                graph_decoder=self.graph_decoder,
                graph_predictor=self.graph_predictor,
                graph_encoder=self.graph_encoder,
                token_id_dict=token_id_dict,
                tokenizer=self.tokenizer,
            )

            # 设置连接器
            if self.connectors:
                if 'graph_to_lm_connector.pt' in self.connectors:
                    llamole_model.graph_to_lm_connector.load_state_dict(self.connectors['graph_to_lm_connector.pt'])
                if 'lm_to_graph_decoder.pt' in self.connectors:
                    llamole_model.lm_to_graph_decoder.load_state_dict(self.connectors['lm_to_graph_decoder.pt'])
                if 'lm_to_graph_predictor.pt' in self.connectors:
                    llamole_model.lm_to_graph_predictor.load_state_dict(self.connectors['lm_to_graph_predictor.pt'])

            self.model = llamole_model
            print("✓ Complete Llamole model setup successful")

        except Exception as e:
            print(f"❌ Failed to setup Llamole model: {e}")
            print("Falling back to language model only")
            self.use_llamole_graph_models = False

    def setup_lora_config(self) -> LoraConfig:
        """设置LoRA配置"""
        modules_to_save = self.lora_config.get("modules_to_save", None)
        if modules_to_save:
            # 显式保留 Llamole 的化学 token 适配：
            # Llamole 适配器里含 embed_tokens / lm_head 的 modules_to_save，
            # GBM LoRA 若不带，训练后保存的适配器会丢掉这两项，
            # 导致 <molecule> / <design_body> 等特殊 token 的表示退回基座。
            modules_to_save = list(modules_to_save)

        lora_config = LoraConfig(
            r=self.lora_config.get("r", 16),
            lora_alpha=self.lora_config.get("lora_alpha", 32),
            target_modules=self.lora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            lora_dropout=self.lora_config.get("lora_dropout", 0.05),
            bias=self.lora_config.get("bias", "none"),
            task_type=self.lora_config.get("task_type", "CAUSAL_LM"),
            modules_to_save=modules_to_save,
        )

        return lora_config

    def prepare_model_for_training(self):
        """准备模型进行LoRA训练"""
        # 应用LoRA配置
        lora_config = self.setup_lora_config()
        self.model = get_peft_model(self.model, lora_config)

        # 梯度检查点：LoRA 微调时必须开启，否则 7.6B 模型 + 长 prompt 的激活值
        # 会把 24GB 卡撑爆。配合 use_cache=False。
        if self.lora_config.get("gradient_checkpointing", False):
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            print("✓ gradient checkpointing enabled")
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

        # 打印可训练参数
        self.model.print_trainable_parameters()

        # 根据环境变量决定是否准备模型用于kbit训练（内存敏感环境可跳过）
        skip_kbit = os.environ.get("SKIP_KBIT_PREP", "0") == "1"
        if skip_kbit:
            print("SKIP_KBIT_PREP=1 -> 跳过 prepare_model_for_kbit_training()（减少显存峰值）")
        else:
            try:
                # 准备模型用于kbit训练（如果需要）
                self.model = prepare_model_for_kbit_training(self.model)
            except Exception as e:
                # 如果在准备过程中出现 OOM 或其他错误，记录并重新抛出以便上层处理
                print(f"Warning: prepare_model_for_kbit_training failed: {e}")
                raise

        return self.model

    def load_dataset(self, dataset_path: str) -> Dataset:
        """加载训练数据集 - 支持新旧两种格式"""
        print(f"Loading dataset from {dataset_path}")

        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 转换为datasets格式
        formatted_data = []
        for item in data:
            # 检查数据集格式
            if 'response' in item and 'instruction' in item:
                # 新格式: instruction + response (评估指标导向数据集)
                text = f"Instruction: {item['instruction']}\nResponse: {item['response']}"

                formatted_data.append({
                    "text": text,
                    "instruction": item["instruction"],
                    "input": "",  # 空输入
                    "output": item["response"]
                })
            elif 'input' in item and 'output' in item:
                # 旧格式: instruction + input + output
                # 与推理端 (generate_with_new_adapter.py) 严格对齐:
                # "Input:" 后不带空格, input 为空时不留尾随空格
                _mid = f"\nInput: {item['input']}" if item['input'] else "\nInput:"
                text = f"Instruction: {item['instruction']}{_mid}\nResponse: {item['output']}"

                formatted_data.append({
                    "text": text,
                    "instruction": item["instruction"],
                    "input": item["input"],
                    "output": item["output"]
                })
            else:
                raise ValueError(f"Unsupported dataset format for item: {item.keys()}")

        dataset = Dataset.from_list(formatted_data)
        print(f"Loaded {len(dataset)} training samples")

        return dataset

    def tokenize_function(self, examples):
        """token化函数

        关键: 只对 response 段计算 loss（label = -100 mask 掉 prompt 部分）。

        旧实现 labels = input_ids（全序列 loss），实测 instruction 73 token /
        response 14 token -> 84% 的梯度花在背 prompt 模板，只有 16% 在教化学，
        是数据塌陷之外的第二个致命伤。

        训练文本形态为:
            Instruction: {instruction}\nInput: {input}\nResponse: {output}
        其中 "Response:" 之后的 token 才参与 loss。
        """
        texts = []
        for instruction, input_text, output in zip(
            examples["instruction"],
            examples["input"],
            examples["output"]
        ):
            _mid = f"\nInput: {input_text}" if input_text else "\nInput:"
            text = f"Instruction: {instruction}{_mid}\nResponse: {output}"
            texts.append(text)

        # token化
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=self.lora_config.get("max_length", 1024),
            return_tensors=None
        )

        # 关键: 显式追加 EOS。
        # Qwen2 tokenizer 默认 add_special_tokens=True 但**不会**加 EOS
        # （实测 "Response: CCC" -> [2582, 25, 91474]，无 eos）。
        # 于是模型从未学到"该停了"，推理时一路跑题输出混合物 + 解释文字。
        # 追加 EOS 后，模型学到 response 结束 -> 立刻 <|im_end|>。
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            # 关键: 追加 EOS 时必须**同步**维护 attention_mask。
            # 旧实现只给 input_ids 追加 EOS, attention_mask 保持原长度,
            # 两者差 1 -> Qwen2 的 sdpa 把 mask 扩成 [b,1,q_len,kv_len] 时
            # 报 "expanded size (447) must match existing size (446)",
            # 训练在第一个 batch 就崩。
            for key in ("input_ids",):
                seqs = tokenized.get(key, [])
                new_seqs = []
                for ids in seqs:
                    if not ids or ids[-1] != eos_id:
                        ids = list(ids) + [eos_id]
                    new_seqs.append(ids)
                tokenized[key] = new_seqs

            masks = tokenized.get("attention_mask")
            if masks is not None:
                aligned = []
                for m, ids in zip(masks, tokenized["input_ids"]):
                    m = list(m)
                    if len(m) < len(ids):
                        m = m + [1] * (len(ids) - len(m))
                    elif len(m) > len(ids):
                        m = m[:len(ids)]
                    aligned.append(m)
                tokenized["attention_mask"] = aligned

        if "input_ids" not in tokenized:
            return tokenized

        response_mask = self.lora_config.get("response_only_loss", True)
        if response_mask:
            # 逐样本找出 "Response:" 之后的位置，把之前的 label 全部置 -100
            resp_ids = self.tokenizer("Response:", add_special_tokens=False)["input_ids"]
            n_resp = len(resp_ids)
            new_labels = []
            for ids in tokenized["input_ids"]:
                labels = list(ids)
                start = None
                if n_resp > 0:
                    # 找 "Response:" token 序列的结束位置
                    for j in range(0, len(ids) - n_resp + 1):
                        if ids[j:j + n_resp] == resp_ids:
                            start = j + n_resp
                            break
                if start is None:
                    # 找不到标记 -> 退回只保留最后 1/4 的 token（保守策略）
                    start = max(0, len(ids) - max(1, len(ids) // 4))
                for j in range(min(start, len(labels))):
                    labels[j] = -100
                new_labels.append(labels)
            tokenized["labels"] = new_labels
        else:
            tokenized["labels"] = [list(ids) for ids in tokenized["input_ids"]]

        return tokenized

    def setup_training_args(self, output_dir: str) -> TrainingArguments:
        """设置训练参数"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.lora_config.get("num_epochs", 3),
            per_device_train_batch_size=self.lora_config.get("batch_size", 4),
            per_device_eval_batch_size=self.lora_config.get("batch_size", 4),
            gradient_accumulation_steps=self.lora_config.get("gradient_accumulation_steps", 1),
            optim=self.lora_config.get("optim", "adamw_torch"),
            save_steps=self.lora_config.get("save_steps", 100),
            logging_steps=self.lora_config.get("logging_steps", 10),
            learning_rate=self.lora_config.get("learning_rate", 2e-4),
            fp16=self.lora_config.get("fp16", True),
            bf16=self.lora_config.get("bf16", False),
            max_grad_norm=self.lora_config.get("max_grad_norm", 0.3),
            warmup_ratio=self.lora_config.get("warmup_ratio", 0.03),
            lr_scheduler_type=self.lora_config.get("lr_scheduler_type", "cosine"),
            remove_unused_columns=self.lora_config.get("remove_unused_columns", False),
            evaluation_strategy=self.lora_config.get("evaluation_strategy", "no"),
            save_strategy=self.lora_config.get("save_strategy", "steps"),
            load_best_model_at_end=self.lora_config.get("load_best_model_at_end", False),
            report_to=self.lora_config.get("report_to", "none"),
            # 7.6B fp16 模型 + 764 token 长 prompt，激活值显存吃紧，
            # 不开梯度检查点会 OOM（实测 GPU0 自身已占 20.88GB 后再申请 48MB 失败）
            gradient_checkpointing=self.lora_config.get("gradient_checkpointing", False),
            gradient_checkpointing_kwargs=self.lora_config.get(
                "gradient_checkpointing_kwargs", {"use_reentrant": False}),
        )

        return training_args

    def train(self, dataset_path: str, output_dir: str, llamole_adapter_path: str = None):
        """执行LoRA微调"""
        # 加载模型和tokenizer
        self.load_model_and_tokenizer(llamole_adapter_path)

        # 准备模型训练
        self.prepare_model_for_training()

        # 加载数据集
        dataset = self.load_dataset(dataset_path)

        # 分割训练集和验证集
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        # token化数据集
        tokenized_train = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )

        tokenized_eval = eval_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names
        )

        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
        )

        # 设置训练参数
        training_args = self.setup_training_args(output_dir)

        # 创建trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
        )

        # 开始训练
        print("Starting LoRA fine-tuning...")
        trainer.train()

        # 保存模型
        trainer.save_model(output_dir)
        print(f"Model saved to {output_dir}")

        # 规范化 safetensors 的 key 前缀 + 校验 LoRA 权重真的落盘
        self.normalize_saved_adapter(output_dir)

        return trainer

    def normalize_saved_adapter(self, output_dir: str):
        """规范化保存的适配器 key 前缀并做完整性校验。

        问题：训练时基座已经是 merge 过的 PeftModel（merge_and_unload 后返回裸模型，
        但 save_pretrained 走的是 Trainer -> PeftModel 的路径），不同 PEFT 版本
        保存出的 key 前缀可能是
            base_model.model.base_model.model.model.layers.0...   (双层)
        而 PEFT 加载时期望
            base_model.model.model.layers.0...                     (单层)
        前缀不匹配时 PeftModel.from_pretrained 会打印 missing keys 警告后
        **静默跳过**（lora_B 初始化为 0 -> 恒等映射 -> 等价于未微调基座）。

        这里做两件事：
          1) 把 key 前缀统一成 PEFT 期望的单层形式
          2) 加载回来校验 lora_B 非零 + modules_to_save 未丢失
        """
        import json
        import shutil
        from pathlib import Path

        from safetensors.torch import load_file, save_file

        out = Path(output_dir)
        st_path = out / "adapter_model.safetensors"
        if not st_path.exists():
            print("⚠ 未找到 adapter_model.safetensors，跳过规范化")
            return

        state = load_file(str(st_path))
        DOUBLE = "base_model.model.base_model.model."
        SINGLE = "base_model.model."

        normalized = {}
        n_changed = 0
        for k, v in state.items():
            new_k = k
            if k.startswith(DOUBLE):
                new_k = SINGLE + k[len(DOUBLE):]
                n_changed += 1
            elif k.startswith(SINGLE):
                new_k = k
            normalized[new_k] = v

        if n_changed:
            save_file(normalized, str(st_path))
            print(f"✓ 规范化 {n_changed} 个 key 前缀 -> {st_path}")
        else:
            print("✓ key 前缀已符合 PEFT 期望")

        # 校验：lora_B 非零
        b_means = [float(v.abs().mean()) for k, v in normalized.items() if "lora_B" in k]
        if b_means:
            nz = sum(1 for v in b_means if v > 1e-8)
            print(f"  lora_B: {len(b_means)} 个, 非零 {nz}, "
                  f"mean|A|={sum(b_means)/len(b_means):.6f}")
            if nz == 0:
                raise RuntimeError("保存的 LoRA 权重全为零，训练未生效！")

        has_embed = any("embed_tokens" in k for k in normalized)
        has_lm_head = any("lm_head" in k for k in normalized)
        print(f"  modules_to_save: embed_tokens={has_embed} lm_head={has_lm_head}")

        # 写一个 adapter_config 的备份说明，便于排查
        cfg_path = out / "adapter_config.json"
        if cfg_path.exists():
            cfg = json.load(open(cfg_path))
            print(f"  adapter_config: r={cfg.get('r')} "
                  f"target_modules={sorted(cfg.get('target_modules') or [])} "
                  f"modules_to_save={cfg.get('modules_to_save')}")

    def load_finetuned_model(self, model_path: str, llamole_adapter_path: str = None):
        """使用原始Llamole方法加载微调后的完整模型

        Args:
            model_path: GBM LoRA适配器路径
            llamole_adapter_path: 预训练的Llamole适配器路径
        """
        print(f"Loading fine-tuned model using original Llamole method from {model_path}")

        # 创建兼容的模型参数 (使用原始Llamole的ModelArguments)
        from src.hparams.model_args import ModelArguments

        # 基础模型参数 - 使用原始Qwen2
        model_args = ModelArguments(
            model_name_or_path=self.base_model_path,
            graph_decoder_path=self.graph_decoder_path if self.use_llamole_graph_models else "",
            graph_encoder_path=self.graph_encoder_path if self.use_llamole_graph_models else "",
            graph_predictor_path=self.graph_predictor_path if self.use_llamole_graph_models else "",
            graph_lm_connector_path=self.graph_lm_connector_path if self.use_llamole_graph_models else "",
            adapter_name_or_path=",".join([llamole_adapter_path, model_path] if llamole_adapter_path else [model_path]),
            flash_attn="disabled",  # 强制使用eager注意力机制
            attn_implementation=None,
            cache_dir=None,
            use_fast_tokenizer=True,
            disable_graph_model_gradient=True,
            resize_vocab=False,
            split_special_tokens=False,
            model_revision="main",
            low_cpu_mem_usage=True,
            quantization_method="bitsandbytes",
            quantization_bit=None,
            quantization_type="nf4",
            double_quantization=True,
            rope_scaling=None,
            shift_attn=False,
            mixture_of_depths=None,
            use_unsloth=False,
            new_special_tokens=",".join(["<design_start>", "<design_end>", "<design_body>", "<molecule>", "<retro_start>", "<retro_end>", "<retro_body>", "<rollback_start>", "<rollback_end>"]) if self.use_llamole_graph_models else None
        )

        # 数据参数 (最小化配置)
        from types import SimpleNamespace
        data_args = SimpleNamespace()
        data_args.learned_query_size = 8
        data_args.ignore_pad_token_for_loss = True

        # 训练参数 (推理模式)
        training_args = SimpleNamespace()
        training_args.do_train = False
        training_args.generation_max_length = 2048
        training_args.generation_num_beams = 1

        # 微调参数
        finetuning_args = SimpleNamespace()
        finetuning_args.loss_weight_lm = 1.0
        finetuning_args.loss_weight_design = 1.0
        finetuning_args.loss_weight_retro = 1.0

        # 加载tokenizer
        from src.model.loader import load_tokenizer
        tokenizer_module = load_tokenizer(model_args)
        tokenizer = tokenizer_module["tokenizer"]

        # 使用原始Llamole的from_pretrained方法加载完整模型
        try:
            from src.model.modeling_llamole import GraphLLMForCausalMLM
            print("Using original Llamole GraphLLMForCausalMLM.from_pretrained() method...")

            model = GraphLLMForCausalMLM.from_pretrained(
                tokenizer,
                model_args,
                data_args,
                training_args,
                finetuning_args,
                load_adapter=True  # 加载适配器
            )

            print("✓ Complete Llamole model loaded using original method")

        except Exception as e:
            print(f"❌ Original Llamole method failed: {e}")
            print("Falling back to manual assembly...")

            # 回退到手动组装方法
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True
            )

            # 加载适配器
            if llamole_adapter_path:
                print(f"Loading Llamole adapter: {llamole_adapter_path}")
                base_model = PeftModel.from_pretrained(base_model, llamole_adapter_path)

            print(f"Loading GBM LoRA adapter: {model_path}")
            model = PeftModel.from_pretrained(base_model, model_path)

            # 如果启用Llamole图模型，加载并集成它们
            if self.use_llamole_graph_models and LLAMOLE_AVAILABLE:
                print("Loading Llamole graph models for inference...")
                self.load_llamole_graph_models()
                if self.graph_decoder and self.graph_encoder and self.graph_predictor:
                    self.load_model_and_tokenizer()  # 确保tokenizer已加载
                    self.setup_llamole_model()
                    model = self.model  # 使用完整的Llamole模型
                    print("✓ Complete Llamole model loaded (fallback method)")
                else:
                    print("⚠ Graph models failed to load, using language model only")

        return model, tokenizer

    def generate_with_finetuned_model(self, model, tokenizer, instruction: str, input_text: str = "") -> str:
        """使用微调模型生成响应"""
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取Response部分
        if "Response:" in response:
            response = response.split("Response:")[-1].strip()

        return response


def run_gbm_lora_finetuning(
    base_model_path: str,
    dataset_path: str,
    output_dir: str,
    llamole_adapter_path: str = None,
    lora_config: Optional[Dict[str, Any]] = None
):
    """运行GBM LoRA微调的主函数

    Args:
        base_model_path: 基础模型路径 (Qwen2-7B-Instruct)
        llamole_adapter_path: 预训练的Llamole适配器路径
        dataset_path: 训练数据集路径
        output_dir: 输出目录
        lora_config: LoRA配置
    """

    # 默认LoRA配置
    if lora_config is None:
        lora_config = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "num_epochs": 3,
            "batch_size": 2,  # 小批量以适应内存
            "learning_rate": 2e-4,
            "gradient_accumulation_steps": 4,
            "save_steps": 50,
            "logging_steps": 10,
            "fp16": True
        }

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 初始化微调器
    finetuner = GBMLoRAFinetuner(base_model_path, lora_config)

    # 执行训练
    trainer = finetuner.train(dataset_path, output_dir, llamole_adapter_path)

    print("GBM LoRA fine-tuning completed!")

    return finetuner


if __name__ == "__main__":
    # 示例用法
    base_model_path = "../../saves/Llamole-Qwen2-7B-Instruct-Adapter"
    dataset_path = "../data/lora_datasets/gbm_connectivity_english.json"
    output_dir = "../../saves/GBM_LoRA_Connectivity"

    # 运行微调
    finetuner = run_gbm_lora_finetuning(
        base_model_path=base_model_path,
        dataset_path=dataset_path,
        output_dir=output_dir
    )

    # 测试微调效果
    print("\nTesting fine-tuned model...")
    model, tokenizer = finetuner.load_finetuned_model(output_dir)

    test_instruction = "Design a GBM therapeutic molecule targeting Mesenchymal-like GBM cells phenotype."
    test_input = "Drug: afatinib, Cell State: MES, Score: 0.750"

    response = finetuner.generate_with_finetuned_model(
        model, tokenizer, test_instruction, test_input
    )

    print(f"Test Instruction: {test_instruction}")
    print(f"Test Input: {test_input}")
    print(f"Generated Response: {response}")
