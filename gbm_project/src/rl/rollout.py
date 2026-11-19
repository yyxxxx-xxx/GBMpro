"""
Rollout 采样引擎
================

从 SFT 模型采样 SMILES, 供 GRPO 计算 advantage。

实测性能 (单卡 RTX 3090, IDH1, 220 max_new_tokens):
    B=16:  10.7 s  (0.666 s/分子)  峰值 15.3 GB
    B=32:  11.0 s  (0.343 s/分子)  峰值 15.6 GB
    -> 比 Vina 对接 (0.86 s/分子) 还快 2.5 倍, **GPU 不是瓶颈**

所以 **不需要 vLLM**: HF `generate` 已经足够快, 而 vLLM 会引入
RL LoRA 权重热更新的复杂度。

⚠️ 实测踩坑: **生成阶段必须关掉 gradient checkpointing**。
`gradient_checkpointing_enable()` 会强制 `use_cache=False`, KV cache
失效 -> 生成从 11 s 变成数分钟 (实测 B=16 卡住 > 60 s 无输出)。
正确做法是生成/训练两个阶段切换。

⚠️ SMILES 抽取必须复用 `generate_with_new_adapter.py` 的
`extract_candidate` + `repair_smiles` + `_heavy_atoms`,
**与反馈闭环完全同一套** (否则前后不可比)。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
GBM_ROOT = PROJECT_ROOT / "gbm_project"

if str(GBM_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(GBM_ROOT / "scripts"))

# 与 generate_feedback_loop.py 完全同源
from generate_with_new_adapter import (  # noqa: E402
    extract_candidate, repair_smiles, _heavy_atoms,
)


@dataclass
class RolloutConfig:
    """采样配置。"""

    group_size: int = 8          # G: 每条 prompt 采多少个分子
    temperature: float = 1.0     # 与 REINVENT 一致, 保证探索
    top_p: float = 0.95
    max_new_tokens: int = 220    # ⚠️ 128 会截断长 SMILES (已实测)
    min_heavy_atoms: int = 12
    repetition_penalty: float = 1.1
    seed: int = 0


@dataclass
class RolloutSample:
    """单个采样结果。"""

    prompt_id: int
    group_idx: int
    raw_response: str = ""
    smiles: Optional[str] = None
    n_tokens: int = 0
    valid: bool = False
    error: Optional[str] = None


class RolloutEngine:
    """
    基于 HF `generate` 的采样器。

    用法:
        eng = RolloutEngine(model, tokenizer, cfg)
        samples = eng.generate(prompts, n_groups=4, group_size=8)
    """

    def __init__(self, model, tokenizer, cfg: Optional[RolloutConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg or RolloutConfig()

    def generate(
        self,
        prompts: Sequence[str],
        n_groups: int = 4,
        group_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[RolloutSample]:
        """
        对每条 prompt 采 group_size 个分子。

        Args:
            prompts: 完整 prompt (已含 "Instruction: ...\nInput:\nResponse:")
            n_groups: 用前 n_groups 条 prompt
            group_size: 每条 prompt 采多少个 (默认 cfg.group_size)

        Returns:
            展平的 RolloutSample 列表, 顺序 = group 优先 (便于还原 group)
        """
        import torch

        G = group_size or self.cfg.group_size
        use_prompts = list(prompts)[:n_groups]
        samples: List[RolloutSample] = []

        # ⚠️ 生成阶段: 关 gradient checkpointing, 开 KV cache
        was_training = self.model.training
        self.model.eval()
        self.model.config.use_cache = True
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

        try:
            for pid, prompt in enumerate(use_prompts):
                enc = self.tokenizer(prompt, return_tensors="pt").to(
                    self.model.device)
                prompt_len = enc["input_ids"].shape[1]

                if seed is not None:
                    torch.manual_seed(seed + pid * 1000)

                with torch.no_grad():
                    out = self.model.generate(
                        **enc,
                        max_new_tokens=self.cfg.max_new_tokens,
                        temperature=self.cfg.temperature,
                        top_p=self.cfg.top_p,
                        do_sample=True,
                        repetition_penalty=self.cfg.repetition_penalty,
                        num_return_sequences=G,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                for gi in range(G):
                    resp_ids = out[gi][prompt_len:]
                    text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
                    samples.append(self._parse(prompt_id=pid, group_idx=gi,
                                              text=text,
                                              n_tokens=int(resp_ids.shape[0])))
        finally:
            # 恢复训练模式 (训练阶段会自己再开 gradient checkpointing)
            if was_training:
                self.model.train()

        return samples

    def _parse(self, prompt_id: int, group_idx: int,
               text: str, n_tokens: int) -> RolloutSample:
        """从生成文本里抽 SMILES (与反馈闭环同源)。"""
        s = RolloutSample(prompt_id=prompt_id, group_idx=group_idx,
                          raw_response=text, n_tokens=n_tokens)
        try:
            cand = extract_candidate(text)
            if not cand:
                s.error = "no_candidate"
                return s
            fixed, _ = repair_smiles(cand)
            if fixed is None:
                s.error = "invalid_smiles"
                return s
            if _heavy_atoms(fixed) < self.cfg.min_heavy_atoms:
                s.error = f"too_few_heavy_atoms({_heavy_atoms(fixed)})"
                return s
            s.smiles = fixed
            s.valid = True
        except Exception as e:
            s.error = f"{type(e).__name__}: {str(e)[:100]}"
        return s


def group_samples(samples: Sequence[RolloutSample]) -> dict:
    """按 prompt_id 分组, 保持组内顺序。"""
    groups: dict = {}
    for s in samples:
        groups.setdefault(s.prompt_id, []).append(s)
    return groups
