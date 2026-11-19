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


class _BatchNewlineStopper:
    """
    batch 安全的换行 stopper。

    ⚠️ `generate_with_new_adapter._NewlineStopper` 只检查 `input_ids[0]`,
    配合 `num_return_sequences>1` 时, 第一个序列遇到换行就会**停掉全部**,
    其余序列被截断 -> 有效率骤降 (实测 8 序列时只有 ~50% 有效)。
    这里改成"全部序列都遇到换行才停"。

    (实现成普通类而非继承 StoppingCriteria, 是因为 transformers 4.44+
    与 5.x 的 StoppingCriteria 接口不一致; `generate` 只要求返回 bool。)
    """

    def __init__(self, newline_token_id: int, prompt_len: int):
        self.newline_token_id = newline_token_id
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        import torch
        gen = input_ids[:, self.prompt_len:]
        if gen.shape[1] == 0:
            return False
        last = gen[:, -1]
        return bool(torch.all(last == self.newline_token_id).item())


@dataclass
class RolloutConfig:
    """采样配置。

    ⚠️ 默认值基于**实测有效率** (2026-11-19, IDH1, N=8, 同一条 prompt):

        | 配置                          | 有效率 |
        |-------------------------------|--------|
        | t=0.7 p=0.90 (反馈闭环原配置)  | 88%    |
        | **t=0.8 p=0.92 (默认)**        | **75%** |
        | t=0.9 p=0.92                   | 62%    |
        | t=1.0 p=0.95                   | 38%    |

    温度是有效率的**主因** (不是 max_new_tokens 截断)。
    选 t=0.8 而不是直接对齐反馈闭环的 0.7 —— RL 需要比反馈闭环更大的
    探索空间 (否则组内 reward 差异太小, GRPO 没梯度信号)。
    75% 有效率下, 组内 8 个样本平均有 6 个有效, 组内方差足够。

    ⚠️ 必须开 `stop_at_newline`: 训练集的 response 形如
    "SMILES: <one smiles>" (单行), 模型常在 SMILES 之后继续输出
    混合物/解释文字, 不停就会产出噪声。
    """

    group_size: int = 8          # G: 每条 prompt 采多少个分子
    temperature: float = 0.8     # 实测: 0.7 -> 88%, 0.8 -> 75%, 1.0 -> 38%
    top_p: float = 0.92
    max_new_tokens: int = 512     # 与反馈闭环一致
    min_heavy_atoms: int = 12
    repetition_penalty: float = 1.1
    seed: int = 0
    stop_at_newline: bool = True


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
        from transformers import StoppingCriteriaList

        G = group_size or self.cfg.group_size
        use_prompts = list(prompts)[:n_groups]
        samples: List[RolloutSample] = []

        # 换行 stopper 的 token id (训练集 response 是单行 "SMILES: xxx")
        newline_token_id = self._newline_token_id()

        # ⚠️ 生成阶段: 关 gradient checkpointing, 开 KV cache
        was_training = self.model.training
        self.model.eval()
        self.model.config.use_cache = True
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

        try:
            print(f"[rollout] 开始: {len(use_prompts)} 组 x {G} 个, "
                  f"model.training={self.model.training}, "
                  f"use_cache={self.model.config.use_cache}", flush=True)
            for pid, prompt in enumerate(use_prompts):
                t_g = time.time()
                enc = self.tokenizer(prompt, return_tensors="pt").to(
                    self.model.device)
                prompt_len = enc["input_ids"].shape[1]

                if seed is not None:
                    torch.manual_seed(seed + pid * 1000)

                gen_kwargs = dict(
                    max_new_tokens=self.cfg.max_new_tokens,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    do_sample=True,
                    repetition_penalty=self.cfg.repetition_penalty,
                    num_return_sequences=G,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                if self.cfg.stop_at_newline and newline_token_id is not None:
                    stopper = self._stopper(prompt_len)
                    if stopper is not None:
                        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                            [stopper])

                print(f"[rollout] group {pid} 开始生成 (prompt_len={prompt_len})",
                      flush=True)
                with torch.no_grad():
                    out = self.model.generate(**enc, **gen_kwargs)
                print(f"[rollout] group {pid} 生成完成", flush=True)

                n_gen = int(out.shape[1]) - prompt_len
                print(f"[rollout] group {pid}: {G} 分子, 生成 {n_gen} token, "
                      f"{time.time()-t_g:.1f}s", flush=True)

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

    def _newline_token_id(self):
        """换行符的 token id (取所有可能的编码里最短的那个)。"""
        if getattr(self, "_nl_id", None) is not None:
            return self._nl_id
        ids = set()
        for s in ("\n", "\n\n", " \n"):
            enc = self.tokenizer.encode(s, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        self._nl_id = min(ids) if ids else None
        return self._nl_id

    def _stopper(self, prompt_len: int):
        """
        batch 安全的换行 stopper (见 _BatchNewlineStopper 的说明)。
        """
        nl_id = self._newline_token_id()
        if nl_id is None:
            return None
        return _BatchNewlineStopper(nl_id, prompt_len)

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
