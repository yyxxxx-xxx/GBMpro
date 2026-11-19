"""
冻结的 Prompt Bank
==================

RL 期间 prompt **必须冻结**。原因:
    GRPO 的 prior_lls 和 agent_lls 必须在**同一条件分布**上计算。
    反馈闭环那种"每轮换 prompt(注入反馈)"的做法与 RL 不兼容 ——
    prior(冻结的 SFT 模型)从没在那种 prompt 上训练过, 其 logprob
    没有意义, KL 项会变成噪声。

prompt 集合的构造必须与 SFT 训练 / 反馈闭环**逐字符一致**:
    - 复用 `scripts/generate_with_new_adapter.py` 的 `build_prompt`
    - 推理时的完整形态: f"Instruction: {instr}\nInput:\nResponse:"
      ⚠️ 注意 `Response:` 后面**没有换行** (与训练格式一致)

每一步随机抽 n_prompts 条 prompt, 保持多样性, 避免过拟合单条 prompt。
同一 group 内的 G 个分子**必须用同一条 prompt** (GRPO 的前提)。
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
GBM_ROOT = PROJECT_ROOT / "gbm_project"

if str(GBM_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(GBM_ROOT / "scripts"))

PROMPTS_YAML = GBM_ROOT / "configs" / "english_gbm_prompts.yaml"


@dataclass
class PromptBank:
    """RL 期间冻结的 prompt 集合。"""

    prompts: List[str]
    target: str

    @property
    def fingerprint(self) -> str:
        """prompt 集合的哈希 —— 存进 checkpoint, resume 时校验一致性。"""
        h = hashlib.sha256()
        for p in self.prompts:
            h.update(p.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()[:16]

    @classmethod
    def build(
        cls,
        target: str,
        n_prompts: int = 4,
        prompts_yaml: Optional[Path] = None,
        seed: int = 0,
    ) -> "PromptBank":
        """
        构造 prompt 集合。

        遍历 CELL_STATES × METRIC_VARIANTS 取前 n_prompts 个,
        与反馈闭环第 1 轮的构造方式一致 (use_cot 交替)。
        """
        import yaml
        from generate_with_new_adapter import (
            build_prompt, CELL_STATES, METRIC_VARIANTS,
        )

        path = Path(prompts_yaml) if prompts_yaml else PROMPTS_YAML
        prompts_cfg = yaml.safe_load(open(path, encoding="utf-8"))

        out: List[str] = []
        # 交错 CELL_STATE 和 METRIC_VARIANT, 保证覆盖面
        variants = []
        for i, cell in enumerate(CELL_STATES):
            metric = METRIC_VARIANTS[i % len(METRIC_VARIANTS)]
            variants.append((cell, metric[0], i % 2 == 0))

        for cell, metric, use_cot in variants:
            if len(out) >= n_prompts:
                break
            instr = build_prompt(prompts_cfg, target, cell, metric, use_cot=use_cot)
            out.append(instr)

        if not out:
            raise RuntimeError(f"未能为靶点 {target} 构造任何 prompt")

        return cls(prompts=out, target=target)

    def sample(self, rng: random.Random, n: Optional[int] = None) -> List[str]:
        """
        随机抽 n 条 prompt (可重复)。

        ⚠️ 同一次调用返回的 prompt 列表, 每一条对应一个独立的 group ——
        group 内的 G 个分子共用其中一条。
        """
        n = n or len(self.prompts)
        return [rng.choice(self.prompts) for _ in range(n)]

    def to_dict(self) -> dict:
        return {"target": self.target, "prompts": self.prompts,
                "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, d: dict) -> "PromptBank":
        return cls(prompts=list(d["prompts"]), target=d["target"])


def format_full_prompt(instruction: str) -> str:
    """
    把 instruction 拼成训练时的完整 prompt。

    ⚠️ 格式必须与 `generate_feedback_loop.py` 的 gen_batch **逐字符一致**:
        f"Instruction: {instr}\nInput:\nResponse:"
    注意 `Response:` 后面没有换行 —— 训练时 response 是 "SMILES: xxx"。
    """
    return f"Instruction: {instruction}\nInput:\nResponse:"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="IDH1")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    bank = PromptBank.build(args.target, n_prompts=args.n)
    print(f"target={bank.target}  n={len(bank.prompts)}  "
          f"fingerprint={bank.fingerprint}\n")
    for i, p in enumerate(bank.prompts):
        print(f"--- prompt {i} ({len(p)} chars) ---")
        print(p[:300].replace("\n", " | "))
        print()
