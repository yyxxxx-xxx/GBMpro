"""
多样性过滤器
============

RL 必然向 reward 最高点收敛 (mode collapse)。REINVENT 4 用
`IdenticalMurckoScaffold` + `bucket_size` 限制每个 Murcko 骨架最多保留
N 个高分分子。

**必须与 REINVENT 配置完全一致** (`configs/reinvent_tlrl_idh1.toml`):
    [diversity_filter]
    type = "IdenticalMurckoScaffold"
    bucket_size = 10
    minscore = 0.4

否则 Ours 与 REINVENT 的多样性指标不可比 —— 审稿人一定会问
"你们的骨架多样性更高, 是不是因为过滤器更松?"

与 REINVENT 的差异 (已在论文中说明):
    REINVENT 的 filter 影响的是**训练时进入策略梯度的样本**
    (bucket 满了的样本被丢弃); 我们的实现同样如此 ——
    bucket 满了的分子不参与 advantage 计算, 避免模型反复优化同一个骨架。

用法:
    df = ScaffoldDiversityFilter(bucket_size=10, minscore=0.4)
    df.update(["CCO", "c1ccccc1", ...])   # 只收 reward >= minscore 的
    df.n_buckets, df.total_kept, df.is_full(smiles)
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
GBM_ROOT = PROJECT_ROOT / "gbm_project"

if str(GBM_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(GBM_ROOT / "src"))


def murcko_scaffold(smiles: str) -> Optional[str]:
    """取 Murcko 骨架 (无骨架则返回空串)。"""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.Scaffolds import MurckoScaffold
        RDLogger.DisableLog("rdApp")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core)
    except Exception:
        return None


class ScaffoldDiversityFilter:
    """
    IdenticalMurckoScaffold 多样性过滤器。

    每个 Murcko 骨架最多保留 bucket_size 个高分分子。
    """

    def __init__(self, bucket_size: int = 10, minscore: float = 0.4):
        self.bucket_size = bucket_size
        self.minscore = minscore
        self._buckets: Dict[str, List[float]] = defaultdict(list)
        self._seen: set = set()

    # ------------------------------------------------------------------
    def is_full(self, smiles: str) -> bool:
        """该分子所属骨架的 bucket 是否已满。"""
        core = murcko_scaffold(smiles)
        if core is None:
            return False
        return len(self._buckets.get(core, [])) >= self.bucket_size

    def update(self, smiles_list: Sequence[str], rewards: Optional[Sequence[float]] = None):
        """
        把分子加入过滤器。

        Args:
            smiles_list: 分子列表
            rewards: 对应 reward (None 则视为 1.0)。只有 reward >= minscore
                的分子才进 bucket (与 REINVENT 的 minscore 语义一致)。
        """
        for i, smi in enumerate(smiles_list):
            if smi in self._seen:
                continue
            r = float(rewards[i]) if rewards is not None else 1.0
            if r < self.minscore:
                continue
            core = murcko_scaffold(smi)
            if core is None:
                continue
            self._seen.add(smi)
            if len(self._buckets[core]) < self.bucket_size:
                self._buckets[core].append(r)

    # ------------------------------------------------------------------
    @property
    def n_buckets(self) -> int:
        return len(self._buckets)

    @property
    def total_kept(self) -> int:
        return sum(len(v) for v in self._buckets.values())

    @property
    def bucket_sizes(self) -> List[int]:
        return [len(v) for v in self._buckets.values()]

    def stats(self) -> dict:
        """给训练日志用的统计。"""
        sizes = self.bucket_sizes
        return {
            "n_buckets": self.n_buckets,
            "total_kept": self.total_kept,
            "mean_bucket": (sum(sizes) / len(sizes)) if sizes else 0.0,
            "max_bucket": max(sizes) if sizes else 0,
            "n_full_buckets": sum(1 for s in sizes if s >= self.bucket_size),
        }

    def state_dict(self) -> dict:
        return {
            "bucket_size": self.bucket_size,
            "minscore": self.minscore,
            "buckets": {k: list(v) for k, v in self._buckets.items()},
            "seen": sorted(self._seen),
        }

    def load_state_dict(self, d: dict):
        self.bucket_size = d.get("bucket_size", self.bucket_size)
        self.minscore = d.get("minscore", self.minscore)
        self._buckets = defaultdict(list, {k: list(v) for k, v in d.get("buckets", {}).items()})
        self._seen = set(d.get("seen", []))