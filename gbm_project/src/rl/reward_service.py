"""
RL 奖励服务 (Reward Service)
============================

职责:
    把"评测"从"生成"里剥离出来, 提供一个**纯函数式**的批量打分接口,
    供 GRPO / DAP 训练循环调用。

设计要点 (全部来自实测与踩坑记录, 详见 docs/RL_INTEGRATION_DESIGN.md):

1. **不复制逻辑**: 复用 `gbm_target_evaluator.build_evaluator_for_target()`,
   它内部走 `GBMPhysicalEvaluator` -> `property_pipeline.docking.VinaDocking`
   (vina Python API 1.2.7 + Meeko) + admet-ai。与 `generate_feedback_loop.py`
   **完全同源**, 保证 RL 的 reward 与历史反馈闭环结果在同一尺度。
   (实测: 重算 IDH1_history.json 里的分子, reward 差 3.4e-9 = float 精度差)

2. **进程/线程安全**: 每个 worker 必须持有**独立的** VinaDocking 实例。
   共享实例会导致 grid map 并发读写, 分数错乱
   (实测: -9.077 kcal/mol 变成 31.744)。
   -> 用 ProcessPoolExecutor, 每个子进程自己建 evaluator。

3. **pruned 分子必须给明确的最差奖励**: 硬截断命中的分子 reward = 0.0,
   不能静默跳过, 否则模型学不到"违反 BBB 硬约束是有代价的"。

4. **失败也要给奖励**: 对接失败给 reward = 0.0 并记 error,
   不能从 batch 里丢掉 (丢掉会让 group 内 advantage 计算失真)。

5. **admet-ai 锁 CPU**: 实测 admet-ai 会**自动选 GPU**
   (日志: "GPU available: True (cuda), used: True"),
   会和 7B 训练模型抢显存导致 OOM。
   -> 子进程显式设 CUDA_VISIBLE_DEVICES=""。

6. **磁盘缓存**: RL 过程中同一 SMILES 会被反复采样(尤其早期),
   缓存能省掉大量 Vina 时间 (预估命中率 20-40%)。

用法:
    from rl.reward_service import RewardService, RewardConfig
    svc = RewardService(RewardConfig(target="IDH1"))
    records = svc.score_batch(["CCO", "c1ccccc1"])
    print(records[0].reward)
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT / "gbm_project" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "gbm_project" / "src"))


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """奖励计算配置。"""

    target: str = "IDH1"
    exhaustiveness: int = 4       # RL 内循环用 4 (与 REINVENT TLRL 配置一致)
    n_poses: int = 5
    cpu_per_worker: int = 4       # 每个 Vina 实例用的核数
    n_workers: int = 15           # 60 核 / 4
    use_admet_ai: bool = True
    admet_device: str = "cpu"     # ★ 必须锁 CPU, 否则和 7B 抢显存
    cache_path: Optional[Path] = None
    use_cache: bool = True
    # 结构合理性守卫 (防 reward hacking), 见 HACKING_GUARDS
    enable_hacking_guards: bool = True


# ---------------------------------------------------------------------------
# Reward hacking 防线 (第二层)
# ---------------------------------------------------------------------------
#
# 有梯度压力时, 模型会学会"骗 scorer"而不是设计更好的分子。
# 第一层防线在 reward 函数本身 (BBB 硬截断 / ENDPOINT_FLOOR / 几何平均),
# 这一层是**结构合理性检查** —— 挡住明显不是药物的分子。
#
# 阈值参考 `scripts/rescore_parallel.py` 的类药预筛 (所有方法共用同一门槛):
#   MW <= 500, 重原子 <= 40, 可旋转键 <= 10
HACKING_GUARDS = {
    "max_rotatable_bonds": 10,
    "max_heavy_atoms": 40,
    "max_mw": 500.0,
    "max_ring_count": 6,
    "max_amide_bonds": 3,
    "max_aliphatic_chain": 12,   # 连续 sp3 碳链长度
}


@dataclass
class RewardRecord:
    """单个分子的评分结果。"""

    smiles: str
    reward: float = 0.0
    vina: Optional[float] = None
    dili: Optional[float] = None
    herg: Optional[float] = None
    bbb: Optional[float] = None
    pgp: Optional[float] = None
    ames: Optional[float] = None
    qed: Optional[float] = None
    pruned: bool = False
    prune_reason: Optional[str] = None
    error: Optional[str] = None
    guard_violation: Optional[str] = None
    elapsed: float = 0.0

    @property
    def is_valid(self) -> bool:
        """True = 分子能被正常解析并打分 (无论分数高低)。"""
        return self.error is None

    def to_dict(self) -> dict:
        return {
            "smiles": self.smiles,
            "reward": round(self.reward, 6),
            "vina": self.vina,
            "dili": self.dili,
            "herg": self.herg,
            "bbb": self.bbb,
            "pgp": self.pgp,
            "ames": self.ames,
            "qed": self.qed,
            "pruned": self.pruned,
            "prune_reason": self.prune_reason,
            "guard_violation": self.guard_violation,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# 结构守卫 (纯 RDKit, 极快, 在对接之前跑)
# ---------------------------------------------------------------------------

def check_hacking_guards(smiles: str) -> Optional[str]:
    """
    检查结构合理性。返回违规描述, 合规则返回 None。

    ⚠️ 与 `rescore_parallel._is_druglike` 的区别: 那个是**评测**用的预筛
    (挡在对接前, 通过率作为指标报告), 这个是**训练时**的守卫,
    违规直接给 reward = 0。
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        from rdkit.Chem import Lipinski

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "invalid_smiles"

        mw = Descriptors.MolWt(mol)
        heavy = mol.GetNumHeavyAtoms()
        rot = Descriptors.NumRotatableBonds(mol)
        rings = mol.GetRingInfo().NumRings()

        if mw > HACKING_GUARDS["max_mw"]:
            return f"guard_mw={mw:.0f}"
        if heavy > HACKING_GUARDS["max_heavy_atoms"]:
            return f"guard_heavy={heavy}"
        if rot > HACKING_GUARDS["max_rotatable_bonds"]:
            return f"guard_rot={rot}"
        if rings > HACKING_GUARDS["max_ring_count"]:
            return f"guard_rings={rings}"

        amides = len(mol.GetSubstructMatches(
            Chem.MolFromSmarts("[NX3][CX3](=[OX1])")))
        if amides > HACKING_GUARDS["max_amide_bonds"]:
            return f"guard_amide={amides}"

        # 连续 sp3 碳链 (防长链烷烃 —— 实测基线里出现过 rot=147 的退化产物)
        longest = _longest_aliphatic_chain(mol)
        if longest > HACKING_GUARDS["max_aliphatic_chain"]:
            return f"guard_chain={longest}"

    except Exception as e:  # 守卫本身不该让整个 batch 挂掉
        return f"guard_error:{type(e).__name__}"
    return None


def _longest_aliphatic_chain(mol) -> int:
    """最长连续 sp3 碳链长度 (相邻 C-C 单键, 都非芳香)。"""
    try:
        from rdkit import Chem
        longest = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "C" or atom.GetIsAromatic():
                continue
            # BFS 找最长链
            seen = set()
            stack = [(atom, 1)]
            while stack:
                cur, length = stack.pop()
                if cur.GetIdx() in seen:
                    continue
                seen.add(cur.GetIdx())
                longest = max(longest, length)
                for nb in cur.GetNeighbors():
                    if (nb.GetSymbol() == "C" and not nb.GetIsAromatic()
                            and nb.GetIdx() not in seen
                            and not any(b.GetIsAromatic() for b in nb.GetBonds())):
                        stack.append((nb, length + 1))
        return longest
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# 子进程 worker
# ---------------------------------------------------------------------------

_WORKER_EVALUATOR = None
_WORKER_CFG = None


def _get_worker_evaluator(cfg: RewardConfig):
    """
    每个 worker 进程建自己的 evaluator。

    ⚠️ 必须独立实例: 共享 Vina 实例会导致 grid map 并发竞争,
    分数错乱 (实测 -9.077 -> 31.744)。
    """
    global _WORKER_EVALUATOR, _WORKER_CFG
    if _WORKER_EVALUATOR is None or _WORKER_CFG != cfg:
        # admet-ai 会自动抢 GPU, 这里锁死 CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from gbm_target_evaluator import build_evaluator_for_target
        ev = build_evaluator_for_target(
            cfg.target,
            exhaustiveness=cfg.exhaustiveness,
            cpu=cfg.cpu_per_worker,
            n_poses=cfg.n_poses,
            use_admet_ai=cfg.use_admet_ai,
        )
        if ev is None:
            raise RuntimeError(
                f"无法为靶点 {cfg.target} 构建评估器 "
                f"(检查 data/receptors/ 与 configs/receptors.yaml)")
        _WORKER_EVALUATOR = ev
        _WORKER_CFG = cfg
    return _WORKER_EVALUATOR


def _score_one(args) -> dict:
    """子进程内评分单个分子。返回 dict (可 pickle)。"""
    smiles, cfg = args
    t0 = time.time()
    try:
        if cfg.enable_hacking_guards:
            violation = check_hacking_guards(smiles)
            if violation is not None:
                rec = RewardRecord(
                    smiles=smiles, reward=0.0,
                    guard_violation=violation,
                    prune_reason=f"hacking_guard: {violation}",
                    pruned=True,
                    elapsed=time.time() - t0,
                )
                return rec.to_dict()

        ev = _get_worker_evaluator(cfg)
        res = ev.evaluate(smiles)

        # 硬截断命中的分子: reward 强制为 0
        reward = 0.0 if res.is_pruned else float(res.reward)
        rec = RewardRecord(
            smiles=smiles,
            reward=reward,
            vina=res.vina_score,
            dili=res.dili_prob,
            herg=res.herg_prob,
            bbb=res.bbb_score,
            pgp=res.pgp_prob,
            ames=res.ames_prob,
            qed=res.qed,
            pruned=bool(res.is_pruned),
            prune_reason=res.prune_reason,
            elapsed=time.time() - t0,
        )
        return rec.to_dict()

    except Exception as e:
        return RewardRecord(
            smiles=smiles, reward=0.0,
            error=f"{type(e).__name__}: {str(e)[:200]}",
            elapsed=time.time() - t0,
        ).to_dict()


# ---------------------------------------------------------------------------
# 磁盘缓存
# ---------------------------------------------------------------------------

class RewardCache:
    """JSONL append-only 缓存。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mem: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._fh = None
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        self._mem[d["smiles"]] = d
                    except Exception:
                        continue

    def get(self, smiles: str) -> Optional[dict]:
        return self._mem.get(smiles)

    def put(self, rec: dict):
        with self._lock:
            self._mem[rec["smiles"]] = rec
            if self._fh is None:
                self._fh = open(self.path, "a", encoding="utf-8")
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __len__(self):
        return len(self._mem)


# ---------------------------------------------------------------------------
# 主服务
# ---------------------------------------------------------------------------

class RewardService:
    """
    批量奖励计算服务。

    用法:
        svc = RewardService(RewardConfig(target="IDH1", cache_path=Path(".../cache.jsonl")))
        records = svc.score_batch(["CCO", "c1ccccc1"])
    """

    def __init__(self, cfg: Optional[RewardConfig] = None):
        self.cfg = cfg or RewardConfig()
        self.cache = (RewardCache(self.cfg.cache_path)
                      if (self.cfg.use_cache and self.cfg.cache_path) else None)
        self._executor: Optional[ProcessPoolExecutor] = None

    def _get_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.cfg.n_workers)
        return self._executor

    def score_batch(self, smiles_list: Sequence[str]) -> List[RewardRecord]:
        """
        批量打分。返回与输入等长的 RewardRecord 列表。

        顺序保证: 第 i 个 record 对应第 i 个输入 SMILES。
        """
        smiles_list = list(smiles_list)
        if not smiles_list:
            return []

        results: List[Optional[dict]] = [None] * len(smiles_list)

        # 1) 查缓存
        todo_idx = []
        for i, smi in enumerate(smiles_list):
            hit = self.cache.get(smi) if self.cache else None
            if hit is not None:
                results[i] = hit
            else:
                todo_idx.append(i)

        # 2) 并行计算缺失部分
        if todo_idx:
            uniq: Dict[str, int] = {}
            for i in todo_idx:
                uniq.setdefault(smiles_list[i], i)
            uniq_smiles = list(uniq.keys())

            ex = self._get_executor()
            payload = [(s, self.cfg) for s in uniq_smiles]
            for smi, d in zip(uniq_smiles, ex.map(_score_one, payload)):
                if self.cache:
                    self.cache.put(d)
                results[uniq[smi]] = d

        return [RewardRecord(**{k: v for k, v in r.items()
                                if k in RewardRecord.__dataclass_fields__})
                for r in results]

    def score(self, smiles: str) -> RewardRecord:
        return self.score_batch([smiles])[0]

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        if self.cache:
            self.cache.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="IDH1")
    ap.add_argument("--smiles", nargs="+", default=[
        "CCC(N(Cc1ccccc1F)S(=O)(=O)CC(F)(F)F)F",
        "CC(C)Nc1ncc(-c2cnc(N)nc2)c2ccccc12",
    ])
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()

    cfg = RewardConfig(target=args.target, n_workers=args.workers,
                       cache_path=Path(args.cache) if args.cache else None)
    with RewardService(cfg) as svc:
        t0 = time.time()
        recs = svc.score_batch(args.smiles)
        dt = time.time() - t0
        for r in recs:
            print(f"  reward={r.reward:.4f}  vina={r.vina}  bbb={r.bbb}  "
                  f"pruned={r.pruned} {r.prune_reason or ''}")
        print(f"\n{len(recs)} 分子 / {dt:.1f}s -> {dt/len(recs):.2f}s/分子")
        if svc.cache:
            print(f"缓存条目: {len(svc.cache)}")
