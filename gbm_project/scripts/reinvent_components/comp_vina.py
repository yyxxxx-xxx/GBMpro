"""
REINVENT4 自定义评分组件：AutoDock Vina 对接分数
==================================================

用途:
    让 REINVENT 成为**靶点条件化**的基线 —— 采样时用真实 Vina 对接分数
    做奖励信号，从而生成与靶点口袋匹配的分子。

    这是评审最严格的基线形式：如果连"用真实对接分数做奖励的化学语言模型"
    都比不过我们的模型，结论才站得住脚。

接口:
    继承 reinvent 的 component 协议:
        __parameters  -> dataclass, 声明配置字段
        __component   -> 类, 实现 calculate() 返回 (分数数组, 是否有效的掩码)
"""

from __future__ import annotations

__all__ = ["VinaScore"]

import os
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from reinvent_plugins.components.component_results import ComponentResults
from reinvent_plugins.components.add_tag import add_tag

logger = logging.getLogger("reinvent")


def _flatten(values) -> tuple:
    """
    把配置里的坐标展平为 3 元组。

    REINVENT 的 TOML 解析会把 params 里的 list 保留为 list，
    而 VinaDocking 需要 float 标量。这里做防御性展平，
    避免 "float() argument must be ... not 'list'"。
    """
    if values is None:
        return (0.0, 0.0, 0.0)
    if isinstance(values, (list, tuple)):
        flat = []
        for v in values:
            if isinstance(v, (list, tuple)):
                flat.extend(float(x) for x in v)
            else:
                flat.append(float(v))
        return tuple(flat[:3])
    return tuple(float(values) for _ in range(3))


def _scalar(value):
    """
    取标量值。

    REINVENT 的 TOML 解析对 params 中的标量也可能保留为单元素 list，
    这里统一拆包。
    """
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


@add_tag("__parameters")
@dataclass
class Parameters:
    """组件配置。"""
    receptor_pdbqt: str
    center: List[float]
    size: List[float]
    exhaustiveness: int = 8
    n_poses: int = 5
    cpu: int = 4
    n_workers: int = 8


# ---------------- 子进程 worker ----------------

_WORKER_DOCKER = None
_WORKER_PARAMS = None


def _dock_in_worker(smiles: str, params) -> Optional[float]:
    """
    子进程内对接单个分子。

    每个 worker 进程缓存自己的 Vina 实例，grid maps 跨分子复用。
    """
    global _WORKER_DOCKER, _WORKER_PARAMS

    if _WORKER_DOCKER is None or _WORKER_PARAMS is not params:
        import importlib.util
        import sys

        docking_path = os.environ.get(
            "GBM_DOCKING_MODULE",
            "/root/Llamole-main/gbm_project/src/property_pipeline/docking.py",
        )
        if "property_pipeline.docking" not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                "property_pipeline.docking", docking_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["property_pipeline.docking"] = module
            spec.loader.exec_module(module)
        else:
            module = sys.modules["property_pipeline.docking"]

        _WORKER_DOCKER = module.VinaDocking(
            receptor_pdbqt=_scalar(params.receptor_pdbqt),
            center=_flatten(params.center),
            size=_flatten(params.size),
            exhaustiveness=int(_scalar(params.exhaustiveness)),
            n_poses=int(_scalar(params.n_poses)),
            cpu=int(_scalar(params.cpu)),
        )
        _WORKER_PARAMS = params

    try:
        score, _ = _WORKER_DOCKER.dock(smiles)
        return score
    except Exception:
        return None


@add_tag("__component")
class VinaScore:
    """按 Vina 结合分数给分子打分（越负越好，转换到 [0,1]）。"""

    def __init__(self, params: Parameters):
        self.params = params
        self._docker = None
        self._cache = {}
        self.number_of_endpoints = 1
        self._executor = None

    # ---------------- Vina 调用 ----------------

    def _get_docker(self):
        if self._docker is None:
            # REINVENT 跑在自己的进程/venv 里，不能依赖 gbm_project 包的
            # 可导入性 —— 直接按文件路径加载 docking 模块。
            import importlib.util
            import sys

            docking_path = os.environ.get(
                "GBM_DOCKING_MODULE",
                "/root/Llamole-main/gbm_project/src/property_pipeline/docking.py",
            )
            if "property_pipeline.docking" not in sys.modules:
                spec = importlib.util.spec_from_file_location(
                    "property_pipeline.docking", docking_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules["property_pipeline.docking"] = module
                spec.loader.exec_module(module)
            else:
                module = sys.modules["property_pipeline.docking"]

            self._docker = module.VinaDocking(
                receptor_pdbqt=_scalar(self.params.receptor_pdbqt),
                center=_flatten(self.params.center),
                size=_flatten(self.params.size),
                exhaustiveness=int(_scalar(self.params.exhaustiveness)),
                n_poses=int(_scalar(self.params.n_poses)),
                cpu=int(_scalar(self.params.cpu)),
            )
        return self._docker

    def _dock_batch(self, smiles_list: List[str]) -> List[Optional[float]]:
        """
        并行对接一批分子。

        Vina 单次对接 3-15s，串行会让 RL 的每一步都变成瓶颈。
        这里用进程池并行，每个 worker 内部复用 Vina 实例
        （grid maps 跨分子复用，见 docking.py 的性能修复）。
        """
        todo = [s for s in smiles_list if s not in self._cache]
        if todo:
            n_workers = max(1, min(len(todo), int(_scalar(self.params.n_workers) or 8)))
            try:
                from concurrent.futures import ProcessPoolExecutor

                if self._executor is None:
                    self._executor = ProcessPoolExecutor(max_workers=n_workers)
                futures = {
                    self._executor.submit(_dock_in_worker, s, self.params): s
                    for s in todo
                }
                for fut, smi in futures.items():
                    try:
                        self._cache[smi] = fut.result()
                    except Exception as e:
                        logger.warning(f"Vina docking failed for {smi!r}: {e}")
                        self._cache[smi] = None
            except Exception as e:
                logger.warning(f"并行对接失败，回退串行: {e}")
                for s in todo:
                    self._cache[s] = self._dock_one_serial(s)
        return [self._cache.get(s) for s in smiles_list]

    def _dock_one_serial(self, smiles: str) -> Optional[float]:
        if isinstance(smiles, (list, tuple)):
            smiles = smiles[0] if smiles else ""
        if not isinstance(smiles, str) or not smiles:
            return None
        try:
            score, _ = self._get_docker().dock(smiles)
        except Exception as e:
            logger.warning(f"Vina docking failed for {smiles!r}: {e}")
            return None
        return score

    # ---------------- 组件协议 ----------------

    def __call__(self, smiles: List[str]) -> ComponentResults:
        """
        对一批 SMILES 计算 Vina 分数。

        协议（reinvent_plugins component）:
            __call__(smiles) -> ComponentResults
            scores 为 list[np.ndarray]，每个 endpoint 一个数组；
            计算失败必须用 NaN 表示（不能用 0）。
        """
        n = len(smiles)
        scores = np.full(n, np.nan, dtype=np.float32)

        clean = []
        for smi in smiles:
            if isinstance(smi, (list, tuple)):
                smi = smi[0] if smi else ""
            clean.append(smi if isinstance(smi, str) else "")

        raw_scores = self._dock_batch(clean)

        for i, s in enumerate(raw_scores):
            if s is None:
                continue
            # Vina 分数是负数且越负越好；线性映射到 [0,1]
            scores[i] = float(min(max(-s / 10.0, 0.0), 1.0))

        return ComponentResults(scores=[scores])
