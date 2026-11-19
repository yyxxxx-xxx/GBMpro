"""
分子性质计算管线（本地自动化）
=================================

本模块提供统一的、与 baseline 无关的性质计算能力，替代原先"人工到外部网站逐个查询"的流程。

三个子模块:
    descriptors.py  —— RDKit 描述符 + PAINS/Brenk 过滤 + SA score
    admet.py        —— admet-ai 机器学习 ADMET 预测 (BBB/Pgp/hERG/DILI/AMES/QED...)
    docking.py      —— AutoDock Vina 真实对接（替代原先的伪对接）

设计原则:
    1. 所有 baseline 走同一套代码、同一个版本 —— 这是对比实验公平性的技术保证
    2. 每个量都有明确的口径说明（口径不一致是审稿人最容易攻击的点）
    3. 失败要可追溯（返回 NaN + error 字段，而不是静默丢弃）
"""

from .descriptors import compute_descriptors, DESCRIPTOR_COLUMNS
from .admet import compute_admet, ADMET_COLUMNS
from .docking import VinaDocking, DOCKING_COLUMNS
from .scaffold import compute_scaffold_metrics, SCAFFOLD_COLUMNS

__all__ = [
    "compute_descriptors",
    "DESCRIPTOR_COLUMNS",
    "compute_admet",
    "ADMET_COLUMNS",
    "VinaDocking",
    "DOCKING_COLUMNS",
    "compute_scaffold_metrics",
    "SCAFFOLD_COLUMNS",
]
