"""
ADMET 预测（admet-ai 封装）
============================

使用 Swin-Transformer 架构的 ADMET 预训练模型 (admet-ai 2.0.1)，
在 TDC ADMET 基准数据集上训练，提供 52 个端点。本模块只封装与 GBM 药物设计
最相关、且在原始对比实验中缺失的端点。

口径说明:
    BBB permeant   <- BBB_Martins          (AUROC 0.900)  血脑屏障穿透，二分类概率
    Pgp substrate  <- Pgp_Broccatelli      (AUROC 0.947)  P-gp 转运体底物，二分类概率
    hERG           <- hERG                 (AUROC 0.839)  心脏毒性阻断概率
    DILI           <- DILI                 (AUROC 0.881)  药物性肝损伤概率
    AMES           <- AMES                 (AUROC 0.882)  致突变性概率
    QED            <- QED                  定量估计类药性 [0,1]

分类端点输出为概率 [0,1]（>0.5 视为阳性）。同时保留 drugbank 百分位参考，
用于回答"这个分子在已上市药物分布中处于什么位置"。
"""

import warnings
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

# 模型端点 -> 输出列名
_ENDPOINT_MAP = {
    "BBB_Martins": "BBB_prob",
    "Pgp_Broccatelli": "Pgp_prob",
    "hERG": "hERG_prob",
    "DILI": "DILI_prob",
    "AMES": "AMES_prob",
    "QED": "QED",
    "Caco2_Wang": "Caco2",
    "Solubility_AqSolDB": "Solubility",
    "LogP_AstraZeneca": "LogP_AZ",
}

ADMET_COLUMNS = list(_ENDPOINT_MAP.values()) + [
    "BBB permeant",
    "Pgp substrate",
    "hERG risk",
    "DILI risk",
]

_MODEL = None
_LOAD_ERROR = None


def _get_model():
    """惰性加载模型（首次调用会加载 5 个模型集成，约 10 秒）。"""
    global _MODEL, _LOAD_ERROR
    if _MODEL is not None or _LOAD_ERROR is not None:
        return _MODEL
    try:
        from admet_ai import ADMETModel
        _MODEL = ADMETModel()
    except Exception as e:  # pragma: no cover
        _LOAD_ERROR = str(e)
        _MODEL = None
    return _MODEL


def admet_available() -> bool:
    return _get_model() is not None


def compute_admet(smiles_list: List[str]) -> List[Dict]:
    """
    批量计算 ADMET 性质。

    Args:
        smiles_list: SMILES 列表

    Returns:
        与输入等长的 list[dict]。失败时该位置全为 NaN。
    """
    results: List[Dict] = []
    if not smiles_list:
        return results

    model = _get_model()
    if model is None:
        nan_row = {c: float("nan") for c in ADMET_COLUMNS}
        return [dict(nan_row) for _ in smiles_list]

    try:
        df = model.predict(smiles_list)
    except Exception:
        nan_row = {c: float("nan") for c in ADMET_COLUMNS}
        return [dict(nan_row) for _ in smiles_list]

    for idx in range(len(smiles_list)):
        row = {c: float("nan") for c in ADMET_COLUMNS}
        try:
            if idx >= len(df):
                results.append(row)
                continue
            rec = df.iloc[idx]
            for ep, col in _ENDPOINT_MAP.items():
                if ep in df.columns:
                    val = rec[ep]
                    row[col] = float(val) if val == val else float("nan")

            # 二值化标签（与原始对比实验 CSV 的 Yes/No 口径一致）
            row["BBB permeant"] = "Yes" if row.get("BBB_prob", float("nan")) > 0.5 else "No"
            row["Pgp substrate"] = "Yes" if row.get("Pgp_prob", float("nan")) > 0.5 else "No"
            row["hERG risk"] = "Yes" if row.get("hERG_prob", float("nan")) > 0.5 else "No"
            row["DILI risk"] = "Yes" if row.get("DILI_prob", float("nan")) > 0.5 else "No"
        except Exception:
            pass
        results.append(row)

    return results
