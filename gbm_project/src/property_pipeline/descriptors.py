"""
RDKit 描述符 / 结构警示 / 合成可及性
=====================================

口径说明（写入论文 Method 时直接引用）:
    MW               = RDKit Descriptors.MolWt (平均分子量)
    TPSA             = RDKit Descriptors.TPSA (Ertl 拓扑极性表面积)
    LogP             = RDKit Crippen.MolLogP (Wildman-Crippen)
    HBD/HBA          = RDKit Lipinski.NumHDonors / NumHAcceptors
    RotB             = RDKit Lipinski.NumRotatableBonds
    PAINS alerts     = RDKit FilterCatalog, PAINS catalog (A/B/C)
    Brenk alerts     = RDKit FilterCatalog, BRENK catalog
    NIH alerts       = RDKit FilterCatalog, NIH catalog
    SA score         = Ertl & Schuffenhauer RDKit Contrib SA_Score
                       (1 = 极难合成, 10 = 极易合成)

注意: SA score 依赖 fpscores.pkl.gz 片段评分库。本模块优先使用 RDKit Contrib 目录下的
      官方实现；若缺失则回退到 gbm_project/src/sascorer.py 的启发式实现。
"""

import os
import sys
import warnings
from typing import Dict, List, Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski
from rdkit.Chem import FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams

RDLogger.DisableLog("rdApp")
warnings.filterwarnings("ignore")

# ---- 加载 Ertl SA_Score（优先官方 Contrib 版本）----
_SA_SCORER = None
_SA_SOURCE = None


def _load_sascorer():
    global _SA_SCORER, _SA_SOURCE
    if _SA_SCORER is not None:
        return _SA_SCORER

    # 1) RDKit Contrib 官方版本（含 fpscores.pkl.gz）
    try:
        from rdkit.Chem import RDConfig
        contrib = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if os.path.isdir(contrib) and os.path.isfile(os.path.join(contrib, "fpscores.pkl.gz")):
            if contrib not in sys.path:
                sys.path.insert(0, contrib)
            import sascorer as _s
            _SA_SCORER = _s.calculateScore
            _SA_SOURCE = "rdkit_contrib"
            return _SA_SCORER
    except Exception:
        pass

    # 2) 项目内启发式实现
    try:
        gbm_src = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if gbm_src not in sys.path:
            sys.path.insert(0, gbm_src)
        import sascorer as _s
        _SA_SCORER = _s.calculateScore
        _SA_SOURCE = "project_heuristic"
        return _SA_SCORER
    except Exception:
        pass

    _SA_SCORER = lambda mol: float("nan")
    _SA_SOURCE = "unavailable"
    return _SA_SCORER


def sa_source() -> str:
    """返回 SA score 的实际来源，供论文 Method 脚注。"""
    _load_sascorer()
    return _SA_SOURCE


# ---- FilterCatalog 初始化（模块级缓存，只构建一次）----
_CATALOGS: Dict[str, object] = {}


def _get_catalog(kind: str):
    if kind in _CATALOGS:
        return _CATALOGS[kind]
    try:
        params = FilterCatalogParams()
        params.AddCatalog(getattr(FilterCatalogParams.FilterCatalogs, kind))
        catalog = FilterCatalog.FilterCatalog(params)
        _CATALOGS[kind] = catalog
        return catalog
    except Exception:
        return None


DESCRIPTOR_COLUMNS = [
    "MW",
    "TPSA",
    "LogP",
    "HBD",
    "HBA",
    "RotB",
    "PAINS #alerts",
    "Brenk #alerts",
    "NIH #alerts",
    "Synthetic Accessibility",
]


def compute_descriptors(smiles: str) -> Dict:
    """
    计算单分子的 RDKit 描述符与结构警示。

    Returns:
        含 DESCRIPTOR_COLUMNS 中全部键的 dict；解析失败时全部为 NaN。
    """
    out = {c: float("nan") for c in DESCRIPTOR_COLUMNS}
    if not smiles:
        return out

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return out

    try:
        out["MW"] = round(Descriptors.MolWt(mol), 3)
        out["TPSA"] = round(Descriptors.TPSA(mol), 3)
        out["LogP"] = round(Crippen.MolLogP(mol), 4)
        out["HBD"] = Lipinski.NumHDonors(mol)
        out["HBA"] = Lipinski.NumHAcceptors(mol)
        out["RotB"] = Lipinski.NumRotatableBonds(mol)
    except Exception:
        pass

    # 结构警示
    for col, kind in (
        ("PAINS #alerts", "PAINS"),
        ("Brenk #alerts", "BRENK"),
        ("NIH #alerts", "NIH"),
    ):
        catalog = _get_catalog(kind)
        if catalog is None:
            continue
        try:
            out[col] = len(catalog.GetMatches(mol))
        except Exception:
            out[col] = float("nan")

    # 合成可及性
    try:
        calc = _load_sascorer()
        val = calc(mol)
        out["Synthetic Accessibility"] = round(float(val), 3)
    except Exception:
        pass

    return out


def compute_descriptors_batch(smiles_list: List[str]) -> List[Dict]:
    """批量计算描述符。"""
    return [compute_descriptors(s) for s in smiles_list]
