"""
骨架 / 新颖性 / 多样性
======================

补齐对比实验中"生成分子是不是新东西、覆盖够不够广"这两个维度。

    Scaffold            = Bemis-Murcko 骨架 (通用)
    ScaffoldEntropy     = 骨架分布熵 (归一化, 越高越分散)
    FragmentEntropy     = 片段分布熵
    MaxTanimoto_ChEMBL  = 与 ChEMBL 已知分子的最大 Tanimoto 相似度 (越小越新)
    MeanTanimoto_ChEMBL = 与 ChEMBL 已知分子的平均 Tanimoto 相似度
    NovelScaffold       = 是否出现 ChEMBL 中不存在的 Murcko 骨架 (0/1)

新颖性维度需要 ChEMBL 参考集；无参考集时相关列返回 NaN，
骨架与熵指标始终可用。
"""

import math
import os
import pickle
import warnings
from typing import Dict, List, Optional, Sequence

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp")
warnings.filterwarnings("ignore")

SCAFFOLD_COLUMNS = [
    "Scaffold",
    "ScaffoldEntropy",
    "FragmentEntropy",
    "MaxTanimoto_ChEMBL",
    "MeanTanimoto_ChEMBL",
    "NovelScaffold",
]

_MORGAN_GEN = None


def _morgan_generator():
    global _MORGAN_GEN
    if _MORGAN_GEN is None:
        _MORGAN_GEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    return _MORGAN_GEN


def murcko_scaffold(smiles: str) -> Optional[str]:
    """返回 Bemis-Murcko 骨架 SMILES；失败返回 None。"""
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            # 无环骨架（如链状分子）时退回分子本身
            return Chem.MolToSmiles(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None


def _entropy(counter: Dict[str, int]) -> float:
    """
    归一化 Shannon 熵。

    H = -sum(p * log p) / log(N)
    N 为总样本数。值域 [0,1]，1 表示完全均匀分布。
    """
    n = sum(counter.values())
    if n <= 1 or len(counter) <= 1:
        return 0.0
    h = 0.0
    for count in counter.values():
        p = count / n
        h -= p * math.log(p)
    return round(h / math.log(n), 4)


def compute_scaffold_metrics(
    smiles_list: Sequence[str],
    reference_fps: Optional[List] = None,
    reference_scaffolds: Optional[set] = None,
) -> List[Dict]:
    """
    计算骨架/新颖性/多样性指标。

    Args:
        smiles_list: 待评估的 SMILES 列表
        reference_fps: ChEMBL 参考集的 Morgan FP 列表（可选，用于新颖性）
        reference_scaffolds: ChEMBL 参考集的 Murcko 骨架集合（可选）

    Returns:
        与输入等长的 list[dict]。骨架熵等集合级指标只在最后一行
        (index == len-1) 填充，其余为 NaN。
    """
    n = len(smiles_list)
    rows = []

    scaffold_counter: Dict[str, int] = {}
    fragment_counter: Dict[str, int] = {}
    parsed_mols = []

    for smi in smiles_list:
        row = {c: float("nan") for c in SCAFFOLD_COLUMNS}
        row["NovelScaffold"] = float("nan")
        mol = Chem.MolFromSmiles(smi) if smi else None
        parsed_mols.append(mol)

        if mol is None:
            rows.append(row)
            continue

        scaf = murcko_scaffold(smi)
        if scaf:
            row["Scaffold"] = scaf
            scaffold_counter[scaf] = scaffold_counter.get(scaf, 0) + 1
            if reference_scaffolds is not None:
                row["NovelScaffold"] = 0.0 if scaf in reference_scaffolds else 1.0

        # 片段统计
        try:
            frags = rdMolDescriptors.GetMorganFingerprint(mol).GetNonzeroElements()
            for key in frags:
                fragment_counter[key] = fragment_counter.get(key, 0) + 1
        except Exception:
            pass

        # 新颖性
        if reference_fps:
            try:
                fp = _morgan_generator().GetFingerprint(mol)
                sims = DataStructs.BulkTanimotoSimilarity(fp, reference_fps)
                if sims:
                    row["MaxTanimoto_ChEMBL"] = round(max(sims), 4)
                    row["MeanTanimoto_ChEMBL"] = round(sum(sims) / len(sims), 4)
            except Exception:
                pass

        rows.append(row)

    # 集合级指标
    if n:
        rows[-1]["ScaffoldEntropy"] = _entropy(scaffold_counter)
        rows[-1]["FragmentEntropy"] = _entropy(fragment_counter)
        rows[-1]["_n_unique_scaffolds"] = len(scaffold_counter)
        rows[-1]["_n_scaffolds"] = n

    return rows


# ---------------- ChEMBL 参考集缓存 ----------------

def load_reference_set(path: str) -> Dict:
    """
    加载 ChEMBL 参考集缓存（由 fetch_reference_set.py 生成）。

    Returns:
        {"fps": [...], "scaffolds": set(...)}
    """
    if not path or not os.path.isfile(path):
        return {"fps": [], "scaffolds": set()}
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return {
        "fps": payload.get("fps", []),
        "scaffolds": set(payload.get("scaffolds", [])),
    }
