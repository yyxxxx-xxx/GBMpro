"""
SMILES 修复工具
===============

模型输出的分子主体往往是对的，但尾部会拖出噪声（例如
    CC(=O)Nc12ccc3ccccc3c2c1OCc1ccccc11.1H,0.1H,0.2H-1,1,2
其中 "CC(=O)Nc12ccc3ccccc3c2c1OCc1ccccc11" 是完全合法的分子，
后面的 ".1H,0.1H,..." 是生成噪声）。

修复策略（按优先级）:
    1. 整体解析 —— 能过就原样返回（最干净）
    2. 截断到最后一个 '.' 之前（多片段 SMILES 的常见情况）
    3. 最长有效前缀 —— 从右往左逐字符截断，取第一个能通过 RDKit 完整
       sanitize 的最长子串
    4. 失败则丢弃

第 3 步会改变分子，因此必须记录 repaired=True，并在论文中说明
"generated SMILES were repaired by longest-valid-prefix truncation"。
"""

from typing import List, Optional, Tuple

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp")


def _parses(smiles: str) -> bool:
    if not smiles:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        n = mol.GetNumAtoms()
        return 3 <= n <= 150
    except Exception:
        return False


def canonical(smiles: str) -> Optional[str]:
    """返回规范化 SMILES；无效返回 None。"""
    if not _parses(smiles):
        return None
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except Exception:
        return None


def _cut_points(smiles: str):
    """
    找出所有"SMILES 状态完整"的前缀位置。

    SMILES 字符串在某个位置可以安全截断，当且仅当:
        - 所有环闭合数字都已配对 (ring bond numbers balanced)
        - 所有括号 [] () 都已闭合
        - 不在转义字符 \ 之后
        - 不在 '%NN' 双字符环号中间
    """
    points = []
    ring_stack = []   # 未配对的环号
    bracket_depth = 0
    paren_depth = 0
    i = 0
    n = len(smiles)
    while i < n:
        ch = smiles[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "%":
            # %NN 是双字符环号
            i += 3
            continue
        if ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1
        elif ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth -= 1
        elif ch.isdigit():
            if ring_stack and ring_stack[-1] == ch:
                ring_stack.pop()
            else:
                ring_stack.append(ch)
        if bracket_depth == 0 and paren_depth == 0 and not ring_stack:
            points.append(i + 1)
        i += 1
    return points


def repair_smiles(smiles: str) -> Tuple[Optional[str], bool]:
    """
    修复单个 SMILES。

    Returns:
        (修复后的 SMILES 或 None, 是否发生了修复)
    """
    if not smiles:
        return None, False

    smiles = smiles.strip()

    # 1. 直接可用
    if _parses(smiles):
        return smiles, False

    # 2. 状态感知截断: 在所有"状态完整"的位置尝试，取最长的可解析前缀
    for end in reversed(_cut_points(smiles)):
        cand = smiles[:end].strip()
        if _parses(cand):
            return cand, True

    # 3. 兜底: 截断到最后一个 '.'（多片段情况）
    if "." in smiles:
        head = smiles.rsplit(".", 1)[0].strip()
        if _parses(head):
            return head, True

    return None, False


def repair_batch(smiles_list: List[str]) -> Tuple[List[str], int]:
    """
    批量修复。

    Returns:
        (修复后的 SMILES 列表, 发生修复的数量)
    """
    out, n_repaired = [], 0
    for s in smiles_list:
        fixed, changed = repair_smiles(s)
        if fixed:
            out.append(fixed)
            if changed:
                n_repaired += 1
    return out, n_repaired
