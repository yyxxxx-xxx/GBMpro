"""
GBM External Physical Evaluator
===============================
A "外科手术式" replacement for LLM-based molecule evaluation (the "toxic tumor").

Design principles (参考 SyntheMol + chemcrow + REINVENT4):
- SyntheMol pattern: SMILES saved to file -> subprocess.run() -> parse float score
- chemcrow pattern: _run() receives SMILES -> returns natural-language string feedback
- REINVENT4 pattern: 每个属性有独立打分组件 -> 聚合函数(加权几何平均) -> MPO Reward

ABSOLUTE PRINCIPLE: Do NOT touch any LLM generation or knowledge-base logic.
This module is purely computational, callable from any evaluation point.
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Literal
from enum import Enum

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, AllChem

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1: 数据结构 — 物理评估结果
# =============================================================================

class EvaluationVerdict(Enum):
    """ToT 树节点的评估结论，对应原来的 sure/likely/impossible"""
    SURE = "sure"        # 物理指标全部合格，强烈推荐继续
    LIKELY = "likely"    # 有轻微问题但值得探索
    IMPOSSIBLE = "impossible"  # 有硬截断问题，该分支应被剪枝


@dataclass
class PhysicalEvaluationResult:
    """
    单个分子的完整物理评估结果。
    包含原始数值、归一化分数、Reward、以及自然语言反馈。
    """
    smiles: str

    # ---- 原始物理量（从外部计算脚本获取）----
    vina_score: Optional[float] = None       # QVina/Wina 对接分数 (kcal/mol)
    dili_prob: Optional[float] = None        # DILI 风险概率 [0, 1]  (admet-ai DILI, AUROC 0.881)
    herg_prob: Optional[float] = None        # hERG 阻断风险概率 [0, 1] (admet-ai hERG, AUROC 0.839)
    bbb_score: Optional[float] = None        # BBB 评分 [0, 1]        (admet-ai BBB_Martins, AUROC 0.900)
    pgp_prob: Optional[float] = None        # P-gp 外排底物概率 [0, 1] (admet-ai Pgp_Broccatelli, AUROC 0.947)
    ames_prob: Optional[float] = None        # AMES 致突变概率 [0, 1]  (admet-ai AMES, AUROC 0.882)
    qed: Optional[float] = None             # 定量类药性 [0, 1]        (admet-ai QED)

    # ---- RDKit 描述符（本地计算）----
    rd_tpsa: float = 0.0
    rd_mw: float = 0.0
    rd_logp: float = 0.0
    rd_hbd: int = 0
    rd_hba: int = 0
    rd_n_rotatable: int = 0
    rd_n_rings: int = 0

    # ---- 归一化分数 [0, 1]（越高越好）----
    vina_norm: float = 0.0
    dili_norm: float = 1.0      # 1 - dili_prob（越低毒性越好）
    herg_norm: float = 1.0      # 1 - herg_prob
    bbb_norm: float = 0.0
    pgp_norm: float = 1.0      # 1 - pgp_prob（外排风险越低越好）
    ames_norm: float = 1.0     # 1 - ames_prob
    qed_norm: float = 0.0
    tpsa_norm: float = 0.0
    mw_norm: float = 0.0

    # ---- 结构警报命中 ----
    dili_alert_matches: List[str] = field(default_factory=list)
    herg_alert_matches: List[str] = field(default_factory=list)

    # ---- 基线毒性（用于 Delta Scoring）----
    baseline_dili: Optional[float] = None
    baseline_herg: Optional[float] = None

    # ---- MPO Reward ----
    reward: float = 0.0        # 最终 Reward
    is_pruned: bool = False    # 是否被硬截断剪枝

    # ---- ToT verdict ----
    verdict: EvaluationVerdict = EvaluationVerdict.LIKELY
    prune_reason: str = ""      # 剪枝原因（如果有的话）

    # ---- 内部元数据 ----
    vina_timeout: bool = False
    vina_error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'smiles': self.smiles,
            'vina_score': self.vina_score,
            'dili_prob': self.dili_prob,
            'herg_prob': self.herg_prob,
            'bbb_score': self.bbb_score,
            'rd_tpsa': round(self.rd_tpsa, 2),
            'rd_mw': round(self.rd_mw, 2),
            'rd_logp': round(self.rd_logp, 2),
            'vina_norm': round(self.vina_norm, 4),
            'dili_norm': round(self.dili_norm, 4),
            'herg_norm': round(self.herg_norm, 4),
            'bbb_norm': round(self.bbb_norm, 4),
            'pgp_prob': self.pgp_prob,
            'ames_prob': self.ames_prob,
            'qed': self.qed,
            'reward': round(self.reward, 4),
            'is_pruned': self.is_pruned,
            'prune_reason': self.prune_reason,
            'verdict': self.verdict.value,
            'dili_alert_matches': self.dili_alert_matches,
            'herg_alert_matches': self.herg_alert_matches,
        }

    def build_feedback_text(self) -> str:
        """
        将物理评估结果组装成自然语言反馈 Prompt（注入给 LLM）。

        参考 chemcrow/tools/rdkit.py 的 _run() 返回格式：
        返回一段描述性文本，LLM 可据此继续优化。
        """
        parts = []

        # Vina 得分
        if self.vina_score is not None:
            vina_str = f"Vina得分={self.vina_score:.2f} kcal/mol"
            if self.vina_score > -7.0:
                vina_str += " (偏弱，需增强结合)"
            elif self.vina_score < -10.0:
                vina_str += " (强结合)"
            else:
                vina_str += " (中等结合)"
            parts.append(vina_str)
        elif self.vina_error:
            parts.append(f"Vina计算失败: {self.vina_error}")
        else:
            parts.append("Vina得分: N/A")

        # DILI (admet-ai DILI, AUROC 0.881)
        if self.dili_prob is not None:
            dili_str = f"肝毒性(DILI)={self.dili_prob:.2f}"
            if self.dili_prob > 0.85:
                dili_str += " (警告: 高风险)"
            elif self.dili_prob > 0.5:
                dili_str += " (注意: 中等风险)"
            else:
                dili_str += " (低风险)"
            parts.append(dili_str)

        # hERG (admet-ai hERG, AUROC 0.839)
        if self.herg_prob is not None:
            herg_str = f"心脏毒性(hERG)={self.herg_prob:.2f}"
            if self.herg_prob > 0.8:
                herg_str += " (警告: 高风险)"
            elif self.herg_prob > 0.5:
                herg_str += " (注意: 中等风险)"
            else:
                herg_str += " (低风险)"
            parts.append(herg_str)

        # BBB (admet-ai BBB_Martins, AUROC 0.900)
        if self.bbb_score is not None:
            bbb_str = f"BBB评分={self.bbb_score:.2f}"
            if self.bbb_score > 0.7:
                bbb_str += " (高CNS渗透)"
            elif self.bbb_score > 0.4:
                bbb_str += " (中等CNS渗透)"
            else:
                bbb_str += " (低CNS渗透，不适合GBM)"
            parts.append(bbb_str)

        # P-gp 外排 (admet-ai Pgp_Broccatelli, AUROC 0.947)
        # GBM 场景关键: 外排会抵消 BBB 穿透
        if self.pgp_prob is not None:
            pgp_str = f"P-gp外排底物={self.pgp_prob:.2f}"
            if self.pgp_prob > 0.7:
                pgp_str += " (警告: 高外排风险，会抵消BBB穿透)"
            elif self.pgp_prob > 0.4:
                pgp_str += " (注意: 中等外排风险)"
            else:
                pgp_str += " (低外排风险)"
            parts.append(pgp_str)

        # AMES 致突变 (admet-ai AMES, AUROC 0.882)
        if self.ames_prob is not None:
            ames_str = f"致突变(AMES)={self.ames_prob:.2f}"
            if self.ames_prob > 0.6:
                ames_str += " (警告: 高风险)"
            elif self.ames_prob > 0.3:
                ames_str += " (注意: 中等风险)"
            else:
                ames_str += " (低风险)"
            parts.append(ames_str)

        # 理化性质摘要
        parts.append(
            f"理化性质: MW={self.rd_mw:.1f}, LogP={self.rd_logp:.2f}, "
            f"TPSA={self.rd_tpsa:.1f}Å²"
        )

        # Reward
        parts.append(f"Reward={self.reward:.4f}")

        # 剪枝提示
        if self.is_pruned:
            parts.append(f"[警告: 此分支已被硬截断剪枝: {self.prune_reason}]")

        return " | ".join(parts)


# =============================================================================
# Section 2: Vina 对接工具 (参考 SyntheMol generate/scorer.py 的外部工具模式)
# =============================================================================

class VinaDockingTool:
    """
    Vina/QVina 对接工具。

    参考 SyntheMol 的外部评分器模式：
    1. 将 SMILES 写入临时 PDBQT 文件（或使用预生成 PDBQT）
    2. subprocess.run() 调用 QVina / Smina
    3. 解析 stdout 获取 float 对接分数

    使用方式:
        tool = VinaDockingTool(
            vina_executable="/path/to/qvina",
            receptor_pdbqt="/path/to/receptor.pdbqt",
            center=[x, y, z],
            size=[20, 20, 20]
        )
        score = tool.dock("CCO")  # 返回对接分数
    """

    def __init__(
        self,
        vina_executable: Optional[str] = None,
        receptor_pdbqt: Optional[str] = None,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        size: Tuple[float, float, float] = (20.0, 20.0, 20.0),
        timeout: int = 30,
        exhaustiveness: int = 8,
        n_poses: int = 5,
        cpu: int = 4,
        use_python_api: bool = True,
    ):
        """
        Args:
            vina_executable: QVina/Wina 可执行文件路径。如果为 None，尝试 Python API。
            receptor_pdbqt: 受体 PDBQT 文件路径（预准备好的）
            center: 对接盒子中心坐标 (x, y, z)
            size: 对接盒子尺寸 (width, height, depth)
            timeout: 单次对接超时（秒）
            exhaustiveness: Vina exhaustiveness 参数
            n_poses: 返回的 poses 数量
            cpu: Python API 后端的 CPU 核数
            use_python_api: 优先使用 vina Python API（推荐 —— 复用 grid maps，
                省掉每次重算 maps 的大头开销；实测 maps 计算占单次对接 80%+ 时间）。
        """
        self.vina_executable = vina_executable
        self.receptor_pdbqt = receptor_pdbqt
        self.center = tuple(float(c) for c in center)
        self.size = tuple(float(c) for c in size)
        self.timeout = timeout
        self.exhaustiveness = exhaustiveness
        self.n_poses = n_poses
        self.cpu = cpu

        # ---- Python API 后端（property_pipeline.docking.VinaDocking）----
        self._py_docker = None
        self._backend = "pseudo"
        if receptor_pdbqt and os.path.isfile(receptor_pdbqt):
            if use_python_api:
                try:
                    from property_pipeline.docking import VinaDocking
                    self._py_docker = VinaDocking(
                        receptor_pdbqt=receptor_pdbqt,
                        center=self.center,
                        size=self.size,
                        exhaustiveness=exhaustiveness,
                        n_poses=n_poses,
                        cpu=cpu,
                    )
                    self._backend = "python_api"
                    logger.info(
                        f"[VinaDockingTool] Python API backend enabled "
                        f"(exh={exhaustiveness}, n_poses={n_poses}, cpu={cpu})"
                    )
                except Exception as e:
                    logger.warning(f"[VinaDockingTool] Python API 后端不可用: {e}")
            if self._backend == "pseudo" and vina_executable and os.path.isfile(vina_executable):
                self._backend = "subprocess"
                logger.info("[VinaDockingTool] subprocess backend enabled")

        if self._backend == "pseudo":
            logger.warning(
                "[VinaDockingTool] 真实对接未启用 —— 使用基于 MW/LogP 的伪分数。 "
                "请提供 receptor_pdbqt (会自动启用 Python API 后端)。"
            )

    def _smiles_to_pdbqt(self, smiles: str, output_path: str) -> bool:
        """
        将 SMILES 转换为 PDBQT 文件。

        使用 RDKit 生成 3D 构象，然后导出为 PDBQT 格式。
        如果没有外部工具（如 obabel），使用 RDKit 内置方法。

        Returns:
            True if conversion successful, False otherwise.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False

            # 生成 3D 构象
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.UFFOptimizeMolecule(mol)

            # 写 PDBQT（需要 obabel 或手动转换）
            # 方案A: 尝试使用 obabel
            try:
                proc = subprocess.run(
                    [
                        "obabel", "-ipdb",
                        self._write_pdb(mol),
                        "-opdbqt",
                        "-O", output_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                return proc.returncode == 0
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

            # 方案B: 直接写简化 PDBQT（RDKit 不直接支持 PDBQT，用近似方法）
            # 此处为占位：实际需要调用 prepare_ligand.py 或 Meeko
            pdb_content = self._rdkit_to_pdb(mol)
            with open(output_path, 'w') as f:
                f.write(pdb_content)
            return True

        except Exception as e:
            logger.debug(f"[Vina] SMILES->PDBQT failed: {e}")
            return False

    def _write_pdb(self, mol) -> str:
        """将 RDKit mol 写入临时 PDB 文件"""
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False, mode='w') as f:
            block = Chem.MolToPDBBlock(mol)
            f.write(block)
            return f.name

    def _rdkit_to_pdb(self, mol) -> str:
        """RDKit Mol -> PDB 字符串（简化版，原子类型不全）"""
        try:
            return Chem.MolToPDBBlock(mol)
        except Exception:
            return ""

    def dock(self, smiles: str) -> Tuple[Optional[float], str]:
        """
        对接单个分子并返回分数。

        Returns:
            (vina_score, error_message)
            vina_score: kcal/mol，越负越好（通常 -12 ~ -4）
            error_message: 如果出错，返回错误描述
        """
        # ---- Python API 后端（推荐）----
        if self._backend == "python_api" and self._py_docker is not None:
            try:
                score, err = self._py_docker.dock(smiles)
                return score, err
            except Exception as e:
                logger.warning(f"[Vina] Python API 对接失败: {e}")
                return None, f"{type(e).__name__}: {e}"

        if self._backend == "pseudo":
            return self._pseudo_dock(smiles), ""

        if not self._enabled:
            return self._pseudo_dock(smiles), ""

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                ligand_path = os.path.join(tmpdir, "ligand.pdbqt")

                # Step 1: SMILES -> PDBQT
                if not self._smiles_to_pdbqt(smiles, ligand_path):
                    return None, "SMILES to PDBQT conversion failed"

                # Step 2: 构建 Vina 命令
                cmd = [
                    self.vina_executable,
                    "--receptor", self.receptor_pdbqt,
                    "--ligand", ligand_path,
                    "--center_x", str(self.center[0]),
                    "--center_y", str(self.center[1]),
                    "--center_z", str(self.center[2]),
                    "--size_x", str(self.size[0]),
                    "--size_y", str(self.size[1]),
                    "--size_z", str(self.size[2]),
                    "--exhaustiveness", str(self.exhaustiveness),
                    "--num_modes", str(self.n_poses),
                    "--out", os.path.join(tmpdir, "out.pdbqt"),
                ]

                # Step 3: subprocess.run() 调用 Vina（参考 SyntheMol scorer.py 模式）
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )

                if result.returncode != 0:
                    return None, f"Vina exit code {result.returncode}: {result.stderr[:200]}"

                # Step 4: 解析输出中的 affinity (kcal/mol)
                return self._parse_vina_output(result.stdout + result.stderr), ""

        except subprocess.TimeoutExpired:
            return None, "Vina timeout"
        except Exception as e:
            return None, str(e)

    def _parse_vina_output(self, output: str) -> Optional[float]:
        """从 Vina stdout/stderr 中解析最优结合分数"""
        import re
        # Vina 输出格式: "RESULT:  ...   -9.2   0.000  2.345\n"
        # 或 "1       -8.7  0.000  2.1 ..." 行
        for line in output.split('\n'):
            line = line.strip()
            # 找第一列是数字、第二列也是数字的模式
            m = re.search(r'^\s*\d+\s+([-+]?\d+\.\d+)', line)
            if m:
                score = float(m.group(1))
                logger.debug(f"[Vina] Parsed score: {score}")
                return score
        # 备选: 直接搜索 "-9.2" 这样的模式
        m = re.search(r'([-+]?\d+\.\d+)\s+(?:kcal|/mol)', output)
        if m:
            return float(m.group(1))
        return None

    def _pseudo_dock(self, smiles: str) -> Optional[float]:
        """
        伪对接模式：当没有真实 Vina 时使用。

        基于分子描述符估算 Vina 分数（占位实现）。
        参考：Vina ~ -0.1*MW + 0.5*LogP - 0.01*TPSA - 6.0 (经验公式，GBM靶点需调参)

        此伪函数确保整个流程在没有 Vina 的环境下也能端到端运行。
        实际使用时替换为真实对接。
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)

            # 经验公式（需要根据具体靶点/PDB 结构标定）
            pseudo_vina = (
                -0.08 * mw
                + 0.4 * logp
                - 0.008 * tpsa
                - 5.5
            )
            # 限制在合理范围
            return max(-12.0, min(-4.0, pseudo_vina))
        except Exception:
            return None


# =============================================================================
# Section 3: ADMET 计算器 (DILI + BBB 概率)
# =============================================================================

class ADMETCalculator:
    """
    ADMET 属性计算器。

    架构 (2026-11-18 重构):
        打分  -> admet-ai (52 端点, Swin-Transformer 预训练模型)
        标注  -> SMARTS 结构警示 (可解释性层, 不参与打分)

    为什么这样分工:
        原来的纯 SMARTS 启发式打分与真实风险几乎无关 ——
        实测 300 个真实药物上:
            DILI: 与 admet-ai 的分类一致率仅 40.7% (比随机还差)
            hERG: 与 admet-ai 的分类一致率仅 52.0%
        原因是 DILI 的 7 条规则全是"合成中间体级"基团 (酰氯/重氮盐/环氧烷...),
        在上市药物里命中率 0-0.3%; 而 hERG 侧漏掉了最核心的碱性胺药效团,
        却被一条语义错误的"未取代苯环"规则白扣了 60% 的药物。

        现在:
            dili_prob / herg_prob / bbb_prob  <- admet-ai (DILI / hERG / BBB_Martins)
            pgp_prob / ames_prob / qed         <- admet-ai (Pgp_Broccatelli / AMES / QED)
            *_alert_matches                    <- SMARTS (仅用于生成给 LLM 的具体建议)

    端点说明 (AUROC, 来自 TDC ADMET 基准):
        DILI             0.881
        hERG             0.839
        BBB_Martins      0.900
        Pgp_Broccatelli  0.947
        AMES             0.882

    性能:
        admet-ai 在 CPU 上批量推理 ~26 ms/分子 (GPU 上反而更慢, 因为模型太小,
        kernel launch 开销占主导)。GPU 留给 7.6B 生成模型, 两者不争用。
    """

    # DILI high-risk SMARTS patterns (genuine toxicophores only)
    # NOTE: Acrylamide/cyanoacrylamide warheads are excluded here because they are
    # intentional covalent fragments in EGFR inhibitors (e.g., Afatinib, Osimertinib).
    # They are handled separately via EGFR_WARHEAD_SMARTS for exemption logic.
    #
    # 重要 (2026-11-18 修复): 原有两条 pattern 存在结构性缺陷，数学上永不匹配:
    #   - '[cx]1([N+](=O)[O-])ccc1'  环内只有 5 个原子（苯环需 6）=> 永不命中
    #   - 'C1(=O)ccc(=O)c1'            环内只有 5 个原子，且大写 C(非芳香) 与
    #                                  小写 c(芳香) 混用在一个环里不可能成立
    # 现已改为正确的 6 元环写法，并用探针分子验证过能命中。
    DILI_ALERT_SMARTS = [
        r'C(=O)Cl',            # acyl chloride (high reactivity)
        r'N=C=S',              # isothiocyanate
        r'C1CO1',              # epoxide (unsubstituted, highly reactive)
        r'[c]1([N+](=O)[O-])ccc([N+](=O)[O-])cc1',  # dinitroaryl (6-membered ring, two nitro groups)
        r'O=C1C=CC(=O)C=C1',   # quinone (non-aromatic sp2 ring, all uppercase C)
        r'OP(=O)(O)O',        # phosphate ester (fully hydrolyzable)
        r'[N-]=[N+]=C',      # diazonium salt (valid: connects to carbon)
    ]

    # DILI alert metadata: human-readable name + recommended fix
    DILI_ALERT_META: Dict[str, Dict[str, str]] = {
        r'C(=O)Cl': {
            'name': '酰氯基团 (Acyl Chloride)',
            'fix': '用酰胺（-C(=O)NH-）或酯（-C(=O)O-）替换酰氯，以降低反应活性。',
        },
        r'N=C=S': {
            'name': '异硫氰酸酯 (Isothiocyanate)',
            'fix': '异硫氰酸酯具有强亲电性。建议使用碳酰胺（-NH-C(=O)-）替代，或完全去除该基团。',
        },
        r'C1CO1': {
            'name': '环氧烷 (Unsubstituted Epoxide)',
            'fix': '未取代的环氧烷是高风险基因毒性警示结构。建议使用内酰胺环或饱和杂环替代。',
        },
        r'[c]1([N+](=O)[O-])ccc([N+](=O)[O-])cc1': {
            'name': '二硝基芳烃 (Dinitroaryl)',
            'fix': '多硝基芳香族化合物具有强氧化性和潜在基因毒性。建议减少硝基数量或用磺酰胺替代。',
        },
        r'O=C1C=CC(=O)C=C1': {
            'name': '醌类 (Quinone)',
            'fix': '醌类代谢物可产生活性氧(ROS)，引发氧化应激损伤。建议将苯醌替换为氢醌二甲醚或饱和环己酮结构。',
        },
        r'OP(=O)(O)O': {
            'name': '磷酸酯 (Phosphate Ester)',
            'fix': '完全可水解的磷酸酯具有急性毒性风险。建议使用膦酸酯或完全去除该基团。',
        },
        r'[N-]=[N+]=C': {
            'name': '重氮盐 (Diazonium Salt)',
            'fix': '重氮盐是强亲电试剂，可与DNA发生反应。请用稳定的氨基（-NH2）替代。',
        },
    }

    # EGFR covalent inhibitor warhead SMARTS — intentionally used, exempt from DILI penalty
    # These are NOT toxicophores in the context of GBM/EGFR covalent drug design.
    EGFR_WARHEAD_SMARTS = [
        r'C=CC(=O)N',           # acrylamide (primary EGFR warhead, e.g., Afatinib)
        r'C=CC(N)=O',           # substituted acrylamide
        r'C=CC#N',              # acrylonitrile variant
        r'C=CC(=O)C=C',        # diacrylamide / bismaleimide
        r'c1ccc(NC(=O)C=C)cc1',  # aryl acrylamide (common EGFR scaffold)
        r'NC(=O)C=C',           # any acrylamide fragment
        r'N#CC=C',              # cyanoacrylamide warhead (e.g., WZ8040)
    ]

    # hERG channel blocking SMARTS patterns
    # Based on: Aronov & Goldman "Predictive models for hERG blockade",
    # Cavalli et al. SAR analysis of known hERG blockers, and medicinal chemistry literature.
    # Key structural features: lipophilic bases, aryls with heteroatoms, high logP regions.
    #
    # 修复记录 (2026-11-18, 经 300 个真实药物 + 探针分子验证):
    #   1. 碱性胺侧链: 原 '[NX3;H2,H1;!$(NC=O)]-[CX4H2]-[CX4H2]-[cR1]' 要求恰好 2 个
    #      sp3 碳间隔，命中率 0% —— 而碱性胺恰是 hERG 结合的核心药效团。
    #      放宽为 0-1 个 sp3 碳后恢复正常。
    #   2. '未取代苯环': 原 '[cR1]1[cR1][cR1][cR1][cR1][cR1]1' 只表达"6 元芳香环"，
    #      根本没表达"未取代"，对完全取代的苯环也命中（60% 假阳性）。
    #      已删除 —— LogP 项已承担疏水性惩罚，重复计罚无意义。
    #   3. 萘环: 两条写法是同一个萘，重复计 0.40 分。已去重。
    hERG_ALERT_SMARTS = [
        (r'[NX3;H2,H1;!$(NC=O)]-[CX4H2,CX4H3]-[cR1]', '脂溶性碱性胺侧链'),  # lipophilic basic amine (core hERG pharmacophore)
        (r'[cR1]1[cR1][cR1][n;+0][cR1][cR1]1', '含碱性氮的杂芳环'),  # pyridine-type basic N in aromatic ring
        (r'c1ccc2c(c1)cccc2', '萘环 (Naphthalene)'),  # naphthalene scaffold
        (r'[cR1]1([N+](=O)[O-])cccc[cR1]1', '硝基芳香族 (Nitroaryl)'),  # nitroaryl (6-membered ring)
        (r'[CX4H3]-[CX4H2]-[NX3;H2,H1]', '烷基胺侧链'),  # alkyl amine
        (r'[cR1]1[cR1][cR1]([OX2H1])[cR1][cR1][cR1]1', '苯酚/羟基芳香族'),  # phenols (lipophilic)
        (r'c1ccc(C(F)(F)F)cc1', '三氟甲基苯 (Trifluoromethylphenyl)'),  # CF3-aryl
        (r'c1ccc(Cl)cc1', '氯苯 (Chlorophenyl)'),  # chlorophenyl
        (r'c1ccc(Br)cc1', '溴苯 (Bromophenyl)'),  # bromophenyl
    ]

    # 碱性胺 SMARTS (hERG 结合的核心药效团: 疏水端 + 质子化氮)
    # 用于 _compute_herg 的 n_basic 计数。单独维护一份原文, 避免对编译后的
    # Mol 对象做字符串匹配 (那永远是 False)。
    _BASIC_AMINE_SMARTS = [
        r'[NX3;H2,H1;!$(NC=O)]-[CX4H2,CX4H3]-[cR1]',  # 苄胺 / 苯乙胺型
        r'[CX4H3]-[CX4H2]-[NX3;H2,H1]',                 # 烷基胺
        r'[NX3;H2,H1;!$(NC=O)]-[cR1]',                   # 苯胺型 (直接连芳环)
    ]

    # hERG alert metadata
    hERG_ALERT_META: Dict[str, Dict[str, str]] = {
        r'[NX3;H2,H1;!$(NC=O)]-[CX4H2,CX4H3]-[cR1]': {
            'name': '脂溶性碱性胺侧链',
            'fix': '碱性胺与疏水芳环的组合是 hERG 阻断的核心药效团（疏水端插入通道 + '
                   '质子化氮与通道极性位点结合）。建议将胺基酰胺化/磺酰胺化以降低 pKa，'
                   '或在芳环上引入极性取代基（-OH、-SO2NH2、-COOH）打断疏水性。',
        },
        r'[cR1]1[cR1][cR1][n;+0][cR1][cR1]1': {
            'name': '含碱性氮的杂芳环',
            'fix': '杂芳环上的氮原子增强了对 hERG 通道的亲和力。建议用氧原子（-O-）或碳原子（-CH2-）替换该位置的氮。',
        },
        r'c1ccc2c(c1)cccc2': {
            'name': '萘环结构',
            'fix': '萘环是经典的高 hERG 风险骨架。请将萘替换为吡啶环、哌嗪或带极性侧链的苯环以降低 LogP。',
        },
        r'[cR1]1([N+](=O)[O-])cccc[cR1]1': {
            'name': '硝基芳香族',
            'fix': '硝基芳香族同时贡献 hERG 风险和 DILI 风险。建议用磺酰胺（-SO2NH-）或酰胺（-C(=O)NH-）替换硝基。',
        },
        r'[cR1]1[cR1][cR1]([OX2H1])[cR1][cR1][cR1]1': {
            'name': '苯酚/羟基芳香族',
            'fix': '酚羟基去质子化后带负电，通常降低 hERG 风险；但若与高疏水骨架相邻仍可能结合。'
                   '建议确认整体 LogP 并考虑引入更强的极性基团。',
        },
        r'[CX4H3]-[CX4H2]-[NX3;H2,H1]': {
            'name': '烷基胺侧链',
            'fix': '烷基胺的碱性氮原子是 hERG 结合的常见药效团。建议将末端胺基替换为酰胺或磺酰胺，以降低 pKa。',
        },
        r'c1ccc(C(F)(F)F)cc1': {
            'name': '三氟甲基苯 (CF3-Phenyl)',
            'fix': 'CF3 基团的强吸电子性和高 LogP 贡献共同加剧 hERG 阻断风险。请考虑将 -CF3 替换为 -CHF2 或 -CONH2。',
        },
        r'c1ccc(Cl)cc1': {
            'name': '氯苯基团',
            'fix': '氯苯是常见的 hERG 警示结构。建议将氯原子替换为极性基团如 -OH、-NH2，或移动至分子极性端。',
        },
        r'c1ccc(Br)cc1': {
            'name': '溴苯基团',
            'fix': '溴苯与氯苯类似，是常见的 hERG 警示结构。建议将溴原子替换为极性基团如 -OH、-NH2 或 -SO2NH2。',
        },
    }

    def __init__(self):
        self._dili_patterns = []
        for smarts in self.DILI_ALERT_SMARTS:
            try:
                pat = Chem.MolFromSmarts(smarts)
                if pat:
                    self._dili_patterns.append(pat)
            except Exception:
                pass

        self._egfr_warhead_patterns = []
        for smarts in self.EGFR_WARHEAD_SMARTS:
            try:
                pat = Chem.MolFromSmarts(smarts)
                if pat:
                    self._egfr_warhead_patterns.append(pat)
            except Exception:
                pass

        # ---- hERG SMARTS patterns ----
        self._herg_patterns = []
        for smarts, _ in self.hERG_ALERT_SMARTS:
            try:
                pat = Chem.MolFromSmarts(smarts)
                if pat:
                    self._herg_patterns.append((pat, smarts))
            except Exception:
                pass

        # ---- 碱性胺 pattern (用于 n_basic 计数, 见 _compute_herg 修复说明) ----
        self._basic_amine_patterns = {}
        for smarts in self._BASIC_AMINE_SMARTS:
            try:
                pat = Chem.MolFromSmarts(smarts)
                if pat:
                    self._basic_amine_patterns[smarts] = pat
            except Exception:
                pass

        # Clark 公式参数已移除；现在使用 Clark 公式直接在 _compute_bbb 中计算

    # ---------------------------------------------------------------
    # admet-ai 后端
    # ---------------------------------------------------------------
    # 注意: property_pipeline.admet.compute_admet() 返回的 dict 用的是重命名后的
    # 列名 (DILI_prob / hERG_prob / BBB_prob / Pgp_prob / AMES_prob / QED)，
    # 不是 admet-ai 的原始端点名。
    _ADMET_ENDPOINT_MAP = {
        "DILI_prob": "dili_prob",
        "hERG_prob": "herg_prob",
        "BBB_prob": "bbb_score",
        "Pgp_prob": "pgp_prob",
        "AMES_prob": "ames_prob",
        "QED": "qed",
    }

    def __init__(self, use_admet_ai: bool = True):
        """
        Args:
            use_admet_ai: True 时用 admet-ai 打分（推荐）；
                          False 时回退到旧的 SMARTS 启发式（仅用于消融对比）。
        """
        self._dili_patterns = []
        for smarts in self.DILI_ALERT_SMARTS:
            try:
                pat = Chem.MolFromSmarts(smarts)
                if pat:
                    self._dili_patterns.append(pat)
            except Exception:
                pass

        self._egfr_warhead_patterns = []
        for smarts in self.EGFR_WARHEAD_SMARTS:
            try:
                pat = Chem.MolFromSmarts(smarts)
                if pat:
                    self._egfr_warhead_patterns.append(pat)
            except Exception:
                pass

        # ---- hERG SMARTS patterns ----
        self._herg_patterns = []
        for smarts, _ in self.hERG_ALERT_SMARTS:
            try:
                pat = Chem.MolFromSmarts(smarts)
                if pat:
                    self._herg_patterns.append((pat, smarts))
            except Exception:
                pass

        # ---- 碱性胺 pattern (用于 n_basic 计数, 见 _compute_herg 修复说明) ----
        self._basic_amine_patterns = {}
        for smarts in self._BASIC_AMINE_SMARTS:
            try:
                pat = Chem.MolFromSmarts(smarts)
                if pat:
                    self._basic_amine_patterns[smarts] = pat
            except Exception:
                pass

        # ---- admet-ai 后端 ----
        self.use_admet_ai = use_admet_ai
        self._admet_backend = None
        self._admet_cache: Dict[str, Dict[str, float]] = {}
        self._admet_batch_size = 64
        if use_admet_ai:
            self._admet_backend = self._init_admet_backend()
            if self._admet_backend is None:
                logger.warning(
                    "[ADMETCalculator] admet-ai 不可用，回退到 SMARTS 启发式打分 "
                    "(精度显著下降)。"
                )
                self.use_admet_ai = False

    @staticmethod
    def _init_admet_backend():
        """
        加载 admet-ai。

        注意: 绝不修改 os.environ['CUDA_VISIBLE_DEVICES'] —— 本模块可能与
        LLM 生成跑在同一个进程里 (generate_tot_molecules.py)，改写全局环境变量
        会让 7.6B 生成模型失去 GPU。

        admet-ai 的端点模型很小，显存占用约 0.05 GB，与生成模型不冲突；
        实测 GPU 上 85 ms/分子、CPU 上 26 ms/分子，两者都在可接受范围。
        """
        try:
            from property_pipeline.admet import compute_admet, admet_available
            if not admet_available():
                return None
            # 首次调用会加载 5 个模型集成（约 4 s），这里预热一次
            compute_admet(["CCO"])
            return compute_admet
        except Exception as e:
            logger.warning(f"[ADMETCalculator] admet-ai 初始化失败: {e}")
            return None

    def compute_batch(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """
        批量计算 ADMET —— 这是主要入口。

        admet-ai 批量推理在 CPU 上约 26 ms/分子，而逐个调用是 4336 ms/分子
        （batch_size 退化成 1，损失 12×）。所以这里必须走批量。
        """
        if not smiles_list:
            return []

        results: List[Dict[str, Any]] = [None] * len(smiles_list)  # type: ignore

        # ---- 1) admet-ai 打分（含缓存）----
        if self.use_admet_ai and self._admet_backend is not None:
            todo_idx = []
            todo_smis = []
            for i, smi in enumerate(smiles_list):
                if smi in self._admet_cache:
                    results[i] = dict(self._admet_cache[smi])
                else:
                    todo_idx.append(i)
                    todo_smis.append(smi)

            if todo_smis:
                for start in range(0, len(todo_smis), self._admet_batch_size):
                    chunk = todo_smis[start:start + self._admet_batch_size]
                    try:
                        rows = self._admet_backend(chunk)
                    except Exception as e:
                        logger.warning(f"[ADMETCalculator] admet-ai 批量推理失败: {e}")
                        rows = [{}] * len(chunk)
                    for j, row in enumerate(rows):
                        vals = {}
                        for ep, key in self._ADMET_ENDPOINT_MAP.items():
                            v = row.get(ep, float("nan"))
                            try:
                                vals[key] = float(v)
                            except (TypeError, ValueError):
                                vals[key] = float("nan")
                        self._admet_cache[chunk[j]] = vals
                        results[todo_idx[start + j]] = dict(vals)

        # ---- 2) SMARTS 标注层（可解释性，不参与打分）----
        for i, smi in enumerate(smiles_list):
            res = results[i] or {}
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if mol is None:
                res.setdefault("dili_prob", 0.5)
                res.setdefault("herg_prob", 0.5)
                res.setdefault("bbb_score", 0.3)
                res.setdefault("pgp_prob", 0.5)
                res.setdefault("ames_prob", 0.5)
                res.setdefault("qed", 0.5)
                res["dili_alert_matches"] = []
                res["herg_alert_matches"] = []
                results[i] = res
                continue

            try:
                Chem.SanitizeMol(mol, catchErrors=True)
            except Exception:
                pass

            dili_alerts = self._match_dili_alerts(mol)
            herg_alerts = self._match_herg_alerts(mol)

            if not self.use_admet_ai:
                # 回退路径：旧的启发式打分
                dili_result = self._compute_dili(mol)
                herg_result = self._compute_herg(mol)
                bbb_score = self._compute_bbb(mol)
                res["dili_prob"] = float(dili_result["score"])
                res["herg_prob"] = float(herg_result["score"])
                res["bbb_score"] = float(bbb_score)
                res["pgp_prob"] = float("nan")
                res["ames_prob"] = float("nan")
                res["qed"] = float("nan")
            else:
                # admet-ai 打分；若某端点缺失(NaN) 则回退到启发式
                for key, fallback in (
                    ("dili_prob", None),
                    ("herg_prob", None),
                    ("bbb_score", None),
                ):
                    v = res.get(key, float("nan"))
                    if v != v:  # NaN
                        if key == "dili_prob":
                            res[key] = float(self._compute_dili(mol)["score"])
                        elif key == "herg_prob":
                            res[key] = float(self._compute_herg(mol)["score"])
                        else:
                            res[key] = float(self._compute_bbb(mol))

            res["dili_alert_matches"] = dili_alerts
            res["herg_alert_matches"] = herg_alerts
            results[i] = res

        return results

    def compute(self, smiles: str) -> Dict[str, Any]:
        """
        计算 DILI_prob, hERG_prob, BBB_prob (+ Pgp/AMES/QED)。

        Returns:
            {'dili_prob': float [0,1],
             'herg_prob': float [0,1],
             'bbb_score': float [0,1],
             'pgp_prob': float [0,1],
             'ames_prob': float [0,1],
             'qed': float [0,1],
             'dili_alert_matches': List[str],
             'herg_alert_matches': List[str]}
        """
        return self.compute_batch([smiles])[0]

    def _match_dili_alerts(self, mol) -> List[str]:
        """SMARTS 标注层：命中的 DILI 毒性基团名称（去重）。"""
        names = []
        for pat, smarts in zip(self._dili_patterns, self.DILI_ALERT_SMARTS):
            if mol.HasSubstructMatch(pat):
                names.append(self.DILI_ALERT_META.get(smarts, {}).get("name", smarts))
        return list(set(names))

    def _match_herg_alerts(self, mol) -> List[str]:
        """SMARTS 标注层：命中的 hERG 警示结构名称（去重）。"""
        names = []
        for pat, smarts in self._herg_patterns:
            if mol.HasSubstructMatch(pat):
                names.append(self.hERG_ALERT_META.get(smarts, {}).get("name", smarts))
        return list(set(names))

    def _compute_herg(self, mol) -> Dict[str, Any]:
        """
        计算 hERG 阻断风险概率 [0, 1]，并返回命中的警报名称。

        Returns:
            {'score': float, 'alert_names': List[str]}
        """
        score = 0.0
        alert_names: List[str] = []

        # 1. SMARTS 警报匹配
        for pat, smarts in self._herg_patterns:
            if mol.HasSubstructMatch(pat):
                # 找到对应的元数据名称
                meta_name = None
                for key, meta in self.hERG_ALERT_META.items():
                    if key == smarts:
                        meta_name = meta['name']
                        break
                alert_names.append(meta_name or smarts)
                score += 0.20

        # 2. EGFR warhead exemption: acrylamides can still be hERG risks
        # (they are intentionally included, but we reduce the penalty)
        has_egfr_warhead = any(
            mol.HasSubstructMatch(p) for p in self._egfr_warhead_patterns
        )
        if has_egfr_warhead and alert_names:
            # EGFR warheads partially mitigate hERG risk from other moieties
            score *= 0.75

        # 3. High LogP: lipophilicity is a major hERG risk factor
        try:
            logp = Crippen.MolLogP(mol)
            if logp > 5.5:
                score += 0.20
            elif logp > 4.5:
                score += 0.10
        except Exception:
            pass

        # 4. MW penalty: larger molecules with basic nitrogens are higher risk
        try:
            mw = Descriptors.ExactMolWt(mol)
            if mw > 600:
                score += 0.10
        except Exception:
            pass

        # 5. Basic nitrogen count: hERG binders are typically bases with pKa > 7.5
        #    修复 (2026-11-18): 原实现写作 `'NX3' in str(pat)`，但 pat 是 RDKit 编译后的
        #    Mol 对象，其 str() 形如 '<rdkit.Chem.rdchem.Mol object at 0x...>'，
        #    永远不含 'NX3' => n_basic 恒为 0，该分支是死代码。
        #    现改为直接遍历 SMARTS 原文。
        n_basic = sum(
            1 for smarts in self._BASIC_AMINE_SMARTS
            if mol.HasSubstructMatch(self._basic_amine_patterns[smarts])
        )
        if n_basic >= 2:
            score += 0.15

        return {
            'score': min(1.0, max(0.0, score)),
            'alert_names': alert_names,
        }

    def _compute_dili(self, mol) -> Dict[str, Any]:
        """
        计算 DILI 风险概率 [0, 1]，并返回命中的警报名称。

        Returns:
            {'score': float, 'alert_names': List[str]}

        Design principles for GBM/EGFR covalent inhibitor context:
        1. Genuine toxicophores (acyl chloride, isothiocyanate, quinone, etc.) → penalty
        2. EGFR covalent warheads (acrylamide, cyanoacrylamide) → exemption (reduce/ignore penalty)
           These are intentional pharmacophores in approved EGFR inhibitors.
        3. MW / LogP / PAINS → standard penalty
        """
        score = 0.0
        alert_names: List[str] = []

        # --- 1. Check for EGFR covalent warheads (exemption) ---
        has_egfr_warhead = False
        for pat in self._egfr_warhead_patterns:
            if mol.HasSubstructMatch(pat):
                has_egfr_warhead = True
                break

        # --- 2. Genuine toxicophore detection (skip EGFR warheads) ---
        for pat, smarts in zip(self._dili_patterns, self.DILI_ALERT_SMARTS):
            if mol.HasSubstructMatch(pat):
                meta = self.DILI_ALERT_META.get(smarts, {})
                alert_names.append(meta.get('name', smarts))
                score += 0.25
        if len(alert_names) >= 2:
            score += 0.25  # multiple genuine toxicophores = high risk
        alert_names = list(set(alert_names))  # deduplicate

        # --- 3. EGFR warhead exemption: reduce penalty by 75% ---
        if has_egfr_warhead and alert_names:
            score *= 0.25

        # --- 4. MW risk (tightened upper bound for GBM CNS drugs) ---
        try:
            mw = Descriptors.ExactMolWt(mol)
            if mw > 800:
                score += 0.15
            elif mw > 700:
                score += 0.05
        except Exception:
            pass

        # --- 5. LogP risk ---
        try:
            logp = Crippen.MolLogP(mol)
            if logp > 5.5:
                score += 0.10
            elif logp > 4.5:
                score += 0.05
        except Exception:
            pass

        # --- 6. PAINS simplified detection ---
        pains_like = [
            r'c1cc(-n2nc(-c3ccccc3)nc2O)on1',
        ]
        for smarts in pains_like:
            try:
                pat = Chem.MolFromSmarts(smarts)
                if pat and mol.HasSubstructMatch(pat):
                    score += 0.20
                    break
            except Exception:
                pass

        return {
            'score': min(1.0, max(0.0, score)),
            'alert_names': alert_names,
        }

    def _compute_bbb(self, mol) -> float:
        """
        计算 BBB 评分 [0, 1]。

        第一步：Clark 公式（Clark & Delany, 2000）用于 logBB 粗估。
            logBB_est = 0.152 * LogP - 0.0148 * TPSA + 0.139
            仅使用脂水分配系数和拓扑极性表面积。

        第二步：启发式惩罚项（不属于 Clark 原始公式）。
            MW > 450：     -0.20（过大分子不利于 BBB 穿透）
            HBD > 2：      -0.15（过多氢键供体不利于 BBB 穿透）
            HBA > 6：      -0.10（过多氢键受体不利于 BBB 穿透）

        最终 bbb_score = logBB_est + penalty，并 clamp 到 [0, 1]。

        概率映射：logBB > 0.5 -> 1.0, logBB < -1.0 -> 0.0
        """
        try:
            logp  = Crippen.MolLogP(mol)
            tpsa  = Descriptors.TPSA(mol)
            mw    = Descriptors.ExactMolWt(mol)
            hbd   = Descriptors.NumHDonors(mol)
            hba   = Descriptors.NumHAcceptors(mol)
        except Exception:
            return 0.3

        # --- Clark 公式：logBB 粗估（仅 LogP + TPSA）---
        logBB_est = 0.152 * logp - 0.0148 * tpsa + 0.139

        # --- 启发式惩罚项（MW / HBD / HBA，不属于 Clark 原始公式）---
        penalty = 0.0
        if mw > 450:
            penalty -= 0.20
        if hbd > 2:
            penalty -= 0.15
        if hba > 6:
            penalty -= 0.10

        bbb_score = logBB_est + penalty

        # 映射到 [0, 1]: logBB > 0.5 = 1.0, logBB < -1.0 = 0.0
        return max(0.0, min(1.0, (bbb_score + 1.0) / 1.5))


# =============================================================================
# Section 4: 多目标合意度评分器 (MPO Reward Function)
# =============================================================================

class MPORewardCalculator:
    """
    多目标合意度评分器 (Multi-Parameter Optimization Reward)。

    参考 REINVENT4 scoring/aggregators/means.py 的加权几何平均模式：

    Reward = (Vina_norm ** w1) * (DILI_norm ** w2) * (hERG_norm ** w3) * (BBB_norm ** w4) * (TPSA_norm ** w5)

    其中 DILI_norm = 1 - DILI_prob, hERG_norm = 1 - hERG_prob（越低毒性越好，取补数）

    权重设定原则（GBM 场景）：
    - w1 (Vina): 0.30 — 结合亲和力是核心
    - w2 (DILI): 0.20 — 安全性必须保障
    - w3 (hERG): 0.20 — 心脏安全性
    - w4 (BBB):  0.20 — BBB 穿透对 GBM 至关重要
    - w5 (理化): 0.10 — 类药性约束

    硬截断 (Hard Pruning) 参考 REINVENT4 scoring/transforms/steps.py:
    - Vina > -5.0 kcal/mol → 立即剪枝（过弱的结合）
    - MW < 200 或 MW > 700 → 立即剪枝
    - TPSA < 20 或 TPSA > 140 → 立即剪枝

    软惩罚 (Soft Penalty) —— 修复 (2026-11-18):
    - DILI / hERG / Pgp / AMES **不再硬剪枝**，改由 MPO reward 的连续权重体现。
      原因: 多个独立条件各取 p95 阈值，其并集按构造就会拒绝 ~19% 的真实药物
      (实测剪枝率 32.7%)。REINVENT4 的 range filter 也只用于量纲约束。
    - 仅当某端点 > 0.995（比 99.5% 的真实药物更差）才兜底剪枝。

    Delta Scoring: 如果毒性只比父节点升高 <= 0.05，或有所下降，在反馈文本中提示。
    """

    # ---- 硬截断阈值（必须满足，否则分支被剪枝）----
    # 修复 (2026-11-18): 原 dili_max=0.95 / herg_max=0.90 在旧的 SMARTS 启发式
    # 打分下实测永不触发（真实药物最大 dili=0.50, herg=0.65）。切到 admet-ai 后
    # 概率分布变为连续型，阈值按真实分布重新标定。
    #
    # 标定依据: admet-ai 在 1836 个真实上市药物上的分布 (见下表),
    # 阈值取 ~95 分位 —— 即"比 95% 的真实药物更差"才剪枝, 避免误杀。
    #
    #   端点          mean    p50     p75     p85     p90     p95    -> 阈值
    #   dili_prob    0.648  0.788  0.939  0.965  0.975  0.985  ->  0.985
    #   herg_prob    0.570  0.613  0.867  0.928  0.958  0.976  ->  0.975
    #   pgp_prob     0.461  0.481  0.778  0.874  0.924  0.972  ->  0.970
    #   ames_prob    0.269  0.196  0.393  0.516  0.613  0.699  ->  0.700
    HARD_CUTOFFS = {
        'vina_min': -5.0,
        'dili_max': 0.985,
        'herg_max': 0.975,
        'pgp_max': 0.970,
        'ames_max': 0.700,
        # MW / TPSA 范围约束 (类药性, 仿 REINVENT4 range filter)
        #
        # 标定依据 (2026-11-18): 在 1836 个真实上市药物中筛出能过 BBB 的
        # (admet-ai BBB_Martins > 0.5) 的 1247 个 —— 这才是 GBM 设计的
        # 目标人群, 阈值按这个子集的分布标定:
        #   MW:    p50=374  p90=521  p95=580  p99=639  max=813
        #   TPSA:  p50=73   p90=111  p95=121  p99=145  max=194
        # 取 p99 作为上限 => 约 1% 的真实 CNS 药物会被剪掉。
        'mw_min': 180.0,
        'mw_max': 650.0,
        'tpsa_min': 20.0,
        'tpsa_max': 145.0,   # death line: > 145 -> hard prune
        'tpsa_valley_max': 10.0,  # hard valley line: < 10 -> hard prune
        # ---- BBB 硬约束 (2026-11-19) ----
        # 动机: 反馈闭环实测发现 BBB+ 逐轮退化 (62% -> 55% -> 33%)。
        # 根因是几何平均的梯度特性: BBB 归一化分数 (0~1) 与 DILI/hERG 的
        # 补数 (1-prob) 不同量级, 模型只要把 BBB 从 0.6 降到 0.4,
        # 就能让其他端点的乘积项大幅改善, 总 reward 反而上升。
        # 对 GBM (脑瘤) 而言 BBB 穿透是**生物学前提**而非可选指标 ——
        # 打不进血脑屏障的分子对 GBM 无效。
        #
        # 因此把 BBB 提升为**硬约束**: BBB_prob < 0.5 (admet-ai BBB_Martins)
        # 的分子直接剪枝, 不进入 reward 的几何平均。
        # 这与 REINVENT4 用 `minscore` + range filter 的思路一致:
        # 先满足硬性范围, 再在可行集内优化。
        #
        # 阈值依据: 1836 个真实上市药物中 BBB+ 占 67.9% (1247 个),
        # 说明 0.5 是"能进 CNS"的公认分界, 不是人为收紧。
        'bbb_min': 0.5,
    }

    # ---- 哪些端点走硬截断，哪些只走软惩罚 ----
    # 架构修复 (2026-11-18): 原实现把 DILI/hERG/Pgp/AMES 都设成硬截断。
    # 但 6 个独立条件各自取 ~p95 阈值，其并集按构造就会拒绝 ~19% 的真实药物，
    # 实测剪枝率高达 32.7% —— 反馈闭环会把大量合格分子误杀。
    #
    # 参考 REINVENT4 的做法: range filter（硬截断）只用于"物理上不可能是药物"
    # 的量纲约束 (MW / TPSA / 结合强度)；毒性/外排等风险端点应当是**软惩罚**，
    # 通过 MPO reward 的连续权重体现差异，让 LLM 看到"哪个更好"而不是
    # "过线/不过线"。
    #
    # 现在: 硬截断 = MW / TPSA / Vina（范围约束）
    #       软惩罚 = DILI / hERG / Pgp / AMES（进 reward，另有各自的极值兜底）
    SOFT_RISK_ENDPOINTS = ('dili', 'herg', 'pgp', 'ames')

    # ---- Reward 权重（归一化后相加为 1.0）----
    # 修复 (2026-11-18): 新增 Pgp / AMES / QED，并把 Vina 权重从 0.30 降到 0.25
    # (BBB 与 Pgp 对 GBM/CNS 是决定性端点，权重不应低于 Vina)。
    REWARD_WEIGHTS = {
        'vina': 0.25,
        'dili': 0.12,
        'herg': 0.10,
        'bbb': 0.20,
        'pgp': 0.15,
        'ames': 0.08,
        'qed': 0.05,
        'physchem': 0.05,
    }

    # ---- 端点下限 ----
    # 修复 (2026-11-18): 原用 max(s_i, 1e-8)，ln(1e-8) = -18.4 导致任何端点
    # 归零时 reward 直接塌到 ~0.02（断崖效应，无法排序）。
    # 这里给每个端点一个下限，使 reward 保持单调可排序。
    ENDPOINT_FLOOR = 0.05

    def __init__(self, custom_weights: Optional[Dict[str, float]] = None,
                 custom_cutoffs: Optional[Dict[str, float]] = None,
                 aggregation: str = "geometric"):
        """
        Args:
            custom_weights: 自定义权重
            custom_cutoffs: 自定义硬截断阈值
            aggregation: 'geometric'(默认, REINVENT4 风格) / 'harmonic' / 'arithmetic'
        """
        self.aggregation = aggregation
        if custom_weights:
            self.weights = custom_weights
        else:
            self.weights = dict(self.REWARD_WEIGHTS)

        if custom_cutoffs:
            self.cutoffs = {**self.HARD_CUTOFFS, **custom_cutoffs}
        else:
            self.cutoffs = dict(self.HARD_CUTOFFS)

    def check_hard_cutoffs(
        self,
        result: PhysicalEvaluationResult,
        baseline_dili: Optional[float] = None,
        baseline_herg: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        检查硬截断条件，支持 Delta Scoring。

        Delta Scoring: 如果毒性比父节点升高不超过 0.05，或者有所下降，
        即使略微超过硬截断阈值也不触发剪枝，保留探索机会。

        Returns:
            (is_pruned: bool, reason: str)
        """
        cutoffs = self.cutoffs

        # Vina 截断
        if result.vina_score is not None and result.vina_score > cutoffs['vina_min']:
            return True, f"Vina={result.vina_score:.2f} > {cutoffs['vina_min']} (过弱结合)"

        # ---- 风险端点: 软惩罚，不硬剪枝 ----
        # 修复 (2026-11-18): 这三个端点原先是硬截断。6 个独立条件各取 p95 阈值，
        # 并集按构造就会拒绝 ~19% 的真实药物 (实测剪枝率 32.7%)。
        # 现在风险只通过 MPO reward 的连续权重体现；只有极端值 (>0.995,
        # 即比 99.5% 的真实药物更差) 才兜底剪枝。
        _extreme = 0.995
        for _key, _val, _label in (
            ('pgp', result.pgp_prob, 'Pgp'),
            ('ames', result.ames_prob, 'AMES'),
            ('dili', result.dili_prob, 'DILI'),
            ('herg', result.herg_prob, 'hERG'),
        ):
            if _val is not None and _val > _extreme:
                return True, f"{_label}={_val:.3f} > {_extreme} (极端风险，已无可挽回)"

        # DILI / hERG: 已改为软惩罚（见上方"风险端点: 软惩罚"块），此处不再硬剪枝。
        # Delta Scoring 现在用在反馈文本里（generate_toxicity_feedback），
        # 提示 LLM "相比父节点毒性上升了多少"。

        # ---- BBB 硬约束 (2026-11-19) ----
        # BBB 穿透是 GBM 的生物学前提, 不是可选优化项。
        # 见 HARD_CUTOFFS['bbb_min'] 的注释: 反馈闭环实测 BBB+ 逐轮退化
        # (62% -> 33%), 因为几何平均允许模型用"牺牲一点 BBB"换取
        # 其他端点的乘积改善。硬剪枝阻断这条捷径。
        if (result.bbb_score is not None
                and result.bbb_score < cutoffs['bbb_min']):
            return True, (f"BBB={result.bbb_score:.2f} < {cutoffs['bbb_min']} "
                          f"(无法穿透血脑屏障, 对 GBM 无效)")

        # MW 截断
        if not (cutoffs['mw_min'] <= result.rd_mw <= cutoffs['mw_max']):
            return True, f"MW={result.rd_mw:.1f} outside [{cutoffs['mw_min']}, {cutoffs['mw_max']}]"

        # TPSA 截断（双侧死亡线）
        if result.rd_tpsa > cutoffs['tpsa_max']:
            return True, f"TPSA={result.rd_tpsa:.1f} > {cutoffs['tpsa_max']} Å² (肠道吸收困难)"
        if result.rd_tpsa < cutoffs['tpsa_min']:
            return True, f"TPSA={result.rd_tpsa:.1f} < {cutoffs['tpsa_min']} Å² (水溶性极差)"

        return False, ""

    def normalize_vina(self, vina_score: Optional[float]) -> float:
        """
        将 Vina 分数归一化到 [0, 1]（越高越好）。

        Vina 典型范围: -12 ~ -4 kcal/mol
        norm = ((-4) - vina) / 8
        即: -12 -> 1.0, -4 -> 0.0
        """
        if vina_score is None:
            return 0.5
        normalized = ( -4.0 - vina_score) / 8.0
        return max(0.0, min(1.0, normalized))

    def normalize_tpsa(self, tpsa: float) -> float:
        """
        TPSA 梯段式惩罚（Stepped Penalty），分为五个区段：

        1. 死亡线 (Hard Pruning): TPSA > 140 Å² → 返回 -1
           分子连肠道吸收都成问题，直接死刑（在 compute_reward 前由 check_hard_cutoffs 处理）

        2. 重度惩罚: 90 < TPSA <= 140 Å²
           无法入脑，逐级扣分（140 处降到 0.3，不再归零）
           norm = 1.0 - 0.7 * (tpsa - 90) / 50

        3. 理想区间: 40 <= TPSA <= 90 Å²
           不扣分，得满分

        4. 低谷警告: 20 <= TPSA < 40 Å²
           TPSA 过低意味着分子缺乏极性，水溶性差，在体内难以溶解
           norm = 0.8 - 0.8 * (40 - tpsa) / 20

        5. 死亡线 (Hard Pruning): TPSA < 20 Å² → 返回 -1
           由 check_hard_cutoffs 处理
        """
        if tpsa > 90.0:
            # 90 < TPSA <= 140: 逐级扣分
            # 修复 (2026-11-18): 原公式 1.0 - 2.0*(tpsa-90)/50 在 TPSA=115 处
            # norm=0.0、TPSA=120 处 norm=-0.2，reward 直接塌陷/归零 ——
            # 但硬截断宣称 140 才是死亡线，两者自相矛盾。实测 1836 个真实药物里
            # 18.0% 落在 TPSA>=115 的区域（TPSA 偏高恰恰是 CNS 药物的典型特征）。
            # 现改为 90->1.0 线性降到 140->0.3，只有 >140 才是真正的死亡线。
            if tpsa > 140.0:
                return -1.0  # 死亡线
            norm = 1.0 - 0.7 * (tpsa - 90.0) / 50.0
            return max(-1.0, norm)
        if tpsa >= 40.0:
            # 理想区间
            return 1.0
        if tpsa >= 20.0:
            # 低谷警告区：20-40 Å² 缺乏极性，水溶性差，随 TPSA 递增逐渐恢复
            # 20 → 0.5（低谷底部），40 → 1.0（进入理想区），线性过渡
            norm = 0.5 + 0.5 * (tpsa - 20.0) / 20.0
            return norm
        # TPSA < 20: 硬截断，由 check_hard_cutoffs 处理
        return -1.0

    def normalize_mw(self, mw: float) -> float:
        """MW 归一化：最优范围 300-500 Da"""
        optimal_min = 300.0
        optimal_max = 500.0
        if optimal_min <= mw <= optimal_max:
            return 1.0
        if mw < optimal_min:
            return max(0.0, mw / optimal_min)
        if mw > optimal_max:
            # 修复 (2026-11-18): 原 1.0 - (mw-500)/200 在 MW=700 处归零，
            # 而硬截断 mw_max=700 同样是边界 —— 归零造成断崖。改为 700->0.3。
            return max(0.0, 1.0 - 0.7 * (mw - optimal_max) / 200.0)

    def compute_reward(
        self,
        vina_norm: float,
        dili_norm: float,
        herg_norm: float,
        bbb_norm: float,
        tpsa_norm: float,
        mw_norm: float,
        pgp_norm: float = 1.0,
        ames_norm: float = 1.0,
        qed_norm: float = 0.5,
    ) -> float:
        """
        计算 MPO Reward（加权几何平均，参考 REINVENT4 geometric_mean）。

        Reward = exp( sum_i [ w_i * ln(max(s_i, 1e-8)) ] )
               = product_i ( s_i ** w_i )

        因子顺序: vina, dili, herg, bbb, physchem
        """
        # TPSA 死亡线防御（>140 或 <20，由 check_hard_cutoffs 提前剪枝，但保险处理）
        #
        # 修复 (2026-11-18): 原实现直接 return 0.0，导致 TPSA<20 的分子 reward
        # 恒为 0 —— 这些分子明明有合理的 Vina 分数（实测 -8.18 / -8.36）和
        # 很好的 BBB (1.00 / 0.98)，却因为一个物化维度的越界就被"抹零"。
        # 这会污染 reward-Vina 相关性分析（pruned 样本的 reward=0 变成离群点）。
        #
        # 现在: TPSA 越界交给 check_hard_cutoffs 决定剪不剪，reward 本身
        # 用一个很低的 floor 表达"这个分子不理想"，而不是完全归零。
        if tpsa_norm < 0:
            tpsa_norm = self.ENDPOINT_FLOOR * 0.5

        w = self.weights
        floor = self.ENDPOINT_FLOOR

        def _f(x: float) -> float:
            """端点下限保护 —— 避免 ln(1e-8)=-18.4 造成的断崖效应。"""
            v = max(float(x), floor)
            return min(1.0, v)

        # 低谷区（tpsa < 40）已经由 normalize_tpsa 降权，这里自然传导惩罚
        physchem = np.sqrt(_f(tpsa_norm) * _f(mw_norm))

        # 聚合方式 (2026-11-18)
        #
        # 三种方式的实测对比 (EGFR, 45 个已验证活性分子 + 连续 IC50 标签):
        #
        #   方式        rho(reward, pIC50)   稳定性(DILI ±0.1 的最坏相对影响)
        #   geometric      +0.281                14.0%   ← 采用
        #   arithmetic     +0.101                 4.6%
        #   harmonic       +0.320                70.3%
        #
        # harmonic 判别力略高但**极不稳定**: 单个端点 0.1 的抖动能引起 70% 的
        # reward 变化 (因为 w/norm 在 norm->0 时爆炸)。对 LLM 反馈循环来说,
        # 奖励信号抖动 70% 意味着 LLM 收到的是噪声而非梯度, 是灾难性的。
        #
        # geometric (REINVENT4 风格) 在判别力 (+0.281) 与稳定性 (14.0%)
        # 之间最平衡, 因此**保留原实现**。
        #
        # 注: "AUC(活性 vs 非活性)" 会因为活性分子的 ADMET 全面更差
        # (PubChem BioAssay 活性分子是药化优化产物, 为保活性牺牲 DILI/hERG)
        # 而低于 0.5 —— 这是多目标折中的正确行为, 不是缺陷。
        # 更能反映判别力的指标是 rho(reward, pIC50), 见上表。
        if self.aggregation == "harmonic":
            total = sum(
                w[e] / max(v, floor)
                for e, v in (
                    ('vina', vina_norm), ('dili', dili_norm), ('herg', herg_norm),
                    ('bbb', bbb_norm), ('pgp', pgp_norm), ('ames', ames_norm),
                    ('qed', qed_norm), ('physchem', physchem),
                )
            )
            reward = sum(self.weights.values()) / max(total, 1e-8)
        elif self.aggregation == "arithmetic":
            reward = sum(
                w[e] * max(v, 0.0)
                for e, v in (
                    ('vina', vina_norm), ('dili', dili_norm), ('herg', herg_norm),
                    ('bbb', bbb_norm), ('pgp', pgp_norm), ('ames', ames_norm),
                    ('qed', qed_norm), ('physchem', physchem),
                )
            )
        else:  # geometric (REINVENT4 风格，保留作为消融对照)
            total = (
                w['vina']     * np.log(_f(vina_norm))
                + w['dili']   * np.log(_f(dili_norm))
                + w['herg']   * np.log(_f(herg_norm))
                + w['bbb']    * np.log(_f(bbb_norm))
                + w['pgp']    * np.log(_f(pgp_norm))
                + w['ames']   * np.log(_f(ames_norm))
                + w['qed']    * np.log(_f(qed_norm))
                + w['physchem'] * np.log(max(physchem, floor))
            )
            reward = np.exp(total)
        return float(max(0.0, min(1.0, reward)))

    def decide_verdict(self, reward: float, is_pruned: bool) -> EvaluationVerdict:
        """
        根据 Reward 决定 ToT verdict。

        注意：即使 is_pruned=True 也不标记为 IMPOSSIBLE，
        而是将 reward 压低让其自然沉底。目的是保留 Feedback 给 LLM 修正机会，
        而不是直接剪枝导致死胡同。
        - Reward >= 0.7 -> sure
        - Reward >= 0.4 -> likely
        - Reward < 0.4 -> likely（即使很差也保留探索机会）
        """
        if reward >= 0.7:
            return EvaluationVerdict.SURE
        if reward >= 0.4:
            return EvaluationVerdict.LIKELY
        # 修复 (2026-11-18): 原实现这里也是返回 LIKELY，导致
        # EvaluationVerdict.IMPOSSIBLE 永远不会被返回 —— ToT 树里"死路"的概念
        # 形同虚设，只能靠 is_pruned 硬剪枝兜底。
        # 现在让低分分子真正被判为 IMPOSSIBLE，恢复 ToT 的剪枝能力。
        return EvaluationVerdict.IMPOSSIBLE


# =============================================================================
# Section 5: 外部物理评估器主类
# =============================================================================

class GBMPhysicalEvaluator:
    """
    GBM 外部物理评估器。

    这是唯一对外暴露的接口类。它整合了：
    1. Vina 对接（Docking Tool）
    2. ADMET 计算 (admet-ai: DILI + hERG + BBB + Pgp + AMES + QED;
       SMARTS 警示结构作为可解释性标注层)
    3. RDKit 描述符
    4. MPO Reward 计算（加权几何平均 + 硬截断 + 端点下限）
    5. 自然语言反馈生成（供 Prompt 注入使用）

    使用方式（参考 chemcrow/tools/rdkit.py 的 _run() 模式）:

        evaluator = GBMPhysicalEvaluator(
            vina_executable="/path/to/qvina",
            receptor_pdbqt="/path/to/EGFR.pdbqt",
            vina_center=(x, y, z),
            vina_size=(20, 20, 20),
        )

        # 评估单个 SMILES
        result = evaluator.evaluate("CCO")
        print(result.verdict)           # EvaluationVerdict.SURE/LIKELY/IMPOSSIBLE
        print(result.reward)            # 0.85
        print(result.build_feedback_text())  # "Vina得分=-9.2 kcal/mol | ..."

        # 批量评估
        results = evaluator.evaluate_batch(["CCO", "c1ccccc1", ...])
        for r in results:
            print(r.verdict, r.reward)

        # 获取反馈 Prompt（注入给 LLM）
        feedback = evaluator.build_feedback_for_llm(result)
    """

    def __init__(
        self,
        vina_executable: Optional[str] = None,
        receptor_pdbqt: Optional[str] = None,
        vina_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        vina_size: Tuple[float, float, float] = (20.0, 20.0, 20.0),
        reward_weights: Optional[Dict[str, float]] = None,
        hard_cutoffs: Optional[Dict[str, float]] = None,
        enable_vina: bool = True,
        enable_dili: bool = True,
        enable_bbb: bool = True,
        use_admet_ai: bool = True,
        vina_exhaustiveness: int = 8,
        vina_cpu: int = 4,
    ):
        """
        Args:
            vina_executable: QVina/Wina 路径，None=伪模式
            receptor_pdbqt: 受体 PDBQT 文件路径
            vina_center: 对接盒子中心
            vina_size: 对接盒子尺寸
            reward_weights: 自定义 Reward 权重
            hard_cutoffs: 自定义硬截断阈值
            enable_vina: 是否启用 Vina 计算（可关闭用于快速测试）
            enable_dili: 是否启用 DILI 计算
            enable_bbb: 是否启用 BBB 计算
            use_admet_ai: 是否用 admet-ai 打分（默认 True）；False 回退到旧的
                SMARTS 启发式，仅用于消融对比。
            vina_exhaustiveness: Vina 搜索强度（8 = 标准；1 可快 6 倍但噪声更大）
            vina_cpu: Vina 使用的 CPU 核数
        """
        self._vina_tool = None
        self._admet = ADMETCalculator(use_admet_ai=use_admet_ai)
        self._mpo = MPORewardCalculator(
            custom_weights=reward_weights,
            custom_cutoffs=hard_cutoffs,
        )
        self._enable_vina = enable_vina and (vina_executable is not None or
                                            (receptor_pdbqt is not None and
                                             os.path.isfile(receptor_pdbqt)))
        self._enable_dili = enable_dili
        self._enable_bbb = enable_bbb

        if self._enable_vina:
            self._vina_tool = VinaDockingTool(
                vina_executable=vina_executable,
                receptor_pdbqt=receptor_pdbqt,
                center=vina_center,
                size=vina_size,
                exhaustiveness=vina_exhaustiveness,
                cpu=vina_cpu,
            )
            backend = getattr(self._vina_tool, "_backend", "pseudo")
            if backend == "pseudo":
                self._enable_vina = False
                logger.warning(
                    "[GBMPhysicalEvaluator] Vina 真实对接不可用 —— "
                    "退回伪对接模式（基于 MW/LogP）"
                )
            else:
                logger.info(
                    f"[GBMPhysicalEvaluator] Vina enabled (backend={backend}, "
                    f"exh={vina_exhaustiveness}, cpu={vina_cpu})"
                )
        else:
            logger.warning("[GBMPhysicalEvaluator] Vina DISABLED — using pseudo-docking based on MW/LogP")

    def evaluate(
        self,
        smiles: str,
        baseline_dili: Optional[float] = None,
        baseline_herg: Optional[float] = None,
    ) -> PhysicalEvaluationResult:
        """
        评估单个 SMILES，支持 Delta Scoring（相对变化评估）。

        Args:
            smiles: 分子 SMILES 字符串
            baseline_dili: 父节点的原始 DILI 概率（用于 Delta Scoring）
            baseline_herg: 父节点的原始 hERG 概率（用于 Delta Scoring）

        调用顺序（无依赖，可并行化）：
        1. RDKit 描述符（必需，先算）
        2. ADMET（DILI + hERG + BBB）
        3. Vina 对接
        4. MPO Reward
        5. 硬截断检查（应用 Delta Scoring）
        """
        start_time = time.time()
        result = PhysicalEvaluationResult(
            smiles=smiles,
            baseline_dili=baseline_dili,
            baseline_herg=baseline_herg,
        )

        # =========================================================
        # 终极异常防御罩：任何步骤崩溃都不中断程序
        # =========================================================
        try:
            # ---- Step 1: RDKit 描述符 ----
            rd_data = self._compute_rdkit_descriptors(smiles)
            result.rd_tpsa = rd_data['tpsa']
            result.rd_mw = rd_data['mw']
            result.rd_logp = rd_data['logp']
            result.rd_hbd = rd_data['hbd']
            result.rd_hba = rd_data['hba']
            result.rd_n_rotatable = rd_data['n_rotatable']
            result.rd_n_rings = rd_data['n_rings']

            # ---- Step 2: ADMET (DILI + hERG + BBB + Pgp + AMES + QED) ----
            # 数据来源: admet-ai (52 端点 Swin-Transformer 预训练模型)
            if self._enable_dili or self._enable_bbb:
                admet = self._admet.compute(smiles)
                if self._enable_dili:
                    result.dili_prob = admet['dili_prob']
                    result.dili_norm = 1.0 - admet['dili_prob']
                    result.dili_alert_matches = admet['dili_alert_matches']
                # hERG 总是计算（不依赖开关）
                result.herg_prob = admet['herg_prob']
                result.herg_norm = 1.0 - admet['herg_prob']
                result.herg_alert_matches = admet['herg_alert_matches']
                if self._enable_bbb:
                    result.bbb_score = admet['bbb_score']
                    result.bbb_norm  = admet['bbb_score']
                # 新增端点 (2026-11-18): Pgp 外排 / AMES 致突变 / QED
                pgp = admet.get('pgp_prob')
                if pgp is not None and pgp == pgp:  # not NaN
                    result.pgp_prob = float(pgp)
                    result.pgp_norm = 1.0 - float(pgp)
                ames = admet.get('ames_prob')
                if ames is not None and ames == ames:
                    result.ames_prob = float(ames)
                    result.ames_norm = 1.0 - float(ames)
                qed = admet.get('qed')
                if qed is not None and qed == qed:
                    result.qed = float(qed)
                    result.qed_norm = float(qed)

            # ---- Step 3: Vina 对接 ----
            if self._enable_vina and self._vina_tool is not None:
                vina_score, vina_error = self._vina_tool.dock(smiles)
                result.vina_score = vina_score
                result.vina_error = vina_error
                if vina_score is not None:
                    result.vina_norm = self._mpo.normalize_vina(vina_score)
                else:
                    result.vina_timeout = (vina_error == "Vina timeout")
                    result.vina_error = vina_error
            else:
                result.vina_score = self._vina_tool._pseudo_dock(smiles) if self._vina_tool else -7.5
                result.vina_norm = self._mpo.normalize_vina(result.vina_score)

            # ---- Step 4: 其他归一化分数 ----
            result.tpsa_norm = self._mpo.normalize_tpsa(result.rd_tpsa)
            result.mw_norm = self._mpo.normalize_mw(result.rd_mw)

            # ---- Step 5: 硬截断检查（应用 Delta Scoring）----
            is_pruned, prune_reason = self._mpo.check_hard_cutoffs(
                result,
                baseline_dili=baseline_dili,
                baseline_herg=baseline_herg,
            )
            result.is_pruned = is_pruned
            result.prune_reason = prune_reason

            # ---- Step 6: MPO Reward ----
            result.reward = self._mpo.compute_reward(
                vina_norm=result.vina_norm,
                dili_norm=result.dili_norm,
                herg_norm=result.herg_norm,
                bbb_norm=result.bbb_norm,
                tpsa_norm=result.tpsa_norm,
                mw_norm=result.mw_norm,
                pgp_norm=result.pgp_norm,
                ames_norm=result.ames_norm,
                qed_norm=result.qed_norm,
            )

            # ---- Step 7: 决定 verdict ----
            result.verdict = self._mpo.decide_verdict(result.reward, result.is_pruned)

        except Exception as e:
            logger.warning(
                f"[物理引擎跳过] 无法处理该结构: {smiles[:60]}... 原因: {type(e).__name__}: {e}"
            )
            result.vina_score = 0.0
            result.dili_prob = 1.0
            result.dili_norm = 0.0
            result.herg_prob = 1.0
            result.herg_norm = 0.0
            result.bbb_score = 0.0
            result.bbb_norm = 0.0
            result.pgp_prob = 1.0
            result.pgp_norm = 0.0
            result.ames_prob = 1.0
            result.ames_norm = 0.0
            result.qed = 0.0
            result.qed_norm = 0.0
            result.rd_tpsa = 0.0
            result.rd_mw = 0.0
            result.rd_logp = 0.0
            result.rd_hbd = 0
            result.rd_hba = 0
            result.rd_n_rotatable = 0
            result.rd_n_rings = 0
            result.tpsa_norm = 0.0
            result.mw_norm = 0.0
            result.vina_norm = 0.0
            result.reward = 0.0
            # 修复 (2026-11-18): 原代码在这里设 is_pruned=False，导致计算失败的
            # 分子（reward=0 但 verdict=LIKELY）在 ToT 里持续消耗 LLM 预算却
            # 永远产不出好结果。计算失败应当直接剪枝。
            result.is_pruned = True
            result.prune_reason = f"计算异常: {type(e).__name__}: {str(e)[:80]}"
            result.verdict = EvaluationVerdict.LIKELY

        elapsed = time.time() - start_time
        alert_info = ""
        if result.dili_alert_matches:
            alert_info += f" dili_alerts={result.dili_alert_matches}"
        if result.herg_alert_matches:
            alert_info += f" herg_alerts={result.herg_alert_matches}"
        logger.debug(
            f"[PhysicalEval] {smiles[:30]}... -> "
            f"verdict={result.verdict.value}, reward={result.reward:.4f}, "
            f"vina={result.vina_score}, dili={result.dili_prob}, "
            f"herg={result.herg_prob}, bbb={result.bbb_score}{alert_info} ({elapsed:.2f}s)"
        )

        return result

    def evaluate_batch(self, smiles_list: List[str]) -> List[PhysicalEvaluationResult]:
        """
        批量评估多个 SMILES。

        ADMET 部分走 admet-ai 批量推理（CPU 上 ~27 ms/分子，逐个调用是
        4336 ms/分子，差 12×）。Vina 部分仍是逐个对接（可由调用方用
        rescore_parallel.py 的多进程方案并行化）。
        """
        if not smiles_list:
            return []

        # ---- 预取 ADMET（批量 + 缓存）----
        admet_rows = self._admet.compute_batch(smiles_list)

        results = []
        for smi, admet in zip(smiles_list, admet_rows):
            try:
                results.append(self._evaluate_with_admet(smi, admet))
            except Exception as e:
                logger.warning(
                    f"[物理引擎跳过] 无法处理该结构: {smi[:60]}... "
                    f"原因: {type(e).__name__}: {e}"
                )
                res = PhysicalEvaluationResult(smiles=smi)
                res.vina_score = 0.0
                res.dili_prob = 1.0
                res.dili_norm = 0.0
                res.herg_prob = 1.0
                res.herg_norm = 0.0
                res.bbb_score = 0.0
                res.bbb_norm = 0.0
                res.pgp_prob = 1.0
                res.pgp_norm = 0.0
                res.ames_prob = 1.0
                res.ames_norm = 0.0
                res.qed = 0.0
                res.qed_norm = 0.0
                res.reward = 0.0
                res.is_pruned = True
                res.prune_reason = f"计算异常: {type(e).__name__}: {str(e)[:80]}"
                res.verdict = EvaluationVerdict.LIKELY
                results.append(res)
        return results

    def _evaluate_with_admet(self, smiles: str, admet: Dict[str, Any]) -> PhysicalEvaluationResult:
        """用预先算好的 ADMET 结果做评估（evaluate 的内部版本）。"""
        start_time = time.time()
        result = PhysicalEvaluationResult(smiles=smiles)

        rd_data = self._compute_rdkit_descriptors(smiles)
        result.rd_tpsa = rd_data['tpsa']
        result.rd_mw = rd_data['mw']
        result.rd_logp = rd_data['logp']
        result.rd_hbd = rd_data['hbd']
        result.rd_hba = rd_data['hba']
        result.rd_n_rotatable = rd_data['n_rotatable']
        result.rd_n_rings = rd_data['n_rings']

        if self._enable_dili:
            result.dili_prob = admet.get('dili_prob', 0.5)
            result.dili_norm = 1.0 - result.dili_prob
            result.dili_alert_matches = admet.get('dili_alert_matches', [])
        result.herg_prob = admet.get('herg_prob', 0.5)
        result.herg_norm = 1.0 - result.herg_prob
        result.herg_alert_matches = admet.get('herg_alert_matches', [])
        if self._enable_bbb:
            result.bbb_score = admet.get('bbb_score', 0.3)
            result.bbb_norm = result.bbb_score

        pgp = admet.get('pgp_prob')
        if pgp is not None and pgp == pgp:
            result.pgp_prob = float(pgp)
            result.pgp_norm = 1.0 - float(pgp)
        ames = admet.get('ames_prob')
        if ames is not None and ames == ames:
            result.ames_prob = float(ames)
            result.ames_norm = 1.0 - float(ames)
        qed = admet.get('qed')
        if qed is not None and qed == qed:
            result.qed = float(qed)
            result.qed_norm = float(qed)

        # ---- Vina 对接 ----
        if self._enable_vina and self._vina_tool is not None:
            vina_score, vina_error = self._vina_tool.dock(smiles)
            result.vina_score = vina_score
            result.vina_error = vina_error
            if vina_score is not None:
                result.vina_norm = self._mpo.normalize_vina(vina_score)
            else:
                result.vina_timeout = (vina_error == "Vina timeout")
                result.vina_error = vina_error
        else:
            result.vina_score = self._vina_tool._pseudo_dock(smiles) if self._vina_tool else -7.5
            result.vina_norm = self._mpo.normalize_vina(result.vina_score)

        result.tpsa_norm = self._mpo.normalize_tpsa(result.rd_tpsa)
        result.mw_norm = self._mpo.normalize_mw(result.rd_mw)

        is_pruned, prune_reason = self._mpo.check_hard_cutoffs(result)
        result.is_pruned = is_pruned
        result.prune_reason = prune_reason

        result.reward = self._mpo.compute_reward(
            vina_norm=result.vina_norm,
            dili_norm=result.dili_norm,
            herg_norm=result.herg_norm,
            bbb_norm=result.bbb_norm,
            tpsa_norm=result.tpsa_norm,
            mw_norm=result.mw_norm,
            pgp_norm=result.pgp_norm,
            ames_norm=result.ames_norm,
            qed_norm=result.qed_norm,
        )
        result.verdict = self._mpo.decide_verdict(result.reward, result.is_pruned)
        logger.debug(
            f"[PhysicalEval] {smiles[:30]}... -> verdict={result.verdict.value}, "
            f"reward={result.reward:.4f} ({time.time() - start_time:.2f}s)"
        )
        return result

    def build_feedback_for_llm(self, result: PhysicalEvaluationResult) -> str:
        """
        构建注入给 LLM 的反馈 Prompt。

        示例修改前:
            请基于上次的打分继续优化：{llm_fake_score}

        示例修改后:
            物理引擎真实测试结果如下：
            Vina得分=-9.2 kcal/mol (强结合模式)。
            肝毒性(DILI)=0.12 (低风险)。
            BBB穿透=0.78 (高CNS渗透)。
            理化性质: MW=452.3, LogP=3.21, TPSA=68.4Å²。
            Reward=0.84。
            请在保持原有结合模式的前提下，调整基团...
        """
        return result.build_feedback_text()

    def generate_toxicity_feedback(self, result: PhysicalEvaluationResult) -> str:
        """
        生成具象化的毒性反馈（供 LLM 在下一轮修改时使用）。

        如果命中了结构警报，返回具体的化学修改建议；
        如果没有命中但 ML 分数偏高，给出方向性建议。

        示例输出:
            "hERG 风险上升！因为新生成的分子匹配到了 '氯苯基团' 结构。
             氯苯是常见的 hERG 警示结构。建议将氯原子替换为极性基团如 -OH、-NH2，
             或移动至分子极性端。"
        """
        lines = []

        # ---- hERG 反馈 ----
        if result.herg_alert_matches:
            # 命中了 SMARTS 警报：给出精确的化学修改建议
            alerts_str = "、".join(result.herg_alert_matches)
            herg_level = "极高" if result.herg_prob > 0.85 else "升高"
            lines.append(
                f"hERG 风险 {herg_level}！因为你生成的分子匹配到了 hERG 警示结构：'{alerts_str}'。"
            )

            # 从元数据中提取修改建议
            for alert in result.herg_alert_matches:
                for key, meta in ADMETCalculator.hERG_ALERT_META.items():
                    if meta.get('name') == alert:
                        lines.append(meta.get('fix', ''))
                        break
                else:
                    # 没有精确匹配，给出通用建议
                    lines.append(
                        "请考虑用极性基团（-OH、-NH2、-COOH、-SO2NH2）替换该警示基团，"
                        "以降低 hERG 阻断风险。"
                    )
        elif result.herg_prob is not None and result.herg_prob > 0.7:
            # 没有命中警报但 ML 分数偏高：可能是假阳性，给出方向性建议
            lines.append(
                f"hERG 风险轻微升高（prob={result.herg_prob:.2f}），"
                "但未检测到已知结构警报。可能是分子整体高脂溶性导致的。"
                "建议适度增加分子的极性表面积（TPSA），或在 Lipinski 5 规则边界处优化。"
            )

        # ---- DILI 反馈 ----
        if result.dili_alert_matches:
            alerts_str = "、".join(result.dili_alert_matches)
            dili_level = "极高" if result.dili_prob > 0.9 else "升高"
            lines.append(
                f"肝毒性(DILI)风险 {dili_level}！因为你生成的分子匹配到了 '{alerts_str}' 结构。"
            )

            for alert in result.dili_alert_matches:
                for key, meta in ADMETCalculator.DILI_ALERT_META.items():
                    if meta.get('name') == alert:
                        lines.append(meta.get('fix', ''))
                        break
                else:
                    lines.append(
                        "该结构具有潜在的反应性代谢物风险。建议使用化学稳定性更好的生物电子等排体替换。"
                    )
        elif result.dili_prob is not None and result.dili_prob > 0.8:
            lines.append(
                f"肝毒性(DILI)风险轻微升高（prob={result.dili_prob:.2f}），"
                "但未检测到已知结构警报。可能是高 MW(>700) 或高 LogP(>5.0) 导致的。"
                "请控制分子量在 500 以内，LogP 在 4.0 以下。"
            )

        # ---- P-gp 外排反馈 (2026-11-18 新增) ----
        if result.pgp_prob is not None and result.pgp_prob > 0.7:
            lines.append(
                f"P-gp 外排风险 {result.pgp_prob:.2f}（高）。"
                "外排会抵消 BBB 穿透能力，即使分子能过血脑屏障也会被泵回。"
                "建议在碱性胺上引入极性取代基、或降低胺碱性（酰胺化/磺酰胺化），"
                "也可考虑用能被 P-gp 识别的结构改造策略降低亲和力。"
            )

        # ---- AMES 致突变反馈 (2026-11-18 新增) ----
        if result.ames_prob is not None and result.ames_prob > 0.6:
            lines.append(
                f"致突变风险(AMES) {result.ames_prob:.2f}（较高）。"
                "建议检查是否存在可被 CYP450 代谢活化的芳香胺、硝基或亚硝基结构，"
                "并考虑用电子吸取代基降低芳胺的氧化倾向。"
            )

        # ---- Delta Scoring 反馈 ----
        if result.baseline_dili is not None and result.dili_prob is not None:
            delta = result.dili_prob - result.baseline_dili
            if delta > 0.05:
                lines.append(
                    f"注意：相比父节点，DILI 毒性上升了 {delta:+.3f}。"
                    "本次优化在引入新基团时需特别关注代谢稳定性。"
                )

        if result.baseline_herg is not None and result.herg_prob is not None:
            delta = result.herg_prob - result.baseline_herg
            if delta > 0.05:
                lines.append(
                    f"注意：相比父节点，hERG 风险上升了 {delta:+.3f}。"
                    "新增的取代基可能引入了 hERG 阻断药效团，请优先替换。"
                )

        if not lines:
            lines.append("毒性评估通过，未检测到已知结构警报。")

        return "\n".join(lines)

    def build_batch_feedback(
        self,
        results: List[PhysicalEvaluationResult],
        top_k: int = 3,
    ) -> str:
        """
        为批量评估结果生成汇总反馈。

        用于多候选排序后，将 Top-K 的物理数据注入给 LLM 进行下一轮优化。
        """
        if not results:
            return "无有效候选分子。"

        # 按 Reward 排序
        sorted_results = sorted(results, key=lambda r: r.reward, reverse=True)
        top_results = sorted_results[:top_k]

        lines = ["【物理引擎批量评估结果（Top-K）】"]
        lines.append(f"共评估 {len(results)} 个分子，Top-{len(top_results)} 如下：\n")

        for i, r in enumerate(top_results, 1):
            status = "【已剪枝】" if r.is_pruned else "【通过】"
            lines.append(
                f"{i}. {status} {r.smiles[:50]}...\n"
                f"   {r.build_feedback_text()}"
            )

        # 剪枝统计
        pruned_count = sum(1 for r in results if r.is_pruned)
        if pruned_count > 0:
            lines.append(f"\n注意：{pruned_count}/{len(results)} 个分子因硬截断被剪枝。")

        return "\n".join(lines)

    def _compute_rdkit_descriptors(self, smiles: str) -> Dict[str, Any]:
        """计算 RDKit 分子描述符"""
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return self._default_rdkit_data()
            try:
                Chem.SanitizeMol(mol, catchErrors=True)
            except Exception:
                pass
            return {
                'tpsa': float(Descriptors.TPSA(mol)),
                'mw': float(Descriptors.MolWt(mol)),
                'logp': float(Crippen.MolLogP(mol)),
                'hbd': int(Descriptors.NumHDonors(mol)),
                'hba': int(Descriptors.NumHAcceptors(mol)),
                'n_rotatable': int(Descriptors.NumRotatableBonds(mol)),
                'n_rings': int(Chem.rdMolDescriptors.CalcNumRings(mol)),
            }
        except Exception:
            return self._default_rdkit_data()

    def _default_rdkit_data(self) -> Dict[str, Any]:
        return {
            'tpsa': 0.0, 'mw': 0.0, 'logp': 0.0,
            'hbd': 0, 'hba': 0, 'n_rotatable': 0, 'n_rings': 0,
        }
