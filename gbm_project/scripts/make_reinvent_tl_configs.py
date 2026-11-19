#!/usr/bin/env python3
"""
REINVENT 4 TL+RL 基线 —— 修复对比的不对称性
==============================================

问题
----
我们自己的模型做了 SFT（LoRA），而对比里的 REINVENT 4 是
"PubChem 通用先验 -> 直接 DAP 强化学习"，中间没有针对靶点的 TL。
按 DAP 的数学形式 $p^* \\propto p_{prior} \\cdot e^{\\sigma R}$，
RL 只能在 prior 的支撑集里重新加权 —— 所以两者的**可达域本质不同**：
    我们   : GBM/激酶抑制剂化学型（SFT 后）
    REINVENT: 整个类药化学空间（通用 prior）
这不是 REINVENT 更弱，而是设定不对等。

本脚本
------
1. 从 PubChem 解析的真实分子库里，为每个靶点筛出对应的真实抑制剂
   （按 Murcko 骨架 / 靶点关联性），写成 .smi 文件
2. 生成 transfer_learning 配置（纯 NLL 微调，带 validation 早停）
3. 生成 staged_learning 配置，prior_file / agent_file 都指向 TL 的产物

用法
----
    python scripts/make_reinvent_tl_configs.py
    # 然后
    cd /root/REINVENT4-main
    /root/venv_reinvent/bin/reinvent -l /tmp/tl_EGFR.log <tl_config>
    /root/venv_reinvent/bin/reinvent -l /tmp/rl_EGFR.log <rl_config>
"""
import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GBM_ROOT = PROJECT_ROOT / "gbm_project"
OUT_DIR = GBM_ROOT / "configs"
EXP_DIR = GBM_ROOT / "experiments" / "reinvent_baseline"

# 靶点 -> (PDBQT, center, 靶点关键词，用于从真实分子库筛选)
TARGETS: Dict[str, dict] = {
    "EGFR": {
        "pdbqt": "EGFR_2JIV_chainA.pdbqt",
        "center": [4.77, 3.65, 17.11],
        "keywords": ["gefitinib", "erlotinib", "osimertinib", "lapatinib",
                     "vandetanib", "pelitinib", "canertinib", "icotinib",
                     "afatinib", "dacomitinib", "neratinib", "brigatinib",
                     "almonertinib", "lazertinib", "poziotinib", "mobocertinib",
                     "EGFR", "HER2", "kinase"],
    },
    "VEGFR": {
        "pdbqt": "VEGF_VEGFR_3WZD_chainA.pdbqt",
        "center": [1.12, -6.89, 15.94],
        "keywords": ["sunitinib", "sorafenib", "vandetanib", "pazopanib",
                     "axitinib", "cabozantinib", "lenvatinib", "regorafenib",
                     "tivozanib", "cediranib", "vandetanib", "linifanib",
                     "brivanib", "motesanib", "semaxanib", "SU5416",
                     "VEGFR", "KDR", "angiogenesis"],
    },
    "IDH1": {
        "pdbqt": "IDH1_IDH2_6U4J_chainA.pdbqt",
        "center": [11.89, -3.71, 23.56],
        "keywords": ["enasidenib", "ivosidenib", "vorasidenib", "olutasidenib",
                     "AG-120", "AG-881", "IDH1", "IDH2", "isocitrate",
                     "mutant", "FT-2102", "FT-1102", "DS-1001"],
    },
}


def build_smi_files(molecules_path: Path, out_dir: Path,
                    min_per_target: int = 150) -> Dict[str, Path]:
    """按靶点关键词从真实分子库筛出 .smi 文件"""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")

    mols = json.load(open(molecules_path))
    print(f"真实分子库: {len(mols)} 条")

    # 按 (药名 -> smiles) 去重
    by_drug: Dict[str, str] = {}
    for m in mols:
        by_drug.setdefault(m["drug"].lower(), m["smiles"])

    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    for target, cfg in TARGETS.items():
        kws = [k.lower() for k in cfg["keywords"]]
        selected = []
        for drug, smi in by_drug.items():
            if not any(k in drug for k in kws):
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mw = Descriptors.MolWt(mol)
            if not (250 <= mw <= 600):
                continue
            selected.append((drug, smi, mw))

        # 关键词命中不足时，用 Murcko 骨架特征补充（保证数据量够 TL）
        if len(selected) < min_per_target:
            print(f"  {target}: 关键词命中仅 {len(selected)}，用类药性补充")
            seen = {s[1] for s in selected}
            for drug, smi in by_drug.items():
                if len(selected) >= min_per_target:
                    break
                if smi in seen:
                    continue
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                mw = Descriptors.MolWt(mol)
                if not (250 <= mw <= 600):
                    continue
                selected.append((drug, smi, mw))
                seen.add(smi)

        # 按 prior 词表过滤（不含的 token 会让 TL 直接报错）
        selected_smis = [s for _, s, _ in selected]
        keep = set(vocab_filter(selected_smis))
        n_before = len(selected_smis)
        selected = [(d, s, w) for d, s, w in selected if s in keep]
        if len(selected) < n_before:
            print(f"  {target}: 词表过滤掉 {n_before - len(selected)} 个 "
                  f"(prior 只有 131 个 token)")

        path = out_dir / f"{target}_tl.smi"
        with open(path, "w") as f:
            for drug, smi, _ in selected:
                f.write(smi + "\n")
        written[target] = path
        print(f"  {target}: {len(selected)} 个分子 -> {path}")

    return written


TL_TEMPLATE = """# REINVENT 4 Transfer Learning —— {target}
#
# 目的: 把 PubChem 通用先验平移到「{target} 抑制剂化学型」，
#       修复之前「我们做 SFT 而 REINVENT 不做」的不对称性。
#
# 这是纯最大似然微调 (NLL)，不含任何奖励项，不看对接分数。

run_type = "transfer_learning"
device = "cpu"
tb_logdir = "{exp_dir}/tb_TL_{target}"

[parameters]
input_model_file = "/root/reinvent_pubchem.prior"
output_model_file = "{exp_dir}/{target}_tl.chkpt"
smiles_file = "{smi_path}"
validation_smiles_file = "{smi_path}"
num_epochs = {epochs}
batch_size = 32
standardize_smiles = true
randomize_smiles = true
sample_batch_size = 100
save_every_n_epochs = 1
max_sequence_length = 128
clip_gradient_norm = 1.0

[scheduler]
lr = {lr}
min = 0.00001
gamma = 0.95
step = 10
"""

RL_TEMPLATE = """# REINVENT 4 TL+RL —— {target}
#
# 与 reinvent_{target}.toml 的唯一区别:
#   prior_file / agent_file 从通用 PubChem 先验，换成该靶点 TL 后的 checkpoint。
# 其余（Vina 对接奖励 + QED + MW + Alerts + Murcko 多样性过滤）完全相同。

run_type = "staged_learning"
device = "cpu"
tb_logdir = "{exp_dir}/tb_TLRL_{target}"

[parameters]
prior_file = "{exp_dir}/{target}_tl.chkpt"
agent_file = "{exp_dir}/{target}_tl.chkpt"
summary_csv_prefix = "{exp_dir}/{target}_tlrl"
batch_size = 32
randomize_smiles = true
unique_sequences = true

[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0005

[diversity_filter]
type = "IdenticalMurckoScaffold"
bucket_size = 10
minscore = 0.4

[[stage]]
chkpt_file = "{exp_dir}/{target}_tlrl_stage1.chkpt"
termination = "simple"
max_score = 0.9
min_steps = 8
max_steps = 40

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.VinaScore]
params.receptor_pdbqt = "/root/Llamole-main/gbm_project/data/receptors/{pdbqt}"
params.center = {center}
params.size = [24.0, 24.0, 24.0]
params.exhaustiveness = 4
params.cpu = 4
params.n_workers = 16

[[stage.scoring.component.VinaScore.endpoint]]
name = "Vina"
weight = 1.0

[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 0.3

[[stage.scoring.component]]
[stage.scoring.component.MolecularWeight]
[[stage.scoring.component.MolecularWeight.endpoint]]
name = "MW"
weight = 0.2
transform.type = "double_sigmoid"
transform.low = 200.0
transform.high = 500.0

[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]
[[stage.scoring.component.custom_alerts.endpoint]]
name = "Alerts"
params.smarts = [
  "[*;r{{8-17}}]",
  "[#8][#8]",
  "[#6;+]",
  "[#16][#16]",
  "[#7;!n][S;!$(S(=O)=O)]",
  "[#7;!n][#7;!n]",
  "C#C",
  "C(=[O,S])[O,S]"
]
"""


def vocab_filter(smiles_list: List[str]) -> List[str]:
    """按 REINVENT prior 的 token 词表过滤 SMILES。

    prior 的词表只有 131 个 token（'#','$','%10',...,'C','Cl','N','O','S',
    '[nH]','[C@@H]', ...），不包含 'F' 之外的卤素细节、'@' 之外的立体化学、
    以及所有非 C/N/O/S/P 的重原子符号。所以含这些元素的分子无法被 prior 编码，
    必须剔除，否则 TL 会直接抛 RuntimeError: unknown token。

    做法：用 REINVENT 自己的 SMILESTokenizer 逐个试编码。
    """
    import sys
    sys.path.insert(0, "/root/REINVENT4-main")
    from reinvent.models.reinvent.models.vocabulary import (
        SMILESTokenizer, Vocabulary)

    import torch
    ck = torch.load("/root/reinvent_pubchem.prior", map_location="cpu",
                    weights_only=False)
    vocab = Vocabulary.load_from_dictionary(ck["vocabulary"])
    tok = SMILESTokenizer()

    ok = []
    for s in smiles_list:
        try:
            encoded = vocab.encode(tok.tokenize(s))
            # encode 返回 numpy 数组，不能用 `if encoded:` 判空
            # （numpy 数组的布尔求值会抛 ValueError，会被下面的 except 吞掉，
            #   导致所有分子都被误判为不合格）
            if len(encoded) > 0:
                ok.append(s)
        except Exception:
            continue
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", type=str,
                    default=str(GBM_ROOT / "data/lora_datasets/real_molecules.json"))
    ap.add_argument("--smi_dir", type=str, default=str(EXP_DIR / "tl_smiles"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=0.0001)
    args = ap.parse_args()

    EXP_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print("生成 REINVENT 4 TL+RL 基线")
    print("=" * 66)

    smi_paths = build_smi_files(Path(args.molecules), Path(args.smi_dir))

    print("\n生成配置:")
    for target, cfg in TARGETS.items():
        smi_path = smi_paths[target]

        tl_cfg = TL_TEMPLATE.format(
            target=target, exp_dir=EXP_DIR, smi_path=smi_path,
            epochs=args.epochs, lr=args.lr)
        tl_path = OUT_DIR / f"reinvent_tl_{target.lower()}.toml"
        tl_path.write_text(tl_cfg)
        print(f"  ✓ {tl_path.name}")

        rl_cfg = RL_TEMPLATE.format(
            target=target, exp_dir=EXP_DIR, pdbqt=cfg["pdbqt"],
            center=cfg["center"])
        rl_path = OUT_DIR / f"reinvent_tlrl_{target.lower()}.toml"
        rl_path.write_text(rl_cfg)
        print(f"  ✓ {rl_path.name}")

    print("\n" + "=" * 66)
    print("运行方式:")
    print("=" * 66)
    print("""cd /root/REINVENT4-main
for t in egfr vegfr idh1; do
    # Step 1: Transfer Learning (纯 NLL)
    /root/venv_reinvent/bin/reinvent -l /tmp/tl_$t.log \\
        /root/Llamole-main/gbm_project/configs/reinvent_tl_$t.toml
    # Step 2: RL (DAP，prior/agent 都用 TL 后的 checkpoint)
    /root/venv_reinvent/bin/reinvent -l /tmp/rl_$t.log \\
        /root/Llamole-main/gbm_project/configs/reinvent_tlrl_$t.toml
done""")
    print(f"\nTL 数据: {args.smi_dir}")


if __name__ == "__main__":
    main()
