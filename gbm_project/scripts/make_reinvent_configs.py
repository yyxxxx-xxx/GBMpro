#!/usr/bin/env python3
"""
生成 REINVENT 靶点条件化 RL 配置（三个靶点）
=============================================

用法:
    python scripts/make_reinvent_configs.py
    cd /root/REINVENT4-main
    for t in EGFR VEGFR IDH1; do
        /root/venv_reinvent/bin/reinvent -l /tmp/reinvent_$t.log \
            /root/Llamole-main/gbm_project/configs/reinvent_$t.toml
    done
"""

import sys
from pathlib import Path

GBM_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = GBM_ROOT / "configs"
EXP_DIR = GBM_ROOT / "experiments" / "reinvent_baseline"

TARGETS = {
    "EGFR": {
        "pdbqt": "EGFR_2JIV_chainA.pdbqt",
        "center": [4.77, 3.65, 17.11],
    },
    "VEGFR": {
        "pdbqt": "VEGF_VEGFR_3WZD_chainA.pdbqt",
        "center": [1.12, -6.89, 15.94],
    },
    "IDH1": {
        "pdbqt": "IDH1_IDH2_6U4J_chainA.pdbqt",
        "center": [11.89, -3.71, 23.56],
    },
}

TEMPLATE = """# REINVENT4 靶点条件化强化学习 —— {target}
#
# 用真实 AutoDock Vina 对接分数作为奖励信号（自定义 VinaScore 组件），
# 加上 QED / MW / 结构警示过滤，对模型做 DAP 策略梯度微调，
# 使其生成与 {target} 口袋匹配的分子。
#
# 这是最严格的第三方基线形式：专用化学语言模型 + 物理对接奖励 +
# 多样性过滤（IdenticalMurckoScaffold）。

run_type = "staged_learning"
device = "cpu"
tb_logdir = "{exp_dir}/tb_{target}"

[parameters]
prior_file = "/root/reinvent_pubchem.prior"
agent_file = "/root/reinvent_pubchem.prior"
summary_csv_prefix = "{exp_dir}/{target}_rl"
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
chkpt_file = "{exp_dir}/{target}_stage1.chkpt"
termination = "simple"
max_score = 0.9
min_steps = 8
max_steps = 40

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.VinaScore]
params.receptor_pdbqt = "{pdbqt_abs}"
params.center = [{center}]
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    receptors = GBM_ROOT / "data" / "receptors"

    for target, spec in TARGETS.items():
        pdbqt_abs = receptors / spec["pdbqt"]
        if not pdbqt_abs.exists():
            print(f"! 缺少受体: {pdbqt_abs}")
            continue
        content = TEMPLATE.format(
            target=target,
            exp_dir=EXP_DIR,
            pdbqt_abs=pdbqt_abs,
            center=", ".join(str(x) for x in spec["center"]),
        )
        out = OUT_DIR / f"reinvent_{target.lower()}.toml"
        out.write_text(content, encoding="utf-8")
        print(f"-> {out}")


if __name__ == "__main__":
    main()
