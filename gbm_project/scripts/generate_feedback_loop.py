#!/usr/bin/env python3
"""
反馈闭环生成（真实 Vina 对接）
============================

流程:
    生成 batch (N 个分子)
        -> 真实 Vina 对接 + admet-ai 打分
        -> 计算 MPO reward
        -> 选出 top-k / bottom-k 作为"好的反例"和"待改进的分子"
        -> 把物理反馈注入 prompt, 再生成下一轮
        -> 重复 R 轮

与 generate_tot_molecules.py 的区别:
    - ToT 是**结构分解式** (scaffold -> assembly -> SMILES)，需要模型输出中间
      推理节点；我们的 LoRA 是在 "Instruction/Input/Response: SMILES" 上训的，
      ToT 的 prompt 形态与之不匹配。
    - 本脚本是**单步迭代式**: 直接用训练时的 prompt 形态生成 SMILES，
      反馈以自然语言形式追加。prompt 形态与训练完全一致。

真实对接:
    使用 src/gbm_target_evaluator.py -> property_pipeline.docking.VinaDocking
    (vina Python API 1.2.7 + Meeko，复用 grid maps)

用法:
    python3 scripts/generate_feedback_loop.py \
        --target IDH1 --n 50 --rounds 3 --adapter <path>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "gbm_project" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "gbm_project" / "scripts"))

import torch  # noqa: E402
import yaml  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList  # noqa: E402
from peft import PeftModel  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402

RDLogger.DisableLog("rdApp")

from generate_with_new_adapter import (  # noqa: E402
    build_prompt, CELL_STATES, METRIC_VARIANTS, SMILES_RE,
    _best_fragment, _heavy_atoms, extract_candidate, repair_smiles,
)


class _NewlineStopper(StoppingCriteria):
    """训练 response 是单行 "SMILES: xxx"，遇到换行即停。"""

    def __init__(self, newline_token_id: int, prompt_len: int):
        self.newline_token_id = newline_token_id
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if input_ids.shape[1] <= self.prompt_len:
            return False
        return int(input_ids[0, -1]) == self.newline_token_id


def build_feedback_text(res) -> str:
    """把物理评估结果转成给 LLM 的自然语言反馈。

    关键: 必须**具体**。"DILI 高" 没用；"DILI=0.83，请移除导致肝毒性的
    芳环/酰胺结构" 才有用。
    """
    parts: List[str] = []

    v = res.vina_score
    if v is not None:
        if v > -8.0:
            parts.append(f"- Vina binding energy = {v:.2f} kcal/mol (WEAK; "
                         f"the ligand does not bind tightly enough. Strengthen the "
                         f"core scaffold, add hinge-binding H-bond donors/acceptors, "
                         f"or extend into a hydrophobic pocket)")
        elif v > -10.0:
            parts.append(f"- Vina binding energy = {v:.2f} kcal/mol (moderate; "
                         f"acceptable but could be stronger)")
        else:
            parts.append(f"- Vina binding energy = {v:.2f} kcal/mol (STRONG; "
                         f"keep this binding motif)")

    if res.dili_prob is not None:
        if res.dili_prob > 0.5:
            parts.append(f"- DILI (hepatotoxicity) probability = {res.dili_prob:.2f} "
                         f"(TOO HIGH - must be reduced. Prefer removing aryl "
                         f"halides, avoid lipophilic aromatic rings, reduce "
                         f"ester/thioester and nitro groups)")
        else:
            parts.append(f"- DILI probability = {res.dili_prob:.2f} (acceptable)")

    if res.herg_prob is not None:
        if res.herg_prob > 0.5:
            parts.append(f"- hERG (cardiotoxicity) probability = {res.herg_prob:.2f} "
                         f"(TOO HIGH - must be reduced. Avoid a basic tertiary "
                         f"amine flanked by lipophilic groups; lower LogP; "
                         f"add an electron-withdrawing group or reduce basicity)")
        else:
            parts.append(f"- hERG probability = {res.herg_prob:.2f} (acceptable)")

    if res.bbb_score is not None:
        if res.bbb_score > 0.5:
            parts.append(f"- BBB permeability = {res.bbb_score:.2f} (GOOD CNS "
                         f"penetration - required for GBM)")
        else:
            parts.append(f"- BBB permeability = {res.bbb_score:.2f} (TOO LOW - "
                         f"required for GBM. Reduce MW and TPSA, add fewer HBDs, "
                         f"or increase lipophilicity moderately)")

    if res.pgp_prob is not None and res.pgp_prob > 0.5:
        parts.append(f"- P-glycoprotein substrate probability = {res.pgp_prob:.2f} "
                     f"(HIGH - may be effluxed. Reduce MW and H-bond donor count)")

    if res.ames_prob is not None and res.ames_prob > 0.5:
        parts.append(f"- AMES mutagenicity probability = {res.ames_prob:.2f} "
                     f"(HIGH - avoid aromatic nitro groups and hydrazines)")

    if res.qed is not None:
        parts.append(f"- QED (drug-likeness) = {res.qed:.2f} "
                     f"({'low - improve oral drug-likeness' if res.qed < 0.4 else 'acceptable'})")

    parts.append(f"- Overall MPO reward = {res.reward:.4f}")

    if res.is_pruned and res.prune_reason:
        parts.append(f"- REJECTED for: {res.prune_reason}")

    return "\n".join(parts)


def feedback_refinement_instruction(
    base_instr: str,
    good_smiles: Optional[str],
    good_feedback: Optional[str],
    bad_smiles: str,
    bad_feedback: str,
    target: str,
) -> str:
    """构造第 2 轮及以后的 instruction。

    结构:
        <原始领域 prompt>
        <真实的物理反馈: 一个成功案例 + 一个待改进案例>
        <明确的修改指令>
    """
    ref = "\n\n" + "=" * 66 + "\n"
    ref += "ITERATIVE REFINEMENT: the molecules above were actually tested " \
           "with molecular docking and ADMET prediction.\n"
    ref += "=" * 66 + "\n\n"

    if good_smiles and good_feedback:
        ref += "SUCCESSFUL EXAMPLE (keep this binding mode and property profile):\n"
        ref += f"  SMILES: {good_smiles}\n"
        ref += "  Measured properties:\n"
        for line in good_feedback.split("\n"):
            ref += f"  {line}\n"
        ref += "\n"

    ref += f"MOLECULE THAT FAILED (SMILES: {bad_smiles}):\n"
    ref += "  Measured properties:\n"
    for line in bad_feedback.split("\n"):
        ref += f"  {line}\n"
    ref += "\n"

    ref += f"TASK: design a NEW molecule targeting {target} that keeps the "
    ref += "successful example's binding motif while fixing the failed molecule's "
    ref += "problems. Output exactly one SMILES.\n"

    return base_instr + ref


def load_model(base_model_path: str, llamole_adapter: str, gbm_adapter: str,
               device: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  [1/3] 基座模型: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )

    print(f"  [2/3] 合并 Llamole adapter: {llamole_adapter}")
    model = PeftModel.from_pretrained(model, llamole_adapter)
    model = model.merge_and_unload()

    print(f"  [3/3] 加载 GBM LoRA: {gbm_adapter}")
    model = PeftModel.from_pretrained(model, gbm_adapter)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, default="IDH1")
    ap.add_argument("--n", type=int, default=50, help="目标分子数（每轮补充）")
    ap.add_argument("--rounds", type=int, default=3, help="反馈迭代轮数")
    ap.add_argument("--batch_per_round", type=int, default=25,
                    help="每轮生成多少个候选")
    ap.add_argument("--keep_top", type=int, default=3, help="作为正例的分子数")
    ap.add_argument("--adapter", type=str,
                    default="/root/Llamole-main/saves/Llamole-Qwen2-7B-Instruct-Adapter-gbm-active")
    ap.add_argument("--base_model", type=str,
                    default="/root/Llamole-main/models/Qwen2-7B-Instruct")
    ap.add_argument("--llamole_adapter", type=str,
                    default="/root/Llamole-main/saves/Llamole-Qwen2-7B-Instruct-Adapter")
    ap.add_argument("--prompts", type=str,
                    default=str(PROJECT_ROOT / "gbm_project" / "configs" / "english_gbm_prompts.yaml"))
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.92)
    # 220 而非 128: 实测 128 会截断长 SMILES (如 "...n(C1)\N#"),
    # 导致第 1 轮 25 个候选只有 12 个有效 (48%)。
    ap.add_argument("--max_new_tokens", type=int, default=220)
    ap.add_argument("--min_heavy_atoms", type=int, default=12)
    ap.add_argument("--vina_exhaustiveness", type=int, default=4)
    ap.add_argument("--vina_cpu", type=int, default=8)
    ap.add_argument("--vina_workers", type=int, default=4,
                    help="并行评估的线程数 (对接占 99%% 耗时)")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="从已有的 <target>_all.json / <target>_history.json 续跑")
    args = ap.parse_args()

    device = f"cuda:{args.gpu_id}"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))

    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "gbm_project" / "final_results" / "comparative" /
        f"Ours_feedback_{args.target}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"反馈闭环生成 — target={args.target} n={args.n} rounds={args.rounds}")
    print("=" * 70)

    prompts = yaml.safe_load(open(args.prompts, encoding="utf-8"))

    print("加载模型 ...")
    model, tokenizer = load_model(args.base_model, args.llamole_adapter,
                                  args.adapter, device)
    nl_ids = tokenizer.encode("\n", add_special_tokens=False)
    newline_token_id = int(nl_ids[-1]) if nl_ids else tokenizer.eos_token_id

    # ---- 评估器（真实 Vina）----
    from gbm_target_evaluator import build_evaluator_for_target
    evaluator = build_evaluator_for_target(
        args.target, exhaustiveness=args.vina_exhaustiveness, cpu=args.vina_cpu)
    if evaluator is None:
        print("!! 无法构建评估器 —— 退出")
        return
    backend = getattr(evaluator._vina_tool, "_backend", "pseudo")
    print(f"对接后端: {backend}  (exh={args.vina_exhaustiveness})")
    if backend == "pseudo":
        print("!! 警告: 真实对接未启用，反馈将是假的")

    def gen_batch(instr: str, n: int, seed_base: int) -> List[str]:
        """生成 n 个候选分子。"""
        outs = []
        for i in range(n):
            torch.manual_seed(seed_base + i)
            full = f"Instruction: {instr}\nInput:\nResponse:"
            inputs = tokenizer(full, return_tensors="pt").to(device)
            ilen = inputs["input_ids"].shape[1]
            stopper = _NewlineStopper(newline_token_id, ilen)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([stopper]),
                )
            resp = tokenizer.decode(out[0][ilen:], skip_special_tokens=True)
            cand = extract_candidate(resp)
            fixed, _ = repair_smiles(cand)
            if fixed is None:
                continue
            if _heavy_atoms(fixed) < args.min_heavy_atoms:
                continue
            outs.append(fixed)
        return outs

    _local = threading.local()

    def evaluate_one(smi: str):
        """评估单个分子, 用线程私有的 evaluator。

        ⚠ 关键: **每个线程必须用独立的 evaluator 实例**。
        实测共享同一个 evaluator 时, Vina 的 grid map 被并发读写,
        分数会错乱 (实测 -9.077 kcal/mol 变成 31.744)。
        所以这里用 threading.local 给每个线程建自己的 evaluator。
        """
        ev = getattr(_local, "ev", None)
        if ev is None:
            ev = build_evaluator_for_target(
                args.target, exhaustiveness=args.vina_exhaustiveness,
                cpu=args.vina_cpu)
            _local.ev = ev
        return ev.evaluate(smi)

    def evaluate(smiles_list: List[str]):
        """并行评估一批分子 (对接占 99% 耗时)。"""
        if not smiles_list:
            return []
        res: List[Tuple[str, object]] = []
        n_workers = max(1, min(args.vina_workers, len(smiles_list)))

        def _one(smi):
            try:
                return smi, evaluate_one(smi)
            except Exception as e:
                return smi, e

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for smi, r in ex.map(_one, smiles_list):
                if isinstance(r, Exception):
                    print(f"  ! 评估失败 {smi[:40]}: {r}")
                else:
                    res.append((smi, r))
        return res

    # ---- 主循环 ----
    collected: List[dict] = []
    seen_smiles = set()
    history: List[list] = []
    start_round = 0

    # ---- 断点续跑 (2026-11-19) ----
    # 之前任务在第 4 轮评估期间被外部中断（SIGHUP/会话关闭）,
    # 一整轮的结果全丢。现在支持从已落盘的 JSON 继续。
    if args.resume and out_dir.exists():
        all_json = out_dir / f"{args.target}_all.json"
        hist_json = out_dir / f"{args.target}_history.json"
        if all_json.exists() and hist_json.exists():
            try:
                collected = json.loads(all_json.read_text(encoding="utf-8"))
                history = json.loads(hist_json.read_text(encoding="utf-8"))
                seen_smiles = {x["smiles"] for x in collected}
                start_round = len(history)
                print(f"↻ 断点续跑: 已载入 {len(collected)} 个分子, "
                      f"完成 {start_round} 轮, 从第 {start_round+1} 轮继续")
            except Exception as e:
                print(f"! 断点文件损坏, 从头开始: {e}")
                collected, history, seen_smiles, start_round = [], [], set(), 0
        else:
            print("! 无可用的断点文件, 从头开始")

    for rnd in range(start_round, args.rounds):
        print(f"\n{'='*70}\n第 {rnd+1}/{args.rounds} 轮\n{'='*70}")
        # 每轮固定生成 batch_per_round 个候选, 靠去重累积到目标 n。
        # 之前按 (n - len(collected)) 递减, 后面几轮候选太少,
        # 反馈闭环的"对照样本"不够, 第 2 轮只出了 5 个有效分子。
        n_gen = args.batch_per_round

        # 构造本轮 prompt
        if rnd == 0:
            # 第一轮: 纯领域 prompt (与训练完全一致)
            instr = build_prompt(prompts, args.target, CELL_STATES[rnd % len(CELL_STATES)],
                                 METRIC_VARIANTS[0][0], use_cot=(rnd % 2 == 0))
            variant_hint = ""
        else:
            # 后续轮: 注入上一轮的真实反馈
            prev = sorted(history[-1], key=lambda x: -x["reward"])
            good = prev[0] if prev else None
            bad = prev[-1] if prev else None
            base_instr = build_prompt(prompts, args.target,
                                      CELL_STATES[rnd % len(CELL_STATES)],
                                      METRIC_VARIANTS[0][0], use_cot=(rnd % 2 == 0))
            if good and bad:
                instr = feedback_refinement_instruction(
                    base_instr,
                    good_smiles=good["smiles"],
                    good_feedback=good["feedback"],
                    bad_smiles=bad["smiles"],
                    bad_feedback=bad["feedback"],
                    target=args.target,
                )
                print(f"  正例: {good['smiles'][:60]} (reward={good['reward']:.3f})")
                print(f"  反例: {bad['smiles'][:60]} (reward={bad['reward']:.3f})")
            else:
                instr = base_instr

        t0 = time.time()
        cands = gen_batch(instr, n_gen, seed_base=7000 + rnd * 1000)
        print(f"  生成 {len(cands)} 个候选 ({time.time()-t0:.0f}s)")

        # 去重
        new_cands = [s for s in cands if s not in seen_smiles]
        for s in new_cands:
            seen_smiles.add(s)

        # 评估
        # 逐个分子评估并**立即落盘** —— 之前是整批评估完才写,
        # 中途被中断就丢掉整批 (2026-11-19 第 4 轮就是这么丢的)。
        t0 = time.time()
        round_records = []
        for k, s in enumerate(new_cands):
            try:
                r = evaluate_one(s)
            except Exception as e:
                print(f"  ! 评估失败 {s[:40]}: {e}")
                continue
            if r is None:
                continue
            fb = build_feedback_text(r)
            rec = {
                "smiles": s,
                "reward": r.reward,
                "vina": r.vina_score,
                "dili": r.dili_prob,
                "herg": r.herg_prob,
                "bbb": r.bbb_score,
                "pgp": r.pgp_prob,
                "ames": r.ames_prob,
                "qed": r.qed,
                "pruned": r.is_pruned,
                "prune_reason": r.prune_reason,
                "round": rnd + 1,
                "feedback": fb,
            }
            round_records.append(rec)
            collected.append(rec)
            # 每个分子都落盘一次 —— 保证任何时刻中断都不丢结果
            _dump(out_dir, args.target, collected, history + [round_records])

        print(f"  评估 {len(round_records)} 个 ({time.time()-t0:.0f}s)")

        history.append(round_records)

        # 本轮统计
        if round_records:
            vs = [x["vina"] for x in round_records if x["vina"] is not None]
            rws = [x["reward"] for x in round_records]
            print(f"  本轮: Vina mean={sum(vs)/len(vs):.2f}  reward mean={sum(rws)/len(rws):.3f}  "
                  f"best reward={max(rws):.3f}")

        # 增量落盘
        _dump(out_dir, args.target, collected, history)

    # ---- 最终输出 ----
    _dump(out_dir, args.target, collected, history)
    _write_csv(out_dir / f"{args.target}.csv", collected, args.target)
    print(f"\n{'='*70}")
    print(f"完成: {len(collected)} 个分子 -> {out_dir}")
    print(f"{'='*70}")


def _dump(out_dir: Path, target: str, collected: List[dict], history: List[list]):
    """原子落盘。

    写临时文件再 os.replace —— 直接 write_text 在中断时会留下半截 JSON,
    下次 --resume 读到损坏文件就只能从头开始。
    """
    _atomic_write(out_dir / f"{target}_all.json",
                  json.dumps(collected, ensure_ascii=False, indent=2))
    _atomic_write(out_dir / f"{target}_history.json",
                  json.dumps(history, ensure_ascii=False, indent=2))


def _atomic_write(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, collected: List[dict], target: str):
    """原子写 CSV。"""
    # 去重, 按 reward 排序
    best = {}
    for c in collected:
        s = c["smiles"]
        if s not in best or c["reward"] > best[s]["reward"]:
            best[s] = c
    rows = sorted(best.values(), key=lambda x: -x["reward"])
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["entity_id", "mol_id", "library_name", "target", "smiles",
                    "reward", "vina_score", "DILI_prob", "hERG_prob", "BBB_score",
                    "Pgp_prob", "AMES_prob", "QED", "round", "is_pruned"])
        for i, c in enumerate(rows, 1):
            w.writerow([target,
                        f"GBM-PharmaMind-Feedback_{target}_{i}",
                        "GBM-PharmaMind (Ours, feedback loop)",
                        target, c["smiles"],
                        f"{c['reward']:.4f}", c["vina"], c["dili"], c["herg"],
                        c["bbb"], c["pgp"], c["ames"], c["qed"], c["round"],
                        c["pruned"]])
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
