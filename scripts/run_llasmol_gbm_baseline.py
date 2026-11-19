#!/usr/bin/env python3
"""
LlaSMol-Mistral-7B GBM Baseline Experiment
==========================================
使用 Hugging Face osunlp/LlaSMol-Mistral-7B 模型为 GBM 靶点生成候选分子。
LoRA adapter 加载到 mistralai/Mistral-7B-v0.1，4bit 量化以节省显存。

对比实验策略：极简 prompt，无复杂约束，保证 baseline 公平性。
"""

import sys
import os
import re
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def build_prompts(num_per_target: int) -> Dict[str, str]:
    """根据 num_per_target 动态生成极简 prompt。"""
    return {
        "EGFR": (
            f"Generate {num_per_target} candidate drug molecules targeting EGFR "
            f"for glioblastoma. Output valid SMILES only, one per line."
        ),
        "IDH1": (
            f"Generate {num_per_target} candidate drug molecules targeting IDH1 "
            f"for glioblastoma. Output valid SMILES only, one per line."
        ),
        "VEGF_VEGFR": (
            f"Generate {num_per_target} candidate drug molecules targeting VEGF/VEGFR "
            f"for glioblastoma. Output valid SMILES only, one per line."
        ),
    }


def _set_hf_mirror():
    """设置 HF 镜像站（如果默认源不可用）。"""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_HUB_OFFLINE"] = "0"


def _canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize a SMILES string via RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        Chem.SanitizeMol(mol, catchErrors=True)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _is_valid_for_baseline(smiles: str) -> bool:
    """Minimal validity check for baseline - just basic SMILES sanity."""
    if len(smiles) < 5 or len(smiles) > 600:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return False
        Chem.SanitizeMol(mol, catchErrors=True)
        return True
    except Exception:
        return False


def extract_smiles(text: str) -> List[str]:
    """
    从模型原始输出中提取 SMILES。
    策略：
      1. 按行/逗号/分号切分
      2. 去掉行首编号 (1., 2), -, SMILES:)
      3. 去掉解释性文字
      4. RDKit 验证 + canonicalize
      5. 去重
    """
    separators = re.compile(r"[,;\n]")
    results = []
    seen = set()

    for segment in separators.split(text):
        segment = segment.strip()
        segment = re.sub(r"^[\d]+\.[\s]*", "", segment)
        segment = re.sub(r"^[\d]+\)[\s]*", "", segment)
        segment = re.sub(r"^-\s*", "", segment)
        segment = re.sub(r"^SMILES\s*[:\-]?\s*", "", segment, flags=re.IGNORECASE)
        segment = re.sub(r"^[\"']|[\"']$", "", segment)
        segment = segment.strip()

        if not segment:
            continue
        if len(segment) < 3 or len(segment) > 600:
            continue

        invalid_words = [
            "generate", "targeting", "glioblastoma", "candidate",
            "molecule", "drug", "valid", "output", "only",
            "one per line", "please", "here is", "below",
            "smiles", "format", "example", "note",
        ]
        if any(w in segment.lower() for w in invalid_words):
            continue

        if not re.match(r"^[A-Za-z0-9@+\-\[\]\(\)=#%/\.\\]+$", segment):
            continue

        canonical = _canonicalize_smiles(segment)
        if canonical and canonical not in seen:
            seen.add(canonical)
            results.append(canonical)

    return results


class LlaSMolGenerator:
    """LlaSMol-Mistral-7B 分子生成器。"""

    BASE_MODEL = "mistralai/Mistral-7B-v0.1"
    ADAPTER = "osunlp/LlaSMol-Mistral-7B"

    def __init__(
        self,
        device: str = "cuda",
        use_4bit: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
    ):
        self.device = device
        self.use_4bit = use_4bit
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self):
        """加载 base model + LoRA adapter。"""
        _set_hf_mirror()
        from peft import PeftModel, PeftConfig
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            BitsAndBytesConfig,
        )

        logger.info(f"Loading tokenizer from {self.BASE_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.BASE_MODEL,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loading base model: {self.BASE_MODEL} (4bit={self.use_4bit})")
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                self.BASE_MODEL,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map=self.device,
                torch_dtype=torch.float16,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.BASE_MODEL,
                trust_remote_code=True,
                device_map=self.device,
                torch_dtype=torch.float16,
            )

        logger.info(f"Loading LlaSMol LoRA adapter: {self.ADAPTER}")
        self.model = PeftModel.from_pretrained(
            base_model,
            self.ADAPTER,
            torch_dtype=torch.float16,
        )
        self.model.eval()
        logger.info("LlaSMol model loaded successfully")
        self._loaded = True

    def generate(self, prompt: str) -> List[str]:
        """给定 prompt，返回提取后的 SMILES 列表。"""
        if not self._loaded:
            self.load()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)

        return extract_smiles(response)


def run_llasmol_experiment(
    targets: List[str],
    num_per_target: int = 50,
    max_attempts: int = 30,
    output_path: str = "outputs/llasmol_gbm_generated.csv",
    use_4bit: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 512,
    device: str = "cuda",
):
    """运行 LlaSMol 实验主流程。"""
    _set_hf_mirror()
    prompts = build_prompts(num_per_target)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    output_dir = Path(output_path).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = output_dir / f"llasmol_mistral_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    exp_cfg = {
        "base_model": LlaSMolGenerator.BASE_MODEL,
        "adapter": LlaSMolGenerator.ADAPTER,
        "targets": targets,
        "num_per_target": num_per_target,
        "max_attempts": max_attempts,
        "use_4bit": use_4bit,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "timestamp": datetime.now().isoformat(),
    }
    with open(exp_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(exp_cfg, f, indent=2)

    generator = LlaSMolGenerator(
        device=device,
        use_4bit=use_4bit,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )

    all_records = []

    for target in targets:
        prompt_text = prompts.get(target)
        if not prompt_text:
            logger.warning(f"No prompt for {target}, skipping")
            continue

        logger.info("=" * 80)
        logger.info(f"[LlaSMol] Processing target: {target}")
        logger.info("=" * 80)

        target_dir = exp_dir / target
        target_dir.mkdir(exist_ok=True)

        all_smiles = []
        seen = set()
        attempt = 0
        no_new_streak = 0

        while len(all_smiles) < num_per_target and attempt < max_attempts:
            attempt += 1
            try:
                smiles_list = generator.generate(prompt_text)
            except Exception as e:
                logger.error(f"  Attempt {attempt} error: {e}")
                continue

            new_count = 0
            for sm in smiles_list:
                if sm not in seen:
                    seen.add(sm)
                    all_smiles.append(sm)
                    new_count += 1

            logger.info(
                f"  Attempt {attempt}: {len(smiles_list)} extracted, "
                f"{new_count} new, total {len(all_smiles)}/{num_per_target}"
            )

            if new_count > 0:
                no_new_streak = 0
            else:
                no_new_streak += 1
                if no_new_streak >= 10:
                    logger.warning(
                        f"  No new molecules for 10 consecutive attempts, stopping {target}"
                    )
                    break

        logger.info(
            f"[{target}] Done: {len(all_smiles)} molecules from {attempt} attempts"
        )

        for i, sm in enumerate(all_smiles):
            try:
                mol = Chem.MolFromSmiles(sm)
                if mol:
                    tpsa = round(Descriptors.TPSA(mol), 2)
                    mw = round(Descriptors.MolWt(mol), 2)
                    logp = round(Descriptors.MolLogP(mol), 2)
                else:
                    tpsa = mw = logp = None
            except Exception:
                tpsa = mw = logp = None

            all_records.append({
                "id": i + 1,
                "smiles": sm,
                "target": target,
                "tpsa": tpsa,
                "mw": mw,
                "logp": logp,
                "generation_method": "LlaSMol-Mistral-7B",
            })

        csv_path = target_dir / f"{target}.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("id,smiles,target,tpsa,mw,logp,generation_method\n")
            for r in all_records:
                if r["target"] == target:
                    f.write(
                        f"{r['id']},{r['smiles']},{r['target']},"
                        f"{r['tpsa']},{r['mw']},{r['logp']},{r['generation_method']}\n"
                    )
        logger.info(f"  Saved: {csv_path} ({len(all_smiles)} molecules)")

    if all_records:
        import csv as csvlib
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csvlib.DictWriter(
                f, fieldnames=["id", "smiles", "target", "tpsa", "mw", "logp", "generation_method"]
            )
            writer.writeheader()
            writer.writerows(all_records)
        logger.info(f"Combined CSV saved: {output_path} ({len(all_records)} total molecules)")

    logger.info(f"Experiment output: {exp_dir}")
    return exp_dir, all_records


def main():
    parser = argparse.ArgumentParser(
        description="LlaSMol-Mistral-7B GBM Baseline Experiment"
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["EGFR", "IDH1", "VEGF_VEGFR"],
        help="Target proteins (default: EGFR IDH1 VEGF_VEGFR)",
    )
    parser.add_argument(
        "--num_per_target",
        type=int,
        default=50,
        help="Number of molecules per target (default: 50)",
    )
    parser.add_argument(
        "--output",
        default="outputs/llasmol_gbm_generated.csv",
        help="Output CSV path (default: outputs/llasmol_gbm_generated.csv)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top_p (default: 0.9)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max new tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--use_4bit",
        type=bool,
        default=True,
        help="Use 4bit quantization (default: True)",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=30,
        help="Max generation attempts per target (default: 30)",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU device ID (default: 0)",
    )
    args = parser.parse_args()

    _set_hf_mirror()
    device = f"cuda:{args.gpu_id}"

    run_llasmol_experiment(
        targets=args.targets,
        num_per_target=args.num_per_target,
        max_attempts=args.max_attempts,
        output_path=args.output,
        use_4bit=args.use_4bit,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )


if __name__ == "__main__":
    main()
