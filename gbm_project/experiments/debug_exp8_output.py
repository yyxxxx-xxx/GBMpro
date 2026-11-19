#!/usr/bin/env python3
"""Quick debug: see what the model actually outputs."""
import sys, re
from pathlib import Path
import torch
import warnings; warnings.filterwarnings("ignore")

PROJECT_ROOT = Path('/root/Llamole-main')
sys.path.insert(0, str(PROJECT_ROOT))

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

base_model = str(PROJECT_ROOT / "models" / "Qwen2-7B-Instruct")
adapter = str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter")

tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, padding_side="right")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

bm = AutoModelForCausalLM.from_pretrained(
    base_model, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda:3"
)
model = PeftModel.from_pretrained(bm, adapter, torch_dtype=torch.float16)
model.eval()

device = "cuda:3"

prompt = (
    "You are a medicinal chemist. Generate exactly 5 valid SMILES strings for drug molecules "
    "targeting EGFR (Epidermal Growth Factor Receptor) for Glioblastoma (GBM) treatment.\n\n"
    "Requirements:\n"
    "- Molecular Weight: 300-500 Da\n"
    "- Blood-Brain Barrier penetration: HIGH\n"
    "- LogP: 2.0-4.0\n"
    "- Synthesizable\n\n"
    "Output format:\n"
    "SMILES 1: <smiles>\n"
    "SMILES 2: <smiles>\n"
    "SMILES 3: <smiles>\n"
    "SMILES 4: <smiles>\n"
    "SMILES 5: <smiles>\n\n"
    "Only output the numbered SMILES list above, nothing else."
)

msgs = [{"role": "user", "content": prompt}]
formatted = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print("=== FORMATTED INPUT (last 500 chars) ===")
print(repr(formatted[-500:]))
print("==========================================")

inputs = tok(formatted, return_tensors="pt").to(device)
inp_len = inputs["input_ids"].shape[1]

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=600,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

resp = tok.decode(out[0][inp_len:], skip_special_tokens=True)
print("\n=== RAW OUTPUT ===")
print(resp)
print("==================")

# Test extraction
seen = set()
results = []
for m in re.findall(r'SMILES\s*\d*\s*:\s*([^\n]+)', resp, re.IGNORECASE):
    smi = re.sub(r'\s+', '', m.strip().strip(' ,.;:"\'`'))
    print(f"  Strategy1 match: {repr(smi)}")
    if smi and smi not in seen:
        seen.add(smi); results.append(smi)

if not results:
    print("\n  No matches from numbered format. Trying raw lines...")
    for line in resp.strip().split('\n'):
        line = line.strip()
        line = re.sub(r'^[\d]+\.\s*', '', line)
        line = re.sub(r'^-\s*', '', line)
        smi = re.sub(r'\s+', '', line.strip().strip(' ,.;:"\'`'))
        print(f"  Raw line: {repr(smi[:80])}")
        if smi and smi not in seen:
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smi, sanitize=False)
                if mol:
                    seen.add(smi); results.append(smi)
                    print(f"    -> VALID!")
            except:
                pass

print(f"\nExtracted {len(results)} SMILES: {results}")
