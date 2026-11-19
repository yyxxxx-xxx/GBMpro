#!/usr/bin/env python3
"""补充生成 VEGF_VEGFR 20个 + IDH1_IDH2 10个"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys, re, json, csv
from pathlib import Path
from datetime import datetime
from typing import List, Set
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
import warnings; warnings.filterwarnings('ignore')
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = Path('/root/Llamole-main')
sys.path.insert(0, str(PROJECT_ROOT))

def _rdkit_ok(s):
    try:
        mol = Chem.MolFromSmiles(s, sanitize=False)
        if mol is None: return False
        Chem.SanitizeMol(mol, catchErrors=True)
        return True
    except: return False

def extract_smiles(text):
    results, seen = [], set()
    for chunk in re.split(r'[,;\n]', text):
        chunk = re.sub(r'^[\-\u2022\u2023\u25e6\*\u2043]\s*', '', chunk.strip())
        chunk = re.sub(r'^[\d]+[\.\)\:]+\s*', '', chunk)
        chunk = re.sub(r'^[Ss][Mm][Ii][Ll][Ee][Ss]\s*[:\-]?\s*', '', chunk)
        chunk = re.sub(r'[\`\`\`]', '', chunk).strip()
        if chunk and len(chunk) >= 5 and re.search(r'[CNOSPFIcnosp]', chunk) and chunk not in seen and _rdkit_ok(chunk):
            seen.add(chunk); results.append(chunk)
    return results

base_dir = PROJECT_ROOT / 'gbm_project' / 'experiments' / 'exp9_llasmol_simple_20260507_140225'
existing = {'VEGF_VEGFR': set(), 'IDH1_IDH2': set()}
for f in (base_dir / 'csv_output').glob('*.csv'):
    t = f.stem
    if t not in existing: continue
    with open(f) as fh:
        for row in csv.reader(fh):
            if len(row) >= 4: smi = row[3].strip()
            elif len(row) >= 2: smi = row[1].strip()
            else: continue
            if smi and _rdkit_ok(smi): existing[t].add(smi)

print('Existing:', {t: len(s) for t, s in existing.items()})

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

print('Loading model...')
tok = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1', trust_remote_code=True, padding_side='left')
if tok.pad_token is None: tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4')
bm = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1', trust_remote_code=True, quantization_config=bnb, device_map='auto', torch_dtype=torch.float16)
m = PeftModel.from_pretrained(bm, 'osunlp/LlaSMol-Mistral-7B', torch_dtype=torch.float16)
m.eval()
print('Model ready!')

PROMPTS = {
    'IDH1_IDH2': 'Generate 3 valid SMILES for molecules targeting IDH1 or IDH2 for glioblastoma. Output SMILES only, one per line.',
    'VEGF_VEGFR': 'Generate 3 valid SMILES for molecules targeting VEGF or VEGFR for glioblastoma. Output SMILES only, one per line.',
}
NEEDED = {'IDH1_IDH2': 10, 'VEGF_VEGFR': 20}
START_IDS = {'IDH1_IDH2': 36, 'VEGF_VEGFR': 20}
csv_dir = base_dir / 'csv_output'

for target, need in NEEDED.items():
    print(f'\n--- {target} (need {need}) ---')
    prompt = PROMPTS[target]
    seen = set(); new_mols = []; dup_mols = []; total = 0; calls = 0

    while len(new_mols) < need and calls < 20:
        calls += 1
        try:
            inputs = tok(prompt, return_tensors='pt')
            inputs = {k: v.to('cuda:3') for k, v in inputs.items()}
            with torch.no_grad():
                out = m.generate(**inputs, max_new_tokens=300, temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=tok.pad_token_id)
            resp = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            smis = extract_smiles(resp)
            total += len(smis)
            for s in smis:
                if s in existing[target]:
                    if s not in dup_mols: dup_mols.append(s)
                else:
                    if s not in seen: seen.add(s); new_mols.append(s)
            nov = len(new_mols)/total*100 if total else 0
            print(f'  Call {calls}: extracted={len(smis)}, new={len(new_mols)}, dup={len(dup_mols)}, total_new={len(new_mols)}/{need}, novelty={nov:.0f}%')
        except Exception as e:
            print(f'  Call {calls} ERROR: {e}')
            continue

    csv_path = csv_dir / f'{target}.csv'
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        for i, smi in enumerate(new_mols):
            try:
                mol = Chem.MolFromSmiles(smi)
                mw = Descriptors.MolWt(mol); tp = Descriptors.TPSA(mol); lp = Descriptors.MlogP(mol)
            except: mw = tp = lp = 0.0
            w.writerow([f'llasmol_supplement', f'{target}_{START_IDS[target]+i+1:05d}', 'LlaSMol-Mistral-7B', smi, f'{mw:.2f}', f'{tp:.2f}', f'{lp:.2f}', target])
    print(f'  Appended {len(new_mols)} molecules to {csv_path}')

print('\nDone!')
