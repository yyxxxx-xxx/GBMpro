#!/usr/bin/env python3
"""
GBM靶向药物分子生成器（简化版）
==============================
功能:
1. 直接使用PeftModel加载微调模型
2. 结合CoT思维链进行推理
3. 针对不同靶点分别生成5个独立分子
4. 使用处理后的靶点结构信息
5. 在GPU0上运行
6. 无需rdkit依赖

使用方式:
python generate_target_specific_molecules.py --target EGFR --num_molecules 5
"""

import os
import sys
import json
import torch
import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TargetSpecificMoleculeGenerator:
    """针对特定靶点的药物分子生成器（无评估版）"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = f"cuda:{config['gpu_id']}" if torch.cuda.is_available() else "cpu"
        
        # 路径配置
        self.base_model_path = config['base_model_path']
        self.llamole_adapter_path = config['llamole_adapter_path']
        self.gbm_adapter_path = config['gbm_adapter_path']
        
        # 加载模型和tokenizer
        self.model = None
        self.tokenizer = None
        
        # 靶点信息
        self.targets_info = self._load_targets_info()
        
        # 生成配置
        self.generation_config = config.get('generation', {
            'max_new_tokens': 1024,
            'temperature': 0.9,
            'top_p': 0.95,
            'do_sample': True
        })
        
        # COT配置
        self.cot_config = config.get('cot', {
            'enabled': True,
            'reasoning_steps': 7,
            'include_market_reference': True
        })
        
    def _load_targets_info(self) -> Dict[str, Any]:
        """加载处理后的靶点信息"""
        targets_info = {}
        processed_dir = PROJECT_ROOT / "gbm_project" / "data" / "processed_targets"
        
        if not processed_dir.exists():
            logger.warning(f"Processed targets directory not found: {processed_dir}")
            return targets_info
        
        for summary_file in processed_dir.glob("*_summary.json"):
            target_name = summary_file.stem.replace("_summary", "")
            with open(summary_file, 'r') as f:
                targets_info[target_name] = json.load(f)
        
        logger.info(f"Loaded {len(targets_info)} target information files")
        return targets_info
    
    def load_models(self):
        """直接使用PeftModel加载模型"""
        logger.info("Loading models with PeftModel...")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            
            # 加载tokenizer
            logger.info(f"Loading tokenizer from {self.base_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True,
                padding_side="right"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载基础模型
            logger.info(f"Loading base model from {self.base_model_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                device_map={"": self.device},
                trust_remote_code=True
            )
            
            # 加载Llamole适配器
            if self.llamole_adapter_path and os.path.exists(self.llamole_adapter_path):
                logger.info(f"Loading Llamole adapter from {self.llamole_adapter_path}")
                base_model = PeftModel.from_pretrained(base_model, self.llamole_adapter_path)
            
            # 加载GBM适配器
            if self.gbm_adapter_path and os.path.exists(self.gbm_adapter_path):
                logger.info(f"Loading GBM adapter from {self.gbm_adapter_path}")
                self.model = PeftModel.from_pretrained(base_model, self.gbm_adapter_path)
            else:
                self.model = base_model
            
            logger.info(f"Model loaded: {self.model.__class__.__name__}")
            logger.info("✓ All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def generate_prompt(self, target_name: str, target_info: Dict) -> str:
        """生成带有CoT推理的提示词"""
        
        # 获取靶点结构信息
        seq_info = target_info.get('sequence', {})
        binding_sites = target_info.get('binding_sites', [])
        
        # 构建靶点描述
        target_desc = self._get_target_description(target_name)
        
        # 构建CoT推理模板
        if self.cot_config['enabled']:
            cot_prompt = self._build_cot_prompt(
                target_name, target_desc, seq_info, binding_sites
            )
        else:
            cot_prompt = self._build_simple_prompt(target_name, target_desc)
        
        return cot_prompt
    
    def _get_target_description(self, target_name: str) -> str:
        """获取靶点描述"""
        descriptions = {
            'EGFR': """
EGFR (Epidermal Growth Factor Receptor) in GBM:
- Most common gene amplification (~40% of GBM cases)
- Drives excessive tumor cell proliferation
- Key mutation: EGFRvIII with immunogenicity
- Resistance mechanisms: T790M, MET amplification
Binding site: Kinase domain ATP pocket
Key residues: Lys745 (H-bond), Asp855 (catalytic), Met769 (hydrophobic)
            """,
            
            'VEGFR': """
VEGF/VEGFR pathway in GBM:
- Drives pathological angiogenesis
- Creates immunosuppressive microenvironment
- Promotes tumor growth and invasion
Binding site: VEGFR kinase domain
            """,
            
            'IDH1': """
IDH1/IDH2 mutations in GBM:
- Mainly in secondary GBM (~10%)
- Leads to 2-HG accumulation
- Affects epigenetic regulation
Binding site: IDH active site with NADP+ binding pocket
            """,
            
            'MGMT': """
MGMT in GBM:
- Repairs TMZ-induced DNA damage
- MGMT expression affects TMZ response
- Epigenetic silencing can reverse resistance
Target: MGMT repair mechanism bypass
            """,
            
            'PD1_PDL1': """
PD-1/PD-L1 pathway in GBM:
- Key regulator of immune microenvironment
- Immunosuppressive microenvironment
- Target for immunotherapy combinations
            """,
            
            'PI3K_AKT_mTOR': """
PI3K/AKT/mTOR pathway in GBM:
- Most commonly activated pathway (~80%)
- Mutations: PTEN loss, PIK3CA mutation, AKT amplification
            """,
            
            'TP53_MDM2': """
p53/MDM2 pathway in GBM:
- TP53 mutation rate ~30%
- MDM2 amplification ~10%
- Regulates cell apoptosis
            """,
            
            'MDM2': """
MDM2 in GBM:
- E3 ubiquitin ligase for p53
- Amplification in ~10% of GBM
- Target for p53 pathway restoration
            """
        }
        
        return descriptions.get(target_name, f"Target: {target_name}")
    
    def _build_cot_prompt(self, target_name: str, target_desc: str, 
                          seq_info: Dict, binding_sites: List) -> str:
        """构建CoT思维链推理提示"""
        
        seq_length = seq_info.get('length', 'N/A')
        
        # 结合位点描述
        sites_desc = ""
        for i, site in enumerate(binding_sites[:3], 1):
            sites_desc += f"""
Site {i}: {site.get('site_name', 'Unknown')}
- Key residues: {', '.join(site.get('key_residues', [])[:5])}
            """
        
        prompt = f"""<design_start>
Chain-of-Thought Reasoning for {target_name} Inhibitor Design

## Analysis Phase

### Target Analysis
{target_desc}

Protein sequence length: {seq_length} amino acids

Predicted binding sites:{sites_desc}

### Market Reference Analysis
Successful {target_name} inhibitors share these properties:
- Molecular weight: 350-500 Da (optimal BBB penetration)
- LogP: 2.0-4.0 (balance hydrophobicity)
- Core scaffolds: heterocyclic aromatic systems
- Key interactions: H-bond with key residues, hydrophobic contacts

### Molecular Design Strategy
Based on target analysis:
1. Select BBB-permeable core scaffold (MW 150-250 Da)
2. Add functional groups matching pharmacophore profile
3. Ensure H-bond capability with key binding site residues
4. Maintain Lipinski's rules compliance

### Safety Considerations
- AVOID aniline structures (risk of hepatotoxicity)
- Limit aromatic rings to ≤3
- LogP < 4.0 to prevent liver accumulation
- Include polar groups for metabolism

## Generation Phase

Generate 5 novel {target_name} inhibitor candidates following the analysis above.

First, provide your Chain-of-Thought reasoning for molecular design.
Then, output exactly 5 valid SMILES strings.

After your reasoning, end with:
SMILES1:<valid_smiles_1>
SMILES2:<valid_smiles_2>
SMILES3:<valid_smiles_3>
SMILES4:<valid_smiles_4>
SMILES5:<valid_smiles_5>
"""
        
        return prompt
    
    def _build_simple_prompt(self, target_name: str, target_desc: str) -> str:
        """构建简单提示（无CoT）"""
        
        prompt = f"""<design_start>
Design 5 novel drug candidates targeting {target_name} for GBM treatment.

{target_desc}

Requirements:
- Molecular weight: 350-500 Da
- LogP: 2.0-4.0
- High BBB penetration
- Minimize hepatotoxicity (avoid aniline structures)
- Synthetically accessible

Output format (each on separate line):
SMILES: <molecule_smiles>
<design_end>
"""
        
        return prompt
    
    def generate_molecules(self, target_name: str, num_molecules: int = 5) -> List[Dict]:
        """针对特定靶点生成分子"""
        
        if target_name not in self.targets_info:
            logger.error(f"Target {target_name} not found in processed targets")
            return []
        
        target_info = self.targets_info[target_name]
        
        # 生成提示词
        prompt = self.generate_prompt(target_name, target_info)
        
        logger.info(f"Generating {num_molecules} molecules for {target_name}")
        logger.info(f"Prompt length: {len(prompt)} characters")
        
        # 尝试多次生成，直到获得5个有效SMILES或达到最大尝试次数
        all_smiles = []
        max_attempts = 3
        target_count = num_molecules
        
        for attempt in range(max_attempts):
            if len(all_smiles) >= target_count:
                break
                
            logger.info(f"  Attempt {attempt+1}/{max_attempts}...")
            
            try:
                response = self._generate_single_molecule(prompt)
                smiles_list = self._extract_smiles(response)
                
                # 记录这次生成的SMILES
                for smiles in smiles_list:
                    if smiles not in all_smiles:
                        all_smiles.append(smiles)
                        logger.info(f"    ✓ Found valid SMILES ({len(all_smiles)}/{target_count}): {smiles[:50]}...")
                
                if len(all_smiles) >= target_count:
                    break
                    
            except Exception as e:
                logger.error(f"    ✗ Error: {e}")
        
        # 只保留有效的SMILES
        molecules = []
        for i, smiles in enumerate(all_smiles[:target_count], 1):
            molecules.append({
                'smiles': smiles,
                'raw_response': "",  # 不保存完整原始响应
                'target': target_name,
                'generation_id': i
            })
        
        logger.info(f"  Generated {len(molecules)} valid molecules for {target_name}")
        return molecules
    
    def _generate_single_molecule(self, prompt: str) -> str:
        """生成单个分子"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.generation_config.get('max_new_tokens', 1024),
                temperature=self.generation_config.get('temperature', 0.9),
                top_p=self.generation_config.get('top_p', 0.95),
                do_sample=self.generation_config.get('do_sample', True),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取Response部分
        if "Response:" in response:
            response = response.split("Response:")[-1].strip()
        elif "<design_end>" in response:
            response = response.split("<design_end>")[0].strip()
        
        return response
    
    def _extract_smiles(self, response: str) -> List[str]:
        """从响应中提取多个SMILES，只保留有效的"""
        smiles_list = []
        
        # 查找所有SMILES行
        pattern = r'SMILES\d*:\s*([A-Za-z0-9@+\-\[\]\(\)=#%/\s]+)'
        matches = re.findall(pattern, response)
        
        for match in matches:
            smiles = match.strip()
            if self._basic_validate_smiles(smiles):
                smiles_list.append(smiles)
        
        # 去重
        smiles_list = list(dict.fromkeys(smiles_list))
        
        return smiles_list
    
    def _basic_validate_smiles(self, smiles: str) -> bool:
        """基础SMILES验证（不依赖rdkit）- 过滤无效值"""
        # 过滤占位符
        invalid_values = ['Chain-of-Thought', 'valid_smiles', 'smiles_string', '<valid_smiles>',
                         'analysis', 'reasoning', 'design', 'output', 'format']
        if smiles.lower() in [v.lower() for v in invalid_values]:
            return False
        
        if len(smiles) < 5 or len(smiles) > 100:
            return False
        
        # 检查平衡的括号
        open_brackets = sum(1 for c in smiles if c in '([{')
        close_brackets = sum(1 for c in smiles if c in ')]}')
        if open_brackets != close_brackets:
            return False
        
        # 至少包含一个环原子符号
        if not any(c in smiles for c in 'CBNOSPFI'):
            return False
        
        # 必须包含小写原子（有效SMILES特征）
        if not any(c in smiles for c in 'cnops'):
            return False
        
        # 排除明显无效的字符
        invalid_chars = set('!@$%^&*=|\\{}:;"<>,?`~')
        if any(c in invalid_chars for c in smiles):
            return False
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Generate target-specific GBM drug molecules")
    parser.add_argument('--target', type=str, default='EGFR', help='Target name')
    parser.add_argument('--num_molecules', type=int, default=5, help='Number of molecules to generate')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device ID')
    parser.add_argument('--use_cot', action='store_true', default=True, help='Use CoT reasoning')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    # 配置
    config = {
        'gpu_id': args.gpu_id,
        'base_model_path': str(PROJECT_ROOT / "models" / "Qwen2-7B-Instruct"),
        'llamole_adapter_path': str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter"),
        'gbm_adapter_path': str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter-gbm-fixed"),
        'generation': {
            'max_new_tokens': 1024,
            'temperature': 0.9,
            'top_p': 0.95,
            'do_sample': True
        },
        'cot': {
            'enabled': args.use_cot,
            'reasoning_steps': 7,
            'include_market_reference': True
        }
    }
    
    if args.output_dir:
        config['output_dir'] = args.output_dir
    else:
        config['output_dir'] = str(PROJECT_ROOT / "gbm_project" / "experiments" / 
                                   f"target_specific_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # 初始化生成器
    generator = TargetSpecificMoleculeGenerator(config)
    
    # 加载模型
    generator.load_models()
    
    # 生成分子
    molecules = generator.generate_molecules(args.target, args.num_molecules)
    
    # 保存结果
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'config': config,
        'target': args.target,
        'num_generated': len(molecules),
        'molecules': molecules,
        'timestamp': datetime.now().isoformat()
    }
    
    output_file = output_dir / "generation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    
    # 打印生成的分子（简洁格式）
    print("\n" + "=" * 60)
    print(f"Generation Complete for {args.target}")
    print("=" * 60)
    print(f"Target: {args.target}")
    print(f"Molecules generated: {len(molecules)}")
    print(f"CoT enabled: {args.use_cot}")
    print(f"Output: {output_file}")
    print("=" * 60)
    print("Generated SMILES:")
    for i, mol in enumerate(molecules, 1):
        print(f"  {i}. {mol['smiles']}")
    print("=" * 60)
    
    return molecules


if __name__ == "__main__":
    main()
