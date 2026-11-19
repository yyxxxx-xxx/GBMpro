# GBM Drug Design System — Architecture Documentation

> This document provides comprehensive architectural details for generating high-quality scientific diagrams of the GBM (Glioblastoma Multiforme) AI-driven drug design system using Tree-of-Thoughts reasoning.

---

## 1. System Overview

### 1.1 Purpose

The GBM Drug Design System is an AI-powered pipeline that generates candidate drug molecules for Glioblastoma Multiforme treatment. It combines domain knowledge from a curated knowledge base, fine-tuned large language models (Qwen2-7B-Instruct), Tree-of-Thoughts (ToT) reasoning strategy, and physics-based molecular evaluation (AutoDock Vina, ADMET prediction).

### 1.2 Target Diseases

The system addresses **7 key GBM therapeutic targets**:

| Target | Description | Prevalence |
|--------|-------------|------------|
| **EGFR** | Epidermal Growth Factor Receptor | ~40% of GBM |
| **VEGF/VEGFR** | Vascular Endothelial Growth Factor | Angiogenesis driver |
| **IDH1/IDH2** | Isocitrate Dehydrogenase mutations | ~10% of GBM |
| **MGMT** | O6-Methylguanine-DNA Methyltransferase | TMZ resistance |
| **PD-1/PD-L1** | Programmed Death Receptor/Ligand | Immune checkpoint |
| **PI3K/AKT/mTOR** | PI3K/AKT/mTOR pathway | ~80% activated |
| **p53/MDM2** | TP53 tumor suppressor / MDM2 | ~30% of GBM |

### 1.3 Molecular Design Constraints

```
Molecular Weight (MW):  300–600 Da
LogP:                    1.0–4.0
TPSA:                   40–120 Å²
HBD (H-bond donors):     1–5
HBA (H-bond acceptors):  3–10
BBB Permeability:        HIGH (critical for brain tumors)
```

---

## 2. Data Layer

### 2.1 Knowledge Data Files

```
gbm_project/data/
├── gbm_targets/
│   └── gbm_targets.json          # 7 GBM targets with mutations, drugs, challenges
├── gbm_clinical/
│   └── gbm_clinical_data.json   # Treatment protocols, resistance mechanisms, subtypes
├── gbm_molecules/
│   └── gbm_molecules.json       # Reference drugs, clinical candidates, design principles
└── lora_datasets/              # Training datasets for LoRA fine-tuning
```

### 2.2 Target Data Schema (`gbm_targets.json`)

```json
{
  "gbm_targets": [{
    "name": "EGFR",
    "description": "Epidermal Growth Factor Receptor...",
    "mutation_types": [
      "EGFRvIII (exon 2-7 deletion)",
      "EGFR amplification"
    ],
    "current_drugs": ["Erlotinib", "Gefitinib", "Afatinib", "Osimertinib"],
    "challenges": [
      "T790M resistance mutation",
      "Blood-brain barrier penetration",
      "Wild-type EGFR toxicity"
    ],
    "structural_requirements": {
      "molecular_weight": "<500 Da",
      "logP": "2–4",
      "BBB_permeability": "HIGH",
      "selectivity": "EGFRvIII over wild-type EGFR",
      "inhibition_type": "covalent or non-covalent kinase inhibitor"
    }
  }]
}
```

### 2.3 Clinical Data Schema (`gbm_clinical_data.json`)

```json
{
  "clinical_data": {
    "standard_treatment": {
      "Stupp protocol": "Surgery + Temozolomide + Radiation"
    },
    "treatment_challenges": [
      "TMZ resistance (up to 90%)",
      "BBB restricts drug entry",
      "Tumor heterogeneity",
      "Recurrence from stem cells"
    ],
    "molecular_subtypes": {
      "classical": { "marker": "EGFR amplification" },
      "mesenchymal": { "marker": "NF1, MET, CHI3L1" },
      "proneural": { "marker": "PDGFRA, IDH1, NKAIN2" },
      "neural": { "marker": "GABRA1, SLC26A3" }
    }
  }
}
```

### 2.4 Reference Molecules Schema (`gbm_molecules.json`)

```json
{
  "gbm_drugs": [{
    "name": "Temozolomide",
    "smiles": "Cn1cnc2N(C)C(=O)N(C)C(=O)c12",
    "target": "DNA alkylation",
    "status": "Approved",
    "clinical_phase": "Phase 4 (Standard of Care)",
    "mechanism": "Methylates DNA at O6-guanine, causing apoptosis"
  }],
  "structural_motifs": {
    "BBB_permeable_scaffolds": ["quinazoline", "pyridine", "pyrimidine", "indole"],
    "GBM_selective_groups": ["acrylamide (covalent)", "fluorine (lipophilicity)"],
    "toxicity_reduction_features": ["methylimidazole", "morpholine"]
  }
}
```

---

## 3. Configuration Layer

### 3.1 Config Files

```
gbm_project/configs/
├── english_gbm_prompts.yaml    # English prompt templates (MAIN)
├── gbm_prompts.yaml           # Chinese prompt templates
├── gbm_generation.yaml         # Generation hyperparameters
├── llamole_minimal_config.yaml
├── llamole_compatible_config.yaml
├── minimal_llamole_config.yaml
└── baseline_generation.yaml
```

### 3.2 Key Configurations (`gbm_generation.yaml`)

```yaml
model_name_or_path: models/Qwen2-7B-Instruct
adapter_name_or_path: saves/Llamole-Qwen2-7B-Instruct-Adapter-gbm-with-graph-models
graph_encoder_path: saves/graph_encoder_gbm
graph_decoder_path: saves/graph_decoder_gbm
graph_predictor_path: saves/graph_predictor_gbm

max_length: 1024
max_new_tokens: 256
temperature: 0.7
top_p: 0.9
do_sample: true

molecular_constraints:
  MW: [300, 600]       # Da
  LogP: [1, 4]
  TPSA: [60, 120]     # Å²
  HBD: [1, 5]
  HBA: [3, 10]

gbm_target_weights:
  EGFR: 0.20
  VEGF_VEGFR: 0.18
  IDH1_IDH2: 0.15
  MGMT: 0.12
  PD_1_PD_L1: 0.12
  PI3K_AKT_mTOR: 0.13
  p53_MDM2: 0.10

evaluation_metrics:
  - bbb_permeability
  - gbm_activity
  - toxicity
  - synthetic_accessibility
  - druglikeness
```

### 3.3 ToT-Specific Configurations

```yaml
tot_step_max_new_tokens:
  scaffold: 200      # Level 0: scaffold proposal output length
  assembly: 280      # Level 1: assembly strategy output length
  smiles: 450       # Level 2: SMILES generation output length
  evaluate: 60      # Physical evaluation report length
  vote: 100         # Batch evaluation vote output length

tot_max_llm_calls_per_search: 50     # Budget limit per ToT search
tot_max_search_seconds: 300          # Wall-clock time limit per search
tot_depth: 3                          # Scaffold → Assembly → SMILES
tot_k: 3                              # Candidates per level
tot_b: 2                              # Branches to keep (BFS)
tot_refinement_rounds: 1             # Feedback optimization rounds
tot_good_reward_threshold: 0.65      # Early stopping reward threshold
```

---

## 4. Source Code Layer

```
gbm_project/
├── src/                          # Core modules
│   ├── gbm_knowledge_base.py         # Domain knowledge management
│   ├── gbm_prompt_generator.py       # Prompt generation + ToT templates
│   ├── gbm_physical_evaluator.py     # Physics-based evaluation engine
│   ├── gbm_evaluator.py             # Internal RDKit-based evaluation
│   ├── gbm_generator.py             # Main orchestrator
│   ├── gbm_lora_finetuner.py         # LoRA fine-tuning
│   ├── gbm_lora_dataset.py          # Dataset preparation
│   ├── structural_knowledge_injector.py  # Structural guidance
│   ├── gbm_admet_predictor.py        # ADMET prediction
│   └── sascorer.py                  # Synthetic accessibility
├── scripts/                      # Generation scripts
│   ├── generate_tot_molecules.py         # ToT single-target runner
│   ├── batch_generate_all_targets.py     # Batch runner
│   ├── generate_target_specific_molecules.py
│   ├── evaluate_gbm_candidates.py         # Evaluation script
│   └── [ablation / experiment scripts]
├── experiments/                 # Batch experiments
│   ├── generate_tot_batch_candidates.py  # Batch ToT launcher
│   ├── evaluate_gbm_candidates.py
│   └── tot_batch_YYYYMMDD_HHMMSS/        # Experiment outputs
└── configs/                     # Configuration files
```

---

## 5. Knowledge Base Module

### 5.1 GBMKnowledgeBase (`src/gbm_knowledge_base.py`)

Manages all GBM domain knowledge, providing structured access to targets, clinical data, and molecular references.

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `get_target_info(target_name)` | Fetch target details, mutations, drugs, challenges |
| `get_random_target(weights)` | Weighted random target selection |
| `get_similar_molecules(target, limit)` | Get reference molecules for a target |
| `get_clinical_insights()` | Extract treatment insights |
| `get_design_principles()` | Get molecular design guidelines |
| `get_structural_motifs()` | Get structural patterns |
| `get_subtype_info(subtype)` | Get molecular subtype info |
| `get_resistance_mechanisms(target_type)` | Get resistance patterns |

### 5.2 Data Relationships

```
GBMTarget ──references──▶ GBMMolecule (similar drugs)
    │
    ├──mutation_types──▶ GBMClinicalData.resistance_mechanisms
    │
    └──structural_requirements ──▶ GBMPromptGenerator
                                      │
                                      ▼
                               GBMPromptGenerator ──injects──▶ ToT Prompts
```

---

## 6. Prompt Generation Module

### 6.1 GBMPromptGenerator (`src/gbm_prompt_generator.py`)

Generates domain-specific prompts with CoT (Chain-of-Thought) and ToT (Tree-of-Thoughts) support.

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `generate_domain_prompt(target_name, constraints)` | Create GBM domain context |
| `generate_cot_prompt(domain_prompt, target_name, reasoning_type)` | Add CoT reasoning |
| `generate_evaluation_prompt(molecule_description, evaluation_type)` | Create evaluation prompt |
| `generate_full_prompt(target_name, constraints, use_cot, reasoning_type)` | Complete prompt |
| `build_tot_propose_prompt(domain_prompt, current_state, step_type)` | ToT scaffold/assembly/smiles |
| `build_feedback_injected_prompt(...)` | Inject physical feedback |
| `build_tot_propose_prompt_with_feedback(...)` | ToT prompt + physical feedback |

### 6.2 ToT Prompt Templates (from `english_gbm_prompts.yaml`)

**Level 0 — Scaffold Proposal:**
```
You are an expert medicinal chemist designing drugs for GBM (Glioblastoma Multiforme).

[Domain Context from GBMKnowledgeBase]
Target: {target_name}
Challenges: {challenges}
Design Requirements: {structural_requirements}

Task: Propose {k} diverse scaffolds appropriate for this target.

[FORMAT — STRICT]
Scaffold 1:
Name: <scaffold_name>
Rationale: <one-sentence_rationale>
Base MW: <number> Da
BBB Potential: <high/medium/low>

Scaffold 2:
Name: ...
```

**Level 1 — Assembly Strategy:**
```
Selected Scaffold: {scaffold_name} (MW: {scaffold_mw} Da)
Remaining MW Budget: {remaining_mw} Da
Target MW: {target_mw} Da

Task: Propose assembly strategy with warhead and BBB enhancers.

[FORMAT — STRICT]
Strategy 1:
Warhead: <warhead_type>
BBB Enhancers: <enhancer_groups>
Estimated Total MW: <number> Da
Expected LogP: <range>
Expected TPSA: <range>
Rationale: <explanation>
```

**Level 2 — SMILES Generation:**
```
Scaffold: {scaffold_name}
Assembly Strategy: {warhead} + {bbb_enhancers}
Target MW: {target_mw} Da
Target LogP: {logp_range}
Target TPSA: {tpsa_range}

Task: Generate valid SMILES strings.

[FORMAT — STRICT]
SMILES 1: <smiles_string>
```

### 6.3 Physical Feedback Template

```
[PHYSICAL EVALUATION RESULTS]
Vina binding energy = {vina_score} kcal/mol ({strength})
DILI probability = {dili_prob} ({risk_level})
BBB permeability = {bbb_prob} ({penetration_level})
Overall Reward = {reward:.4f}

[FEEDBACK — Use this to guide your next design]
```

---

## 7. Model Layer

### 7.1 Model Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    Qwen2-7B-Instruct (Base)                   │
│                  7 Billion Parameters, 16-bit                    │
└────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Llamole Adapter   │
                    │  (LoRA, merged)    │
                    │  General chemistry │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────────────────────────────┐
                    │         GBM Adapter                         │
                    │  (Domain-specific LoRA, NOT merged)         │
                    │  Fine-tuned on GBM drug design corpus       │
                    └─────────┬─────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
  ┌──────▼──────┐   ┌────────▼────────┐   ┌──────▼──────┐
  │Graph Encoder │   │ Graph Decoder   │   │ Graph Pred.  │
  │  (Molecular  │   │  (Molecule      │   │  (Property   │
  │   Features)  │   │   Generation)   │   │   Prediction)│
  └──────────────┘   └─────────────────┘   └──────────────┘
```

### 7.2 Model Loading Pipeline (`load_models()`)

```
1. Load Qwen2-7B-Instruct tokenizer
   └── Add pad_token if missing

2. Load base model (FP16 or 8-bit quantized)
   └── Move to GPU

3. Load Llamole adapter (PeftModel)
   └── Merge into base model (merge_and_unload)
   └── Free Llamole adapter memory

4. Load GBM adapter (PeftModel, NOT merged)
   └── Keep as separate trainable adapter
   └── Save reference to base_merged_model (for SMILES fallback)
```

### 7.3 Two-Model SMILES Generation Strategy

When the fine-tuned model fails to produce valid SMILES:

```
Fine-tuned model (GBM adapter) generates SMILES
                    │
                    ├── Success ──▶ Return result
                    │
                    └── Failure ──▶ Switch to Base Merged model + Few-shot
                                      │
                                      └── Few-shot examples:
                                          SMILES 1: Cc1cc(nc(-c2ccc(Cl)cc2)n1)C(=O)Nc3ccc(F)cc3
                                          SMILES 2: COc1ccc(cc1)C(=O)Nc2cc(C)nc(-c3ccccn3)c2
                                          SMILES 3: CC(=O)Nc1ccc(-n2ccnc2)cc1C(=O)Nc3ccccc3
                                      │
                                      └── Generate new candidates
```

---

## 8. Tree-of-Thoughts (ToT) Reasoning Layer

### 8.1 Three-Level BFS Architecture

```
Level 0: Scaffold Proposal     Level 1: Assembly Strategy    Level 2: SMILES Generation
┌─────────────────────────┐     ┌─────────────────────────┐   ┌─────────────────────────┐
│                         │     │                         │   │                         │
│  Generate k=3 scaffolds │────▶│  For each scaffold:    │──▶│  For each strategy:    │
│                         │     │  Generate k=3 assembly │   │  Generate k=3 SMILES   │
│  parse_scaffold_proposals     │  strategies             │   │                         │
│  evaluate_states_vote()  │     │                         │   │  _generate_smiles_fallback()
│                         │     │  parse_assembly_strategies   │                         │
│  BFS: Keep b=2 best     │     │  evaluate_states_vote() │   │  validate via RDKit    │
│                         │     │                         │   │                         │
│  [Scaffold 1: quinazoline]│   │  BFS: Keep b=2 best   │   │  _correct_smiles_via_llm()
│  [Scaffold 2: pyridine]   │   │                         │   │                         │
│                         │     │  [Strategy 1]           │   │  PHYSICAL EVALUATION    │
│                         │     │  [Strategy 2]           │   │  GBMPhysicalEvaluator   │
│                         │     │                         │   │  evaluate()             │
│                         │     │                         │   │                         │
└─────────────────────────┘     └─────────────────────────┘   │  HARD PRUNE if is_pruned│
                                                              │                         │
                                                              │  [verdict: sure/likely]│
                                                              └───────────┬─────────────┘
                                                                          │
                                                    ┌─────────────────────┘
                                                    │
                                         ┌──────────▼──────────┐
                                         │                     │
                                         │  Multi-round        │
                                         │  Refinement Loop    │
                                         │                     │
                                         │  tot_refinement_rounds
                                         │  (up to N rounds)   │
                                         │                     │
                                         │  ┌─────────────────┴──────────┐
                                         │  │ Build physical feedback:    │
                                         │  │ "Vina=-9.2 | DILI=0.12 |   │
                                         │  │  BBB=0.78 | Reward=0.84"   │
                                         │  └─────────────────┬──────────┘
                                         │                    │
                                         │         ┌──────────▼──────────┐
                                         │         │ Inject into next     │
                                         │         │ round's prompt       │
                                         │         │ (feedback loop)      │
                                         │         └─────────────────────┘
                                         └─────────────────────────────┘
```

### 8.2 ToT Node Structure

Each node in the reasoning tree:

```python
@dataclass
class ToTNode:
    level: int                    # 0=scaffold, 1=assembly, 2=smiles
    content: str                  # The content (name/smiles)
    parent: Optional['ToTNode']   # Parent node (None for Level 0)
    children: List['ToTNode']    # Child nodes
    evaluation: str               # 'sure' | 'likely' | 'impossible'
    metadata: Dict[str, Any]     # Additional data
    physical_result: Optional[PhysicalEvaluationResult]  # Physics-based evaluation
```

### 8.3 BFS Search Algorithm

```
Algorithm: BFS ToT Search
─────────────────────────────────────────────────────
1.  Initialize search_start_timestamp
2.  Generate domain_prompt from GBMPromptGenerator

3.  LEVEL 0 (Scaffold):
    a.  Build scaffold_prompt
    b.  _generate_structured_with_retry() ──▶ scaffold_response
    c.  parse_scaffold_proposals() ──▶ list of k scaffolds
    d.  evaluate_states_vote() ──▶ [sure/likely/impossible] × k
    e.  Filter: remove "impossible" scaffolds
    f.  Sort by evaluation score: sure > likely
    g.  Select top b scaffolds (BFS pruning)
    h.  FALLBACK if all impossible: hardcoded scaffolds

4.  LEVEL 1 (Assembly) — for each selected scaffold:
    a.  Build assembly_prompt
    b.  _generate_structured_with_retry() ──▶ assembly_response
    c.  parse_assembly_strategies() ──▶ k strategies
    d.  evaluate_states_vote() ──▶ evaluations
    e.  Filter + BFS prune → top b strategies
    f.  FALLBACK if all impossible: hardcoded strategies

5.  LEVEL 2 (SMILES) — for each selected strategy:
    a.  Build smiles_prompt
    b.  For refinement_round in [1..tot_refinement_rounds]:
        i.   Build physical_feedback from previous round
        ii.  _generate_smiles_fallback() ──▶ smiles_response
        iii. parse_smiles() ──▶ list of k candidates
        iv.  For each candidate:
             - _check_rdkit_smiles(): validate TPSA/MW
             - If invalid: _correct_smiles_via_llm() (2 retries)
        v.   Evaluate via GBMPhysicalEvaluator.evaluate()
             - Vina docking, DILI, hERG, BBB
             - Compute MPO reward
             - HARD PRUNE if is_pruned=True
        vi.  Track best_node per branch
        vii. Early stop if reward >= tot_good_reward_threshold
    c.  Collect all valid SMILES nodes → final_smiles_nodes

6.  Budget Check (at each step):
    - LLM calls >= tot_max_llm_calls_per_search → STOP
    - Elapsed time >= tot_max_search_seconds → STOP

7.  Return final_smiles_nodes
```

### 8.4 Self-Correction Retry Mechanism

```python
_generate_structured_with_retry():
    error_warning = ""  # Initialize before loop
    _smiles_pat = r"\b([CNOSPFI][A-Za-z0-9@+\-\[\]\(\)=#%\/\.]{5,})\b"
    
    for attempt in range(max_retries):
        is_retry = (attempt > 0)
        
        # Temperature annealing: first attempt=0.8, retry=0.2
        temperature = 0.2 if is_retry else 0.8
        top_p = 0.85 if is_retry else 0.95
        
        # Token budget: retry gets 60% of original
        retry_tokens = int(max_new_tokens * 0.6) if is_retry else max_new_tokens
        
        # Current prompt = error_warning (if any) + original prompt
        _current_prompt = (error_warning + "\n\n" + prompt) if error_warning else prompt
        
        # Append strict format reminder ONLY on retry
        if is_retry:
            _current_prompt += "\n\n[STRICT FORMAT — copy this EXACT structure:]"
        
        # LLM call
        response = generate_with_model(_current_prompt, max_new_tokens=retry_tokens)
        
        # SMILES validation (only for Level 2)
        if structured_type == "smiles":
            _invalid_smiles = [...]
            if _invalid_smiles:
                error_warning = "【化学错误】..."
                continue  # Stateless: next attempt uses error_warning + original
        
        # Success
        break
```

---

## 9. Physical Evaluation Layer

### 9.1 GBMPhysicalEvaluator Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                  GBMPhysicalEvaluator                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   VinaDockingTool                          │ │
│  │  Input: SMILES                                             │ │
│  │  Output: Best binding pose, affinity (kcal/mol)             │ │
│  │  Method: AutoDock Vina / QVina 2D                          │ │
│  │  Threshold: < -5.0 kcal/mol (valid)                         │ │
│  │  ──────────────────────────────────────────────────────────  │ │
│  │  Fallback: _pseudo_dock() when vina_executable unavailable   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  ADMETCalculator                            │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │ │
│  │  │     DILI     │  │     hERG     │  │       BBB        │  │ │
│  │  │   (Hepatotox)│  │  (Cardiotox) │  │  (Brain Barrier) │  │ │
│  │  │  SMARTS      │  │   SMARTS     │  │  Clark Formula   │  │ │
│  │  │  alerts      │  │  alerts      │  │  TPSA + HBD      │  │ │
│  │  │  + LogP      │  │  + LogP      │  │  > 0 = HIGH     │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              MPORewardCalculator                            │ │
│  │  Multi-Objective Optimization Reward                        │ │
│  │  ─────────────────────────────────────────────────────────  │ │
│  │  Hard Cutoffs:                                             │ │
│  │    Vina > -5.0 kcal/mol  ──────▶  REJECT                   │ │
│  │    DILI probability > 0.95 ──▶  REJECT                    │ │
│  │    hERG probability > 0.90 ──▶  REJECT                    │ │
│  │    MW < 200 or MW > 900 ──────▶  REJECT                    │ │
│  │    TPSA < 20 or TPSA > 140 ───▶  REJECT                    │ │
│  │  ─────────────────────────────────────────────────────────  │ │
│  │  Normalized Scores (0-1):                                   │ │
│  │    Vina_norm = clip(|vina| / 12, 0, 1)                     │ │
│  │    DILI_norm = 1 - dili_prob                                │ │
│  │    hERG_norm = 1 - herg_prob                               │ │
│  │    BBB_norm = bbb_prob                                     │ │
│  │    PhysChem_norm = average(TPSA, LogP, MW norms)           │ │
│  │  ─────────────────────────────────────────────────────────  │ │
│  │  Weights:                                                   │ │
│  │    Vina: 0.30  |  DILI: 0.20  |  hERG: 0.20               │ │
│  │    BBB:  0.20  |  PhysChem: 0.10                           │ │
│  │  ─────────────────────────────────────────────────────────  │ │
│  │  Reward = 0.30·V + 0.20·D + 0.20·H + 0.20·B + 0.10·P     │ │
│  │  ─────────────────────────────────────────────────────────  │ │
│  │  Delta Scoring: Compare to baseline (Temozolomide)          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              PhysicalEvaluationResult                       │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │ vina_score: float        # -12.5 ~ -5.0 kcal/mol     │ │ │
│  │  │ dili_prob: float         # 0.0 ~ 1.0                  │ │ │
│  │  │ herg_prob: float         # 0.0 ~ 1.0                  │ │ │
│  │  │ bbb_prob: float          # 0.0 ~ 1.0                  │ │ │
│  │  │ tpsa: float              # Å²                         │ │ │
│  │  │ mw: float                # Da                         │ │ │
│  │  │ logp: float              # LogP                       │ │ │
│  │  │ reward: float            # 0.0 ~ 1.0                  │ │ │
│  │  │ verdict: Enum            # SURE / LIKELY / IMPOSSIBLE │ │ │
│  │  │ is_pruned: bool          # True = discard this node   │ │ │
│  │  │ prune_reason: str        # Why it was pruned          │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### 9.2 BBB Permeability Calculation (Clark Formula)

```
Clark_logD = LogP - TPSA / 50

If Clark_logD > 0.15  →  bbb_prob = 1.0 (HIGH CNS penetration)
If Clark_logD > -0.3  →  bbb_prob = 0.8 (MODERATE)
If Clark_logD > -1.0  →  bbb_prob = 0.5 (LOW)
Otherwise              →  bbb_prob = 0.1 (VERY LOW)

Plus penalty for:
  - More than 5 H-bond donors (penalty: -0.2)
  - More than 10 H-bond acceptors (penalty: -0.1)
```

### 9.3 SMARTS Alert Rules

**DILI (Drug-Induced Liver Injury) Alerts:**
```
- Michael acceptors (>3)
- Aromatic amines
- Phenol (high MW > 200)
- Halogenated benzene (> 2 halogens)
- Polycyclic aromatic hydrocarbons (> 3 rings)
- Specific toxicophores: allylic alcohols, hydrazines, etc.
```

**hERG (Cardiac Potassium Channel) Alerts:**
```
- Basic nitrogen (pKa > 7.5) + aromatic ring
- Tertiary amines with aromatic substituents
- Quaternary ammonium
- High LogP (> 3.5) with basic centers
```

---

## 10. Output Layer

### 10.1 Output Directory Structure

```
tot_batch_YYYYMMDD_HHMMSS/           # Experiment root
├── experiment_config.json            # Full experiment configuration
├── all_results_checkpoint.json       # Real-time checkpoint (updated every attempt)
├── all_results.json                 # Final aggregated results
├── summary.json                     # Statistics summary
│
├── csv_output/                      # CSV exports for analysis
│   ├── EGFR.csv                     # Each target gets its own CSV
│   ├── VEGF_VEGFR.csv
│   ├── IDH1_IDH2.csv
│   ├── MGMT.csv
│   ├── PD_1_PD_L1.csv
│   ├── PI3K_AKT_mTOR.csv
│   └── p53_MDM2.csv
│
├── EGFR/                            # Per-target detailed results
│   └── tot_generation_results.json
├── VEGF_VEGFR/
│   └── tot_generation_results.json
└── ...
```

### 10.2 CSV Format

```csv
ID,SMILES,TPSA,MW,Target
EGFR_00001,Cc1cc(nc(-c2ccc(Cl)cc2)n1)C(=O)Nc3ccc(F)cc3,54.88,341.77,EGFR
EGFR_00002,COc1ccc(cc1)C(=O)Nc2cc(C)nc(-c3ccccn3)c2,67.89,378.45,EGFR
```

### 10.3 ToT Results JSON Schema

```json
{
  "config": {
    "target": "EGFR",
    "tot_depth": 3,
    "tot_k": 3,
    "tot_b": 2,
    "constraints": {
      "mw_range": "300-500",
      "bbb_requirement": "high",
      "logp_range": "2.0-4.0",
      "tpsa_range": "40-120"
    }
  },
  "num_generated": 50,
  "molecules": [
    {
      "smiles": "Cc1cc(nc(-c2ccc(Cl)cc2)n1)C(=O)Nc3ccc(F)cc3",
      "tpsa": 54.88,
      "mw": 341.77,
      "logp": 4.12,
      "tot_path": [
        {
          "level": 0,
          "content": "quinazoline",
          "evaluation": "likely",
          "metadata": {
            "name": "quinazoline",
            "rationale": "EGFR-approved scaffold with high BBB penetration",
            "base_mw": 130,
            "bbb_potential": "high"
          }
        },
        {
          "level": 1,
          "content": "quinazoline + acrylamide",
          "evaluation": "sure",
          "metadata": {
            "warhead": "acrylamide",
            "bbb_enhancers": "fluorine, methoxy",
            "estimated_mw": 450,
            "expected_logp": "2.5-3.5",
            "expected_tpsa": "60-90"
          }
        },
        {
          "level": 2,
          "content": "Cc1cc(nc(-c2ccc(Cl)cc2)n1)C(=O)Nc3ccc(F)cc3",
          "evaluation": "sure",
          "metadata": {
            "smiles": "Cc1cc(nc(-c2ccc(Cl)cc2)n1)C(=O)Nc3ccc(F)cc3",
            "vina_score": -9.24,
            "dili_prob": 0.12,
            "bbb_prob": 0.78,
            "reward": 0.84,
            "is_pruned": false,
            "refinement_round": 0
          }
        }
      ],
      "target": "EGFR",
      "generation_method": "tot_bfs_feedback",
      "physical_evaluation": {
        "vina_score": -9.24,
        "dili_prob": 0.12,
        "bbb_prob": 0.78,
        "reward": 0.84,
        "verdict": "sure",
        "is_pruned": false
      },
      "physical_feedback": "Vina=-9.24 kcal/mol (STRONG) | DILI=0.12 (minimal risk) | BBB=0.78 (HIGH CNS) | Reward=0.8400"
    }
  ],
  "timestamp": "2026-04-22T16:24:24.123456"
}
```

---

## 11. Complete System Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USAGE ENTRY POINT                                  │
│                    generate_tot_batch_candidates.py                             │
│              python generate_tot_batch_candidates.py --targets EGFR IDH1 ...   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Initialization                                                          │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  Load config from english_gbm_prompts.yaml                                       │
│  Create output directory with timestamp: tot_batch_YYYYMMDD_HHMMSS/              │
│  Initialize TreeOfThoughtsGenerator                                             │
│  Load models: Base → Llamole (merged) → GBM adapter (PeftModel)                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────┐
                    │        For Each Target            │
                    │    EGFR, VEGF_VEGFR, IDH1...      │
                    └───────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Generate for Single Target (generate_molecules_for_target)            │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  While len(all_molecules) < num_molecules AND attempts < max_attempts:         │
│                                                                                  │
│    ┌─────────────────────────────────────────────────────────────────────┐     │
│    │  2a. generator.load_prompt_generator(target_name)                    │     │
│    │       ├─ GBMKnowledgeBase ──────────────┐                           │     │
│    │       ├─ GBMPromptGenerator ────────────┼──▶ domain_prompt          │     │
│    │       └─ GBMPhysicalEvaluator ──────────┘                           │     │
│    └─────────────────────────────────────────────────────────────────────┘     │
│                                        │                                       │
│                                        ▼                                       │
│    ┌─────────────────────────────────────────────────────────────────────┐     │
│    │  2b. generator.generate_molecules(target_name)                      │     │
│    │       └─ bfs_tot_search(target_name)                                │     │
│    │           ├─ Level 0: Scaffold proposal                             │     │
│    │           │   └─ _generate_structured_with_retry()                  │     │
│    │           │       └─ evaluate_states_vote() ──▶ PhysicalEval       │     │
│    │           ├─ Level 1: Assembly strategy                             │     │
│    │           │   └─ _generate_structured_with_retry()                  │     │
│    │           │       └─ evaluate_states_vote() ──▶ PhysicalEval       │     │
│    │           └─ Level 2: SMILES generation                             │     │
│    │               ├─ _generate_smiles_fallback()                        │     │
│    │               │   ├─ [Try: Fine-tuned model]                       │     │
│    │               │   └─ [Fallback: Base merged + Few-shot]            │     │
│    │               ├─ _check_rdkit_smiles() ──▶ validate                 │     │
│    │               ├─ _correct_smiles_via_llm() ──▶ repair              │     │
│    │               └─ GBMPhysicalEvaluator.evaluate()                    │     │
│    │                   ├─ VinaDockingTool.dock()                        │     │
│    │                   ├─ ADMETCalculator.compute()                     │     │
│    │                   ├─ MPORewardCalculator.calculate()               │     │
│    │                   └─ HARD PRUNE if is_pruned=True                  │     │
│    └─────────────────────────────────────────────────────────────────────┘     │
│                                        │                                       │
│                                        ▼                                       │
│    ┌─────────────────────────────────────────────────────────────────────┐     │
│    │  2c. Deduplicate by SMILES against all_molecules                    │     │
│    │      Add new molecules to all_molecules                            │     │
│    └─────────────────────────────────────────────────────────────────────┘     │
│                                        │                                       │
│                                        ▼                                       │
│    ┌─────────────────────────────────────────────────────────────────────┐     │
│    │  2d. Checkpoint Callback (checkpoint_write_fn)                      │     │
│    │      ├─ Update all_results[target_name]                            │     │
│    │      ├─ Write all_results_checkpoint.json                          │     │
│    │      └─ _save_csv_output() ──▶ csv_output/[Target].csv (append)    │     │
│    └─────────────────────────────────────────────────────────────────────┘     │
│                                        │                                       │
│                                        ▼                                       │
│    ┌─────────────────────────────────────────────────────────────────────┐     │
│    │  2e. Early Stop Check                                              │     │
│    │      ├─ no_new_streak >= max_no_new_streak ──▶ BREAK              │     │
│    │      └─ len(all_molecules) >= num_molecules ──▶ BREAK              │     │
│    └─────────────────────────────────────────────────────────────────────┘     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Save Final Results                                                     │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  ├─ Trim all_molecules to num_molecules                                      │
│  ├─ generator.save_results() ──▶ [Target]/tot_generation_results.json         │
│  ├─ Write all_results.json                                                    │
│  ├─ Write summary.json (statistics)                                          │
│  └─ Print summary                                                              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 12. Key Design Patterns

### 12.1 Stateless Retry with Error Feedback

```
Pattern: Prompt = error_warning (from previous) + original_prompt (always fresh)
         ─────────────────────────────────────────────────────────────
         Attempt 1: prompt = [original] ──▶ LLM ──▶ response
                                         │
                                         ├─ Valid? ──▶ BREAK (success)
                                         │
                                         └─ Invalid? ──▶ error_warning = "[化学错误]..."
                                                         ↓
         Attempt 2: prompt = [error_warning] + [original] ──▶ LLM ──▶ response
                                         │
                                         ├─ Valid? ──▶ BREAK (success)
                                         │
                                         └─ Invalid? ──▶ error_warning = "[严重错误]..."
                                                         ↓
         Attempt 3: prompt = [error_warning] + [original] ──▶ LLM ──▶ response
                                                         ↓
                                              Still invalid? ──▶ return last response
```

### 12.2 Two-Model Fallback Strategy

```
Fine-tuned model output:
  ├── parse_smiles() finds valid SMILES ──▶ Use fine-tuned result
  │
  └── parse_smiles() finds NOTHING ──▶
      │
      ├── Save current model/generation_config
      ├── Switch model = _base_merged_model
      ├── Switch generation_config = {temperature: 0.5, top_p: 0.92}
      ├── Append few-shot examples to prompt
      ├── Call generate_with_model() again
      ├── Restore original model/generation_config
      └── Return base model result
```

### 12.3 Multi-Round Refinement with Physical Feedback

```
Round 1:  [Prompt] ──▶ SMILES Generation ──▶ Physical Eval ──▶ Reward=0.45
             │
             ▼
          Feedback: "Vina=-5.5 (weak), DILI=0.2 (low), BBB=0.9 (high)"
             │
             ▼
Round 2:  [Prompt + Feedback] ──▶ SMILES Generation ──▶ Physical Eval ──▶ Reward=0.72
             │
             ▼
          Feedback: "Vina=-7.8 (moderate), DILI=0.15 (minimal), BBB=0.85 (high)"
             │
             ▼
Round 3:  [Prompt + Feedback] ──▶ SMILES Generation ──▶ Physical Eval ──▶ Reward=0.84
             │
             ▼
          reward >= threshold (0.65)? ──▶ YES ──▶ EARLY STOP
```

---

## 13. Experiment Statistics

### 13.1 Output Metrics per Experiment

```json
{
  "summary_stats": {
    "EGFR": {
      "total_generated": 50,
      "unique_smiles": 47,
      "avg_path_length": 3.0,
      "avg_reward": 0.72,
      "avg_vina_score": -8.5,
      "avg_dili_prob": 0.18,
      "avg_bbb_prob": 0.75,
      "verdict_distribution": { "sure": 23, "likely": 19, "impossible": 8 }
    },
    "VEGF_VEGFR": { ... },
    "IDH1_IDH2": { ... }
  },
  "total_targets": 7,
  "total_molecules": 350,
  "targets_breakdown": {
    "EGFR": 50,
    "VEGF_VEGFR": 48,
    "IDH1_IDH2": 52,
    "MGMT": 45,
    "PD_1_PD_L1": 50,
    "PI3K_AKT_mTOR": 53,
    "p53_MDM2": 52
  }
}
```

### 13.2 Guardrail Statistics (Tracked)

```python
guardrail_stats = {
    "vote_parse_incomplete": 0,           # Vote evaluation parse failures
    "all_impossible_scaffold_fallback": 0, # All scaffolds impossible → hardcoded
    "all_impossible_assembly_fallback": 0, # All strategies impossible → hardcoded
    "default_strategy_fallback": 0,        # Parse failure → default strategy
    "llm_self_correction_retries": 0,     # Self-correction attempts
    "llm_self_correction_success": 0,    # Self-correction successes
}
```

---

## 14. Hardware Requirements

```
GPU Memory:
  - Base model (FP16, 7B params):  ~14 GB
  - + Llamole adapter (merged):    ~0 GB (merged into base)
  - + GBM adapter (PeftModel):    ~0.5 GB
  - Total:                        ~14-16 GB per target

  Alternative: 8-bit quantization → ~10 GB total

Training (LoRA):
  - Peak GPU memory: ~24 GB
  - Duration: ~8-16 hours per adapter

Inference Speed:
  - Single ToT search: ~30-120 seconds
  - 50 molecules per target: ~20-60 minutes
```

---

## 15. Color Palette Suggestions for Diagrams

| Component | Color | Hex | Usage |
|-----------|-------|-----|-------|
| Data Layer | Light Blue | `#E3F2FD` | Knowledge base, clinical data boxes |
| Knowledge Layer | Light Green | `#E8F5E9` | Prompt generator, knowledge injector |
| Model Layer | Light Purple | `#F3E5F5` | LLM, adapters |
| ToT Layer | Light Orange | `#FFF3E0` | BFS search, node levels |
| Evaluation Layer | Light Red | `#FFEBEE` | Physical evaluator, scoring |
| Output Layer | Light Gray | `#F5F5F5` | Results, CSV, JSON |
| Arrows | Gray | `#9E9E9E` | Data flow |
| Active Path | Blue | `#2196F3` | Highlighted pipeline |
| GBM Adapter | Purple | `#9C27B0` | Special module emphasis |
| Physical Evaluator | Red | `#F44336` | Critical evaluation path |

---

## 16. Glossary

| Term | Definition |
|------|------------|
| **ToT** | Tree-of-Thoughts — multi-step reasoning strategy |
| **BFS** | Breadth-First Search — breadth over depth in tree traversal |
| **GBM** | Glioblastoma Multiforme — grade IV astrocytoma, most lethal brain tumor |
| **BBB** | Blood-Brain Barrier — selective membrane limiting CNS drug delivery |
| **Vina** | AutoDock Vina — molecular docking software for binding affinity prediction |
| **DILI** | Drug-Induced Liver Injury — hepatotoxicity risk |
| **hERG** | Human Ether-à-go-go-Related Gene — cardiac potassium channel (blockage = arrhythmia) |
| **TPSA** | Topological Polar Surface Area — predictor of oral absorption and BBB permeability |
| **MPO** | Multi-Property Optimization — combining multiple objectives into single reward |
| **LoRA** | Low-Rank Adaptation — parameter-efficient fine-tuning method |
| **SMILES** | Simplified Molecular Input Line Entry System — molecular structure notation |
| **ADMET** | Absorption, Distribution, Metabolism, Excretion, Toxicity — pharmacokinetic properties |
| **Clark LogD** | Clark formula for BBB permeability: LogD = LogP - TPSA/50 |
| **Verdict** | ToT evaluation: SURE (use now) / LIKELY (use) / IMPOSSIBLE (reject) |
| **Reward** | MPO score [0-1], higher = better overall molecule |
| **HARD PRUNE** | Immediate discard of a molecule if it violates hard cutoffs |

---

*Document generated for GPT-Image-2 diagram generation. Last updated: April 2026.*
