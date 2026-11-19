# English Version: Multi-Modal Large Language Model for Glioblastoma Drug Candidate Molecule Generation and Optimization

---

## Abstract

Glioblastoma (GBM) represents the most common and aggressive primary malignant brain tumor in adults, characterized by high malignancy, significant heterogeneity, and extremely poor prognosis. The current standard treatment regimen combining surgery, radiotherapy, and temozolomide chemotherapy achieves a median survival of only 12-15 months, with a five-year survival rate below 5%. The blood-brain barrier (BBB) poses a formidable physiological obstacle to drug delivery, preventing over 98% of small-molecule drugs from effectively reaching tumor sites. The core challenge in GBM drug discovery lies in optimizing BBB penetration while simultaneously improving target binding affinity, pharmacokinetic properties, and safety profiles.

Recent advances in Multi-Modal Large Language Models (MLLMs) have provided novel technical pathways to address these challenges. Existing molecular generation models, although demonstrating strong performance on specific individual metrics, generally lack the capability for deep integration of GBM pathological backgrounds, clinical treatment experiences, and multi-objective pharmacokinetic constraints. To tackle this issue, this thesis proposes an end-to-end GBM drug candidate generation and optimization framework based on the multi-modal LLM Llamole, with the following principal contributions:

First, to address the difficulty of multi-objective constraint optimization in GBM drug discovery, this thesis constructs an end-to-end drug discovery pipeline that integrates GBM domain knowledge retrieval, multi-modal semantic alignment, Chain-of-Thought (CoT) reasoning, Tree-of-Thoughts (ToT) search, and physics-based virtual screening. The pipeline employs Llamole as the control core, integrating Graph Diffusion Transformers and Graph Neural Networks (GNNs) for molecular graph generation and encoding, while utilizing Low-Rank Adaptation (LoRA) parameter-efficient fine-tuning for domain adaptation.

Second, to address the reliability verification of generated molecules, this thesis designs a comprehensive evaluation system integrating molecular docking, ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) prediction, and Multi-Parameter Optimization (MPO) Reward. This system evaluates binding affinity using AutoDock Vina, predicts BBB permeability using the Clark formula, and detects hepatotoxicity (DILI) and cardiotoxicity (hERG) risks via structural alert detection, achieving closed-loop feedback between molecular generation and physical verification.

Third, to address the quality and structural diversity of generated molecules, this thesis proposes a three-stage Breadth-First Search (BFS) strategy based on Tree-of-Thoughts. Through scaffold proposal, fragment assembly, and SMILES generation stages with physics engine feedback injection, the model progressively optimizes molecular structures, significantly improving the generation rate of high-potential candidate molecules.

Fourth, through systematic experimental validation targeting three representative GBM targets—EGFR, VEGF/VEGFR, and IDH1/IDH2—the proposed method demonstrates substantial superiority over baseline models in comprehensive scoring. Among molecules with DrugClip > 0 (predicted active), the DrugClip means are EGFR 0.57, IDH1 0.93, VEGF 0.62, with an overall mean of 0.74. Vina docking means are EGFR -5.34 kcal/mol, IDH1 -7.35 kcal/mol, VEGF -3.89 kcal/mol. The BBB penetration pass rates are EGFR 47.5%, IDH1_IDH2 45.0%, VEGF_VEGFR 55.6%, with an overall rate of 49.3%.

Ablation experiments demonstrate that removing physical evaluation (DrugClip mean drop 24.0%), removing feedback (drop 20.1%), and removing CoT reasoning (drop 15.8%) cause the most significant damage to generation quality, validating the necessity of each module.

In summary, the proposed multi-modal LLM-based GBM candidate molecule generation framework demonstrates strong performance across multiple dimensions including molecular validity, BBB permeability, and target binding affinity, providing an efficient and reliable data-driven pathway for early-stage GBM innovative drug discovery.

**Keywords**: Glioblastoma; Multi-Modal Large Language Model; Molecular Generation; Blood-Brain Barrier; Tree-of-Thoughts; AutoDock Vina; Multi-Parameter Optimization

---

# Chapter 1 Introduction

## 1.1 Research Background and Significance

### 1.1.1 Clinical Challenges of Glioblastoma

Glioblastoma (GBM), also known as Grade IV astrocytoma, represents the most common and aggressive primary malignant brain tumor in adults. According to epidemiological data from the Central Brain Tumor Registry of the United States (CBTRUS), GBM accounts for approximately 14.9% of all primary brain tumors and 57.3% of all gliomas, with an annual incidence of approximately 3.2 per 100,000 population. The median age at diagnosis is 65 years, and the disease shows a higher prevalence in males than females, with a ratio of approximately 1.6:1.

The current standard treatment protocol for newly diagnosed GBM, known as the Stupp protocol, was established by Stupp et al. in 2005 and remains the cornerstone of GBM therapy to this day. This regimen consists of maximal safe surgical resection followed by concurrent radiotherapy (60 Gy in 30 fractions) and daily temozolomide (TMZ) chemotherapy (75 mg/m²/day), followed by adjuvant TMZ (150-200 mg/m²/day for 5 days every 28 days) for 6-12 cycles. Despite this aggressive multimodal treatment approach, the prognosis for GBM patients remains extremely poor, with a median overall survival of only 12-15 months and a five-year survival rate below 5%. The median progression-free survival is approximately 6-7 months, and almost all patients experience tumor recurrence.

The extremely poor prognosis of GBM can be attributed to several intrinsic biological characteristics:

**Molecular Heterogeneity**: GBM exhibits remarkable inter-tumoral and intra-tumoral heterogeneity at the molecular level. Based on transcriptomic profiling, GBM can be classified into four molecular subtypes: Classical (25%), Mesenchymal (25%), Proneural (25%), and Neural (25%). The Classical subtype is characterized by EGFR amplification and chromosome 7 amplification/chromosome 10 loss, while the Proneural subtype often harbors PDGFRA alterations and IDH1 mutations. This molecular heterogeneity leads to differential responses to targeted therapies, making single-target drugs often ineffective.

**Infiltrative Growth Pattern**: GBM cells demonstrate diffuse infiltration into surrounding normal brain tissue, making complete surgical resection impossible. Even with modern neurosurgical techniques including neuronavigation, intraoperative MRI, and fluorescent-guided surgery, microscopic residual tumor cells invariably remain, leading to inevitable recurrence.

**Therapeutic Resistance**: GBM demonstrates intrinsic and acquired resistance to multiple therapeutic modalities. TMZ resistance occurs in approximately 90% of patients, primarily mediated by O6-methylguanine-DNA methyltransferase (MGMT) repair mechanisms. The blood-brain barrier restricts drug delivery to the tumor site, and the tumor microenvironment promotes immunosuppression and treatment resistance.

**The Blood-Brain Barrier Challenge**: The blood-brain barrier (BBB) represents the most significant physiological obstacle to effective GBM chemotherapy. The BBB is a highly specialized endothelial structure that separates the circulating blood from the brain extracellular fluid, protecting the central nervous system from potentially harmful substances. The BBB prevents over 98% of small-molecule drugs and nearly 100% of large-molecule biologics from reaching the brain parenchyma. For a drug to effectively penetrate the BBB and reach therapeutic concentrations in the brain, it must satisfy strict physicochemical constraints: molecular weight < 400-500 Da, topological polar surface area (TPSA) < 90 Å², lipophilicity (LogP) in the range of 1-4, and limited hydrogen bond donors and acceptors.

### 1.1.2 Emerging Opportunities with AI in Drug Discovery

The convergence of artificial intelligence and drug discovery has created unprecedented opportunities for accelerating the identification of novel therapeutic agents. Recent years have witnessed remarkable advances in several AI technologies that are particularly relevant to the challenge of GBM drug discovery:

**Large Language Models (LLMs)**: The rapid development of transformer-based large language models, exemplified by GPT-4, Claude, and open-source models such as LLaMA and Qwen, has demonstrated unprecedented capabilities in natural language understanding, reasoning, and generation. These models can be adapted for scientific domains including drug discovery through fine-tuning and domain-specific knowledge integration.

**Multi-Modal Learning**: Multi-modal large language models that integrate text, images, graphs, and other data modalities offer the potential for unified representations of complex biomedical data. In molecular design, multi-modal models can jointly process molecular structures (SMILES/SELFIES representations), pharmacological descriptions, and physicochemical property data.

**Graph Neural Networks (GNNs)**: GNNs have emerged as powerful tools for learning representations of molecular graphs, enabling accurate prediction of molecular properties, generation of novel molecular structures, and optimization of lead compounds. Diffusion models combined with GNNs have shown particular promise for molecular generation tasks.

**Multi-Objective Optimization**: Multi-parameter optimization (MPO) approaches that balance multiple drug-like properties simultaneously have become increasingly sophisticated. These methods can navigate the complex trade-off space between target affinity, BBB permeability, metabolic stability, and toxicity.

### 1.1.3 Research Significance

The research presented in this thesis addresses critical challenges in AI-driven GBM drug discovery and holds both scientific significance and practical value:

**Scientific Significance**:
- Explores the feasibility of applying multi-modal large language models to vertical disease domains (GBM) for drug discovery
- Establishes a methodology for multi-objective constrained molecular generation and evaluation for CNS drugs
- Provides a novel technical paradigm for AI-driven precision oncology drug discovery

**Practical Significance**:
- Generates a collection of GBM candidate drug molecules with proprietary intellectual property
- Establishes a reproducible "knowledge retrieval → multi-modal generation → physical verification → feedback optimization" end-to-end pipeline
- Provides a high-potential candidate list for subsequent wet laboratory validation

---

## 1.2 Related Work

### 1.2.1 AI-Driven Molecular Generation

The field of AI-driven molecular generation has evolved rapidly over the past decade, with multiple methodological approaches demonstrating success in de novo drug design. This section reviews the major paradigms and representative works in this domain.

**Variational Autoencoders (VAEs) and Autoencoders**: VAEs learn continuous latent representations of molecular structures in an unsupervised manner, enabling sampling from the latent space to generate novel molecules. Gomez-Bombarelli et al. (2018) demonstrated that VAE-learned latent spaces contain smooth interpolation properties, allowing optimization of molecules toward desired property profiles. However, VAEs often produce invalid molecular structures and have limited control over specific molecular features.

**Generative Adversarial Networks (GANs)**: GANs learn to generate molecules by adversarial training between a generator and discriminator. ORGAN (Optimized GAN) and its variants have been applied to molecular generation with property constraints. While GANs can produce diverse molecules, training stability remains a challenge, and mode collapse can limit the diversity of generated structures.

**Recurrent Neural Networks (RNNs) and Transformers**: Sequence-based generative models that treat molecules as SMILES strings have become widely adopted. Segler et al. (2017) demonstrated that RNNs trained on large molecular databases could generate novel, drug-like molecules. The availability of transformer architectures has further improved the quality and diversity of generated SMILES sequences. MolGPT, developed by Bagal et al. (2021), uses a transformer-decoder architecture for molecular generation and demonstrates competitive performance on benchmark datasets.

**Diffusion Models**: Diffusion-based generative models have recently emerged as powerful approaches for molecular generation. These models learn to reverse a noise-addition process, enabling generation of molecular structures through iterative denoising. Xu et al. (2023) introduced geometric latent diffusion models for 3D molecule generation, demonstrating the ability to generate molecules with realistic 3D conformations. Graph Diffusion Transformers, as implemented in Llamole (Liu et al., 2024), combine transformer architectures with graph representations for multi-conditional molecular generation.

**Reinforcement Learning (RL) Approaches**: RL-based methods frame molecular optimization as a sequential decision-making problem, where the agent learns to modify molecular structures to maximize reward signals encoding desired properties. REINVENT 2.0 (Blaschke et al., 2020) combines RNN-based generation with RL optimization using behavior cloning and memory-based policy updates. Zhou et al. (2019) demonstrated deep RL for multi-objective molecular optimization on the GuacaMol benchmark.

### 1.2.2 Multi-Modal Large Language Models in Drug Discovery

The integration of multi-modal information with large language models represents a frontier in AI-driven drug discovery. Several recent works have explored the combination of molecular structures, textual information, and other modalities:

**Molecular Property Prediction with LLMs**: Jin et al. (2025) proposed chain-of-thought enabled large language models for molecular property prediction, demonstrating that structured reasoning can improve the explainability and accuracy of property predictions. Their approach fuses multi-modal molecular information including molecular graphs, 2D images, and textual descriptions.

**Multi-Modal Molecular Understanding**: Liu et al. (2024) developed GIT-Mol, a multi-modal large language model that processes molecular science data including graphs, images, and text within a unified framework. Similarly, ChemVLM (Li et al., 2025) explored the capabilities of vision-language models in the chemistry domain, demonstrating zero-shot and few-shot learning for chemical property prediction.

**LLM-Based Drug Discovery Agents**: Li et al. (2025) introduced DrugPilot, an LLM-based parameterized reasoning agent for drug discovery that integrates multiple tools and databases. This agentic approach enables complex multi-step reasoning chains for virtual screening and lead optimization.

**Inverse Molecular Design with Llamole**: Liu et al. (2024) proposed Llamole, a multi-modal large language model specifically designed for inverse molecular design with retrosynthetic planning. Llamole integrates Graph Diffusion Transformers, GNN encoders, and A* search for retrosynthesis planning, demonstrating superior performance over 14 baseline models on controlled molecular generation and retrosynthesis tasks.

### 1.2.3 GBM-Targeted Drug Development

The pursuit of effective GBM therapies has led to extensive research on various molecular targets. This section reviews the major target classes and associated drug development efforts.

**EGFR and EGFRvIII**: Epidermal Growth Factor Receptor (EGFR) amplification occurs in approximately 40% of GBM cases, making it one of the most frequently mutated oncogenes. EGFRvIII, a constitutively active deletion mutant, is expressed in approximately 20-30% of GBMs. Multiple EGFR tyrosine kinase inhibitors (TKIs) including erlotinib, gefitinib, and afatinib have been tested in GBM clinical trials but demonstrated limited efficacy, primarily due to BBB penetration challenges and activation of compensatory signaling pathways. ABT-414, an EGFR antibody-drug conjugate, showed promise in early trials but failed in Phase III due to insufficient efficacy.

**VEGF/VEGFR Pathway**: The vascular endothelial growth factor (VEGF) pathway drives angiogenesis, which is essential for GBM tumor growth and invasion. Bevacizumab, an anti-VEGF monoclonal antibody, received FDA approval for recurrent GBM but failed to improve overall survival in frontline settings. Cediranib, a VEGFR TKI, demonstrated anti-edema effects but limited anti-tumor activity.

**IDH1/IDH2 Mutations**: Mutations in Isocitrate Dehydrogenase 1 and 2 (IDH1/IDH2) occur in approximately 10% of GBMs, predominantly in secondary GBMs that evolved from lower-grade gliomas. IDH mutations are associated with better prognosis and distinct metabolic vulnerabilities. Ivosidenib and enasidenib, IDH1/IDH2 inhibitors approved for acute myeloid leukemia, are being investigated in IDH-mutant glioma.

**PI3K/AKT/mTOR Pathway**: The PI3K/AKT/mTOR signaling pathway is activated in approximately 80% of GBMs through various mechanisms including PTEN loss, PIK3CA mutations, and AKT amplification. GDC-0084, a brain-penetrant PI3K/mTOR inhibitor developed by Genentech, showed promise in Phase I trials for GBM (Heffron et al., 2016).

Despite extensive research efforts, no targeted therapy has yet demonstrated survival benefit in Phase III clinical trials for newly diagnosed GBM. The major challenges include BBB penetration, molecular heterogeneity, adaptive resistance mechanisms, and the immunosuppressive tumor microenvironment.

---

## 1.3 Research Content

This thesis addresses the following core research questions:

**Question 1**: How to construct a unified multi-modal representation space that effectively integrates GBM pathological backgrounds, pharmacological characteristics, molecular structures, and bioactivity information?

**Question 2**: How to guide multi-modal large language models to perform controllable molecular generation and optimization according to complex multi-objective constraints (BBB permeability, target selectivity, pharmacokinetic properties)?

**Question 3**: How to establish a rapid and reliable virtual verification system for generated molecules, achieving closed-loop feedback between generation and evaluation?

### 1.3.1 Multi-Modal Data Integration and Knowledge Injection Framework for GBM

This thesis constructs a multi-modal knowledge base for GBM, integrating three key data types:

**GBM Target Information Database**: Covers molecular descriptions, mutation types, existing drugs, and design challenges for 8 GBM-related targets: EGFR, VEGF/VEGFR, IDH1/IDH2, MGMT, GBM stem cells, PD-1/PD-L1, PI3K/AKT/mTOR, and p53/MDM2. Structural requirements (MW range, LogP range, BBB permeability requirements) for each target are also integrated.

**GBM Clinical Treatment Database**: Covers standard treatment protocols (Stupp protocol), treatment challenges (TMZ resistance rate ~90%, BBB limitation), failed clinical trial insights, successful treatment patterns, and resistance mechanisms.

**GBM Reference Molecule Database**: Covers approved drugs (temozolomide, bevacizumab, everolimus), clinical candidates (GDC-0084, ABT-414), and design principles/structural motifs.

The structural knowledge injector converts this multi-modal knowledge into natural language prompts understandable by LLMs, injecting molecular scaffold preferences, pharmacophores, and BBB optimization strategies through structured templates.

### 1.3.2 Llamole-Based Molecular Graph-Text Collaborative Generation Architecture

This thesis employs Llamole as the core generation engine, constructing a molecular graph-text collaborative generation architecture:

**Base Model Selection**: Qwen2-7B-Instruct is selected as the base language model, which possesses strong instruction-following and reasoning capabilities.

**Llamole Image-Text Adapter**: The pre-trained Llamole image-text adapter is loaded to establish cross-modal semantic alignment between molecular graphs and text descriptions. The core innovation involves trigger tokens (<design_start>, <design_end>, etc.) and semantic query tokens (<query>) that enable efficient mapping from LLM hidden layer semantic information to the graph model's conditional space.

**GBM Domain LoRA Fine-Tuning Adapter**: On top of the Llamole image-text adapter, LoRA technology (r=16, lora_alpha=32) is applied for parameter-efficient fine-tuning on GBM domain knowledge and molecular generation tasks, resulting in the Llamole-Qwen2-7B-Instruct-Adapter-gbm-with-graph-models adapter.

**Graph Model Components**:
- Graph Diffusion Decoder: Generates 2D graph structures of molecules based on pharmacophore instructions extracted from the LLM
- Graph Neural Network Encoder (GIN): Encodes generated molecular structures back into the vector space understandable by the LLM, enabling closed-loop feedback
- Graph Neural Network Predictor: Predicts bioactivity properties of molecules

### 1.3.3 Chain-of-Thought and Tree-of-Thoughts Collaborative Reasoning Strategy

This thesis designs a CoT and ToT dual-engine collaborative reasoning strategy to guide structured molecular design:

**Chain-of-Thought Reasoning**: Guides the model through 7 steps of molecular design—target selection and validation → scaffold design → pharmacophore optimization → ADMET property optimization → GBM-specific verification → comprehensive evaluation → candidate output. Each step injects target-specific biological constraints.

**Tree-of-Thoughts Search**: Employs Breadth-First Search (BFS) strategy, generating multiple candidate nodes at three stages:
- Stage 0 (Scaffold Proposal): Proposes 3 candidate molecular scaffolds with MW estimates and BBB permeability assessments
- Stage 1 (Fragment Assembly): Designs 3 fragment assembly strategies for the selected scaffold, including MW budget allocation and physicochemical property predictions
- Stage 2 (SMILES Generation): Finally generates SMILES strings satisfying constraints, requiring 2-8 aromatic rings and at least one polar group

**Physics Engine Feedback Injection**: After each ToT search round, the physics evaluation engine injects Vina docking scores, DILI risks, BBB scores, and MPO Rewards into the next round prompt, achieving data-driven structure optimization guidance.

### 1.3.4 Multi-Dimensional Physical Evaluation and Closed-Loop Feedback System

This thesis designs a comprehensive evaluation system integrating physical computation and machine learning prediction:

**Vina Docking Evaluation**: Uses AutoDock Vina to dock generated molecules against GBM target crystal structures (EGFR, VEGF, etc.), evaluating binding affinity (kcal/mol). Hard cutoff threshold: molecules with Vina > -5.0 kcal/mol are directly pruned.

**ADMET Prediction**:
- DILI (Drug-Induced Liver Injury) Risk: Detects structural alerts via SMARTS pattern matching (acid chlorides, isothiocyanates, epoxides, quinones, etc.)
- hERG (Cardiotoxicity) Risk: Detects alkaline nitrogen atoms, naphthalene rings, nitroaromatics, etc.
- BBB Permeability: Uses Clark formula (logBB = 0.152×LogP - 0.0148×TPSA + 0.139) with MW/HBD/HBA heuristic penalties

**MPO Reward**: Uses weighted geometric mean to aggregate multi-dimensional scores:
- Vina normalized score (weight 0.30)
- DILI safety (1-DILI_prob, weight 0.20)
- hERG safety (1-hERG_prob, weight 0.20)
- BBB permeability (weight 0.20)
- Physicochemical properties (weight 0.10)

---

## 1.4 Innovations

### 1.4.1 Multi-Modal Domain Knowledge Injection Mechanism for GBM

This thesis is the first to integrate GBM target biology information, clinical treatment experience, and molecular structure knowledge into a structured multi-modal knowledge base, achieving efficient knowledge transfer to the generation model through domain LoRA fine-tuning. Compared to general-purpose molecular generation models, this mechanism enables the model to perform target-specific molecular design for GBM molecular features such as EGFR amplification and IDH mutations. Experiments show that removing domain knowledge injection reduces high-score molecular generation quality by approximately 20%, validating the necessity of this mechanism.

### 1.4.2 CoT-ToT Collaborative Reasoning and Physics Verification Closed-Loop Multi-Stage Molecular Generation Strategy

This thesis innovatively combines Chain-of-Thought reasoning with Tree-of-Thoughts search, constructing a closed-loop optimization system with quantitative feedback from the physics engine. Compared to direct generation, the CoT-ToT reasoning strategy improves the high-score molecular comprehensive score mean by approximately 26.5%. This strategy decomposes molecular generation into three verifiable stages—scaffold proposal → fragment assembly → SMILES generation—where outputs at each stage can be quantitatively evaluated and feedback-guided through the physics engine.

### 1.4.3 Rapid Drug-Likeness Evaluation System Based on Structural Alert Detection and Clark Formula

This thesis designs a drug-likeness evaluation system that can run without external model APIs. Structural alert detection (based on SMARTS pattern matching) can complete toxicity group screening at millisecond level, while Clark formula with heuristic penalties can complete BBB permeability prediction at microsecond level. MPO Reward achieves multi-objective balance through weighted geometric mean. Experimental results show that the physical evaluation feedback module contributes most significantly to final quality, with generation quality decreasing by approximately 20% after removing this module.

---

## 1.5 Thesis Structure

This thesis is organized as follows:

**Chapter 1: Introduction**
This chapter阐述本文的研究背景与意义，分析国内外研究现状，明确本文的研究内容与主要创新点，并给出本文的组织结构。

**Chapter 2: Related Technologies and Theoretical Foundations**
This chapter introduces the core technologies involved in this thesis, including large language model and multi-modal learning foundations, graph neural networks and graph diffusion models, molecular docking and ADMET prediction theories, and Chain-of-Thought and Tree-of-Thoughts reasoning strategies.

**Chapter 3: GBM Candidate Molecule Generation System Framework Based on Multi-Modal Large Language Model**
This chapter provides a detailed description of the proposed GBM candidate molecule generation and optimization framework's overall architecture, including multi-modal data integration scheme, Llamole model adaptation strategy, CoT-ToT collaborative reasoning mechanism, and physics evaluation closed-loop feedback process.

**Chapter 4: Experimental Results and Analysis**
This chapter conducts systematic experimental validation on three representative GBM targets—EGFR, VEGF/VEGFR, and IDH1/IDH2—evaluating the performance of generated molecules on DrugClip Score, Vina docking score, BBB permeability, and other metrics, and conducts ablation experiments to analyze the contribution of each module.

**Chapter 5: Summary and Outlook**
This chapter summarizes the main work and contributions of this thesis, discusses the limitations and improvement directions of the method, and looks forward to future research.

---

# Chapter 2 System Framework

## 2.1 Introduction

This chapter provides a detailed introduction to the GBM candidate molecule generation and optimization framework based on multi-modal large language models proposed in this thesis. First, the overall system architecture design is presented, followed by detailed descriptions of the multi-modal GBM knowledge base construction method, Llamole model domain adaptation strategy, Chain-of-Thought and Tree-of-Thoughts collaborative reasoning mechanism, and physics engine closed-loop evaluation system.

## 2.2 Overall System Architecture

### 2.2.1 Architecture Design Overview

The GBM candidate molecule generation system proposed in this thesis consists of four core modules: GBM knowledge base, GBM specialized prompt generator, GBM molecular generator, and physical evaluation pipeline. The system architecture is illustrated in Figure 2.1.

**System Inputs**:
- GBM target selection (e.g., EGFR, VEGF/VEGFR, IDH1/IDH2, etc.)
- Molecular generation quantity

**System Outputs**:
- Structurally novel GBM candidate drug molecule SMILES sequences
- Multi-dimensional evaluation reports (DrugClip Score, Vina docking score, BBB score, DILI/hERG risk, etc.)
- MPO Reward comprehensive score

### 2.2.2 Module Responsibilities

**GBMKnowledgeBase Module**: Manages GBM domain multi-modal knowledge, providing target information queries, clinical insight retrieval, and molecular references. The core code defines GBMTarget, GBMClinicalData, and GBMMolecule data classes, along with the GBMKnowledgeBase management class.

**GBMPromptGenerator Module**: Converts GBM knowledge into structured natural language prompts for injection into the Llamole model. Key methods include generate_full_prompt(), build_tot_propose_prompt(), and build_feedback_injected_prompt().

**GBMGenerator Module**: Encapsulates Llamole model calls and molecular generation results. The core code is located in gbm_project/src/gbm_generator.py, with key methods including generate_gbm_molecules() and _generate_single_molecule().

**GBMPhysicalEvaluator Module**: Evaluates generated molecules through Vina docking, ADMET prediction, and multi-objective scoring. The module integrates four evaluation components: VinaDockingTool, ADMETCalculator, MPORewardCalculator, and GBMPhysicalEvaluator.

### 2.3 Llamole Model Architecture and Adaptation

### 2.3.1 Llamole Base Model Architecture

Llamole is a multi-modal large language model specifically designed for inverse molecular design with retrosynthetic planning. Its core architecture consists of the following components:

**Base Language Model**: Qwen2-7B-Instruct serves as the core language understanding and generation engine, with approximately 7 billion parameters.

**Trigger Token Mechanism**: Special tokens including <design_start>, <design_end>, <design_body>, <molecule> mark molecular design boundaries.

**Semantic Query Token** (<query>): Extracts GBM-related semantic information from LLM hidden layers (e.g., "high permeability"), which is projected through a linear layer and aligned to the conditional space of the graph diffusion model.

**Graph-LM Connectors**:
- lm_to_graph_decoder: Maps LLM hidden states to the graph diffusion decoder's conditional space
- graph_to_lm_connector: Maps molecular representations encoded by GNNs back to the LLM's semantic space
- lm_to_graph_predictor: Connects the LLM with the GNN predictor

### 2.3.2 GBM Domain LoRA Fine-Tuning

Llamole employs Low-Rank Adaptation (LoRA) technology for parameter-efficient fine-tuning to adapt to the GBM drug design domain:

```python
lora_config = LoraConfig(
    r=16,                    # Rank of low-rank matrices
    lora_alpha=32,           # Scaling factor
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],                      # Target attention modules
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### 2.4 Chain-of-Thought and Tree-of-Thoughts Reasoning Strategy

### 2.4.1 Chain-of-Thought Reasoning Template

The 7-step reasoning process:

**Step 1: Target Selection and Validation**
- Select the most relevant target based on GBM molecular subtypes
- Consider target expression levels and functions in GBM
- Evaluate target criticality for GBM growth

**Step 2: Molecular Scaffold Design**
- Select scaffolds with BBB penetration potential
- Reference structural features of known GBM drugs
- Ensure synthetic feasibility and drug-likeness

**Step 3: Pharmacophore Optimization**
- Design structures with high target affinity
- Optimize selectivity and binding stability
- Consider resistance mutation impacts

**Step 4: ADMET Property Optimization**
- Predict and optimize BBB permeability
- Assess hepatotoxicity and nephrotoxicity risks
- Optimize metabolic stability and half-life

**Step 5: GBM-Specific Verification**
- Evaluate activity against GBM cells
- Verify toxicity against normal cells
- Confirm target selectivity

**Step 6: Comprehensive Evaluation**
- Balance multi-objective constraints
- Assess structural novelty
- Evaluate synthetic feasibility

**Step 7: Candidate Molecule Output**
- Output SMILES string
- Provide design rationale
- Annotate potential risks and improvement directions

### 2.4.2 Tree-of-Thoughts Search Strategy

**BFS Search Configuration**:
- Candidates per stage (k): 3 or 4 (configurable)
- Maximum depth: 3 (scaffold → assembly → SMILES)
- Evaluation feedback rounds: configurable

**Three-Stage Search Process**:

**Stage 0: Scaffold Proposal**
- Input: Target name, MW range (300-600 Da), BBB requirement, LogP range (1-4)
- Process: Model proposes 3 candidate scaffolds (e.g., quinazoline, pyrimidine, benzimidazole)
- Physical evaluation: Preliminary BBB potential evaluation at scaffold level

**Stage 1: Fragment Assembly**
- Input: Selected scaffold, base MW, remaining MW budget
- Process: Design 3 fragment assembly strategies for the selected scaffold
- Physical evaluation: MW/LogP/TPSA estimation and validation at assembly strategy level

**Stage 2: SMILES Generation**
- Input: Selected scaffold + assembly strategy, warhead type, enhancer groups, target MW/LogP/TPSA ranges
- Requirements: Must contain 2-8 aromatic rings and at least one polar group
- Physical evaluation: Comprehensive evaluation including Vina docking, DILI, hERG, and BBB

### 2.5 Physics Evaluation and Closed-Loop Feedback

### 2.5.1 AutoDock Vina Docking

**Docking Box Parameters**:
```yaml
vina_center: [0.0, 0.0, 0.0]
vina_size: [20.0, 20.0, 20.0]
exhaustiveness: 8
n_poses: 1
timeout: 30
```

### 2.5.2 BBB Permeability Prediction

**Clark Formula**:
```
logBB_est = 0.152 × LogP - 0.0148 × TPSA + 0.139
BBB_score = (logBB_est + penalty + 1.0) / 1.5
# penalty = -0.20 (if MW > 450) + -0.15 (if HBD > 2) + -0.10 (if HBA > 6)
```

### 2.5.3 MPO Reward Calculation

**Weighted Geometric Mean Formula**:
```
Reward = exp( Σ_i w_i × ln(max(s_i, 1e-8)) )
```

Where weights are: Vina (0.30), DILI safety (0.20), hERG safety (0.20), BBB (0.20), Physicochemical (0.10).

## 2.6 Chapter Summary

This chapter has provided a detailed introduction to the overall architecture and core module design principles and technical implementations of the GBM candidate molecule generation and optimization framework based on multi-modal large language models. First, the overall system architecture was presented, clarifying the complete processing flow from GBM knowledge input to candidate molecule output. Then, the construction method of the multi-modal GBM knowledge base was described in detail, covering target information, clinical treatment experience, and molecular reference data. The Llamole model adaptation process from base Qwen2-7B-Instruct to GBM domain LoRA fine-tuning was systematically explained. The CoT-ToT collaborative reasoning strategy's three-stage search mechanism and physics feedback injection method were presented. Finally, the comprehensive evaluation system integrating Vina docking, ADMET prediction, and MPO Reward was introduced, laying the methodological foundation for experimental validation in the next chapter.

---

# Chapter 3 Core Methods

## 3.1 Introduction

This chapter provides a detailed exposition of the core methods for the GBM candidate molecule generation and optimization system based on multi-modal large language models. First, the construction method of the GBM target knowledge base is described, which integrates structural and functional information for 7 key therapeutic targets. Then, the design principles of the specialized prompt generator are presented, which dynamically constructs high-quality molecular design prompts based on target specificity and constraint conditions. The multi-modal molecular generation model is introduced, describing how chemical language, molecular graph structures, and biomedical knowledge are deeply integrated. Finally, the physical evaluation methods are elaborated, including molecular docking, ADMET prediction, MPO Reward comprehensive scoring, hard pruning and Delta Scoring strategies, and the application of the ToT three-stage search algorithm in molecular generation.

## 3.2 GBM Target Knowledge Base Construction

### 3.2.1 Knowledge Base Architecture Design

The GBM target knowledge base serves as the core data foundation for the entire generation system, managing multi-dimensional professional knowledge related to GBM drug development. The knowledge base adopts a modular design with the following four core sub-databases:

**GBM Target Information Database**: Contains detailed information for 7 key GBM therapeutic targets. Each target entry includes: name, description, mutation_types, current_drugs, challenges, and structural_requirements.

**GBM Clinical Treatment Database**: Contains the latest progress and lessons learned in GBM clinical treatment, including standard treatment protocols (Stupp protocol), clinical trial insights, molecular subtype classification, biomarkers, and resistance mechanisms.

**GBM Reference Molecule Database**: Contains structural and activity information for approved drugs and clinical trial molecules, including approved drugs (temozolomide, bevacizumab, everolimus), clinical candidates (ABT-414, GDC-0084), and structural scaffolds (pyrimidine rings, quinazoline rings).

**Structural Knowledge Base**: Provides professional knowledge injection for molecular structure design, including BBB penetration structural modification strategies, PAINS and Brenk warning structure filtering rules, common active pharmacophore identification, and synthetic accessibility assessment.

### 3.2.2 Core Data Classes

```python
@dataclass
class GBMTarget:
    name: str
    description: str
    mutation_types: List[str]
    current_drugs: List[str]
    challenges: List[str]
    structural_requirements: Dict

@dataclass
class GBMClinicalData:
    standard_treatment: str
    challenges: List[str]
    molecular_subtypes: Dict
    biomarkers: List[str]
    resistance_mechanisms: List[str]

@dataclass
class GBMMolecule:
    name: str
    smiles: str
    target: str
    status: str
    mechanism: str
    mol_weight: float
    logp: float
    tpsa: float
```

## 3.3 Specialized Prompt Generator Design

### 3.3.1 Prompt Template System

The specialized prompt generator constructs a multi-level, multi-strategy prompt template system based on YAML configuration files. Core templates include:

**Base GBM Prompt Template**: Defines the general requirement framework for GBM molecular design.

**Target-Specific Prompt Template**: Dynamically constructs target-specific prompts based on target information from the knowledge base.

### 3.3.2 CoT Reasoning Prompt Design

Chain-of-Thought reasoning prompts guide the model through step-by-step molecular design reasoning:

**Step-by-Step Design Template**:
```
Step 1: Target Analysis
- Analyze structural features and pharmacophore requirements of target
- Identify key binding sites and essential pharmacophores

Step 2: Reference Molecule Analysis
- Reference structural features of existing target drugs
- Identify activatable scaffolds and optimizable substituents

Step 3: Structure Design
- Propose candidate scaffold structures
- Design substituent modification strategies
- Evaluate blood-brain barrier penetration

Step 4: SMILES Generation
- Generate SMILES strings meeting above requirements
- Validate SMILES validity
```

### 3.3.3 ToT Three-Stage Prompt Design

Tree-of-Thoughts reasoning prompts decompose the molecular generation process into three progressive stages:

**Stage 1: Scaffold Proposal (Propose Scaffold)**: Proposes candidate molecular scaffolds with constraints including MW range, LogP range, and BBB requirement.

**Stage 2: Assembly Strategy Proposal (Propose Assembly)**: Designs assembly strategies for the selected scaffold, including proposed substituent types, connection positions, MW contributions, and BBB penetration impact assessments.

**Stage 3: SMILES Generation (Generate SMILES)**: Generates SMILES strings meeting the above design specifications, requiring 2-3 aromatic rings and at least one polar group.

### 3.3.4 Feedback Injection Prompt Design

When the physics evaluation pipeline returns evaluation results, the prompt generator injects feedback information into prompts to guide the next round of iterative optimization:

```python
def build_feedback_injected_prompt(
    domain_prompt: str,
    physical_feedback: str,
    previous_smiles: str = "",
    next_action_hint: str = "Please continue optimizing molecular structure based on physics engine test results."
) -> str:
    return f"""{domain_prompt}

=== Physics Engine Real Test Results ===
{physical_feedback}

=== Previous Generated Molecules ===
{previous_smiles}

=== Next Step Optimization Guidance ===
{next_action_hint}
"""
```

## 3.4 Multi-Modal Molecular Generation Model

### 3.4.1 Llamole Model Architecture

Llamole is a multi-modal large language model that deeply integrates text, chemical structures, and molecular graphs, specifically designed for inverse molecular design and retrosynthetic planning. Its core architecture consists of:

**Base Language Model**: Qwen2-7B-Instruct serves as the core for language understanding and generation, with approximately 7 billion parameters, transformer decoder architecture, and instruction tuning training paradigm.

**Graph Diffusion Transformer (GraphDiT)**: The core graph structure generation component of Llamole, capable of diffusion-based generation in molecular graph space. Supports conditional generation based on target constraints.

**Graph Neural Network Encoder/Decoder**: Responsible for bidirectional conversion between molecular graphs and text space:
- Encoder: Encodes SMILES or molecular graphs into continuous vector representations, aligned with the language model space
- Decoder: Decodes LLM outputs into valid SMILES sequences or molecular graph structures

**Graph Neural Network Predictor**: Evaluates properties of generated molecules:
- DrugClip Score: Molecular-target matching score based on multi-modal contrastive learning
- Input: Molecular graph structure + target description text
- Output: Similarity score in range [0, 1]

**Graph-LM Connector**: Aligns the language model's representation space with the GNN's representation space through projection layers and cross-attention fusion.

### 3.4.2 GBM Domain Adaptation Strategy

For GBM drug design tasks, Llamole performs domain adaptation through the following strategies:

**Special Token Definition**: Defines specialized tokens for molecular generation tasks, distinguishing different types of generation content:
- <design_start>, <design_end>, <design_body>, <molecule>
- <retro_start>, <retro_end>, <retro_body>, <rollback_start>, <rollback_end>

**Model Path Configuration**:
```yaml
model_name_or_path: models/Qwen2-7B-Instruct
graph_decoder_path: saves/graph_decoder
graph_encoder_path: saves/graph_encoder
graph_predictor_path: saves/graph_predictor
adapter_name_or_path: saves/Llamole-Qwen2-7B-Instruct-Adapter-gbm-with-graph-models
graph_lm_connector_path: saves/Llamole-Qwen2-7B-Instruct-Adapter/connector
```

### 3.4.3 LoRA Fine-Tuning Strategy

Llamole uses Low-Rank Adaptation (LoRA) technology for efficient fine-tuning of the base language model to adapt to the GBM drug design domain. For a pretrained model weight matrix W₀ ∈ ℝ^(d×k), the LoRA update is:

$$W = W_0 + \Delta W = W_0 + BA$$

Where:
- B ∈ ℝ^(d×r) and A ∈ ℝ^(r×k) are trainable low-rank matrices
- r << min(d, k) is the rank parameter, set to 16 in this project
- Forward pass: h = W₀x + BAx = (W₀ + BA)x

## 3.5 Physical Evaluation Methods

### 3.5.1 Molecular Docking

#### Vina Docking Tool

AutoDock Vina is used to calculate docking affinity between generated molecules and GBM target proteins:

```python
class VinaDockingTool:
    def __init__(self, receptor_pdb_path: str, center: Tuple[float, float, float],
                 box_size: Tuple[float, float, float] = (22.5, 22.5, 22.5),
                 exhaustiveness: int = 32):
        self.receptor_pdb_path = receptor_pdb_path
        self.center = center
        self.box_size = box_size
        self.exhaustiveness = exhaustiveness

    def dock(self, smiles: str) -> float:
        # Step 1: SMILES to PDBQT
        pdbqt_content = self._smiles_to_pdbqt(smiles)
        # Step 2: Run Vina docking
        result = subprocess.run(["vina", ...])
        # Step 3: Parse results
        best_score = self._parse_vina_output(result.stdout)
        return best_score
```

#### Docking Box Parameters

| Target | Center (x, y, z) | Box Size (Å) | Description |
|--------|-------------------|--------------|-------------|
| EGFR (PDB: 1M17) | (-52.5, 15.2, 58.3) | (22.5, 22.5, 22.5) | EGFR kinase domain ATP binding site |
| VEGFR2 (PDB: 3HNG) | (-24.2, 54.6, 52.1) | (22.5, 22.5, 22.5) | VEGFR2 hydrophobic pocket |
| IDH1 R132H (PDB: 5KIW) | (12.3, 8.7, -5.2) | (22.5, 22.5, 22.5) | IDH1 mutant active site |

### 3.5.2 ADMET Prediction

#### DILI Prediction

Drug-Induced Liver Injury (DILI) prediction is based on SMARTS pattern matching and molecular descriptor calculation:

```python
class ADMETCalculator:
    DILI_RISK_PATTERNS = [
        ("acyl_halide", "[CX3](=O)[#9,#17,#35,#53]"),
        ("aldehyde", "[CX3H1]=O"),
        ("alpha_beta_unsaturated", "[CX3]=[CX3][CX3](=O)"),
        ("epoxide", "[OX2R][CX4R][CX4R]"),
        ("hydrazine", "[#7][#7]"),
        ("michael_acceptor", "[CX3]=[CX3]!@[CX3](=O)[OX2]"),
    ]

    def calculate_dili_risk(self, smiles: str) -> float:
        risk_score = 0.0
        for name, pattern in self.DILI_RISK_PATTERNS:
            if mol.HasSubstructMatch(PatternBondPatternRenderer(pattern)):
                risk_score += 0.15
        # MW and LogP penalties
        return min(risk_score, 1.0)
```

#### hERG Prediction

hERG channel blockade risk (cardiac QT prolongation) is a key safety indicator for CNS drugs:

```python
    HERG_RISK_PATTERNS = [
        ("planar_aromatic", "[a]1[a][a][a][a][a]1"),
        ("nitrogen_heterocycle", "[nR1]1[cR1][cR1][cR1][cR1][cR1]1"),
        ("tertiary_amine", "[NX3H0]([CX4])([CX4])[CX4]"),
    ]

    # EGFR warhead exemption rules (acrylamide covalent warheads)
    warheads = ["cC(=O)C=Cl", "C(C=NC#N)CC", "cC(=O)C=CN"]
```

#### BBB Permeability Prediction

BBB permeability prediction is based on molecular property descriptors and the Clark formula:

```python
    def calculate_bbb_permeability(self, smiles: str) -> float:
        bbb_score = (
            0.193 * logp
            - 0.267 * tpsa / 100
            - 0.742 * num_hbd
            + 0.571 * num_hba
            - 0.664 * num_rotatable
        )
        return bbb_score
```

### 3.5.3 MPO Reward Comprehensive Scoring

MPO (Multi-Parameter Optimization) Reward integrates multiple pharmacokinetic and pharmacodynamic parameters to calculate a weighted geometric mean comprehensive score:

```python
class MPORewardCalculator:
    REWARD_WEIGHTS = {
        "vina": 0.30,
        "dili": 0.20,
        "herg": 0.20,
        "bbb": 0.20,
        "physchem": 0.10
    }

    def calculate_mpo_reward(self, eval_result: PhysicalEvaluationResult) -> float:
        vina_norm = self._normalize_vina(eval_result.vina_score)
        dili_norm = self._normalize_risk(eval_result.dili_risk)
        herg_norm = self._normalize_risk(eval_result.herg_risk)
        bbb_norm = self._normalize_bbb(eval_result.bbb_permeability)
        physchem_norm = self._normalize_physchem(eval_result)

        # Weighted geometric mean
        mpo_reward = 1.0
        for score, weight in zip(scores, weights):
            mpo_reward *= (score ** weight)
        return mpo_reward
```

### 3.5.4 Hard Pruning and Delta Scoring

#### Hard Pruning Mechanism

The hard pruning mechanism performs forced filtering of molecules that do not meet basic drug design requirements based on preset strict thresholds:

```python
    HARD_CUTOFFS = {
        "vina_score": {"threshold": -5.0, "operator": "<"},
        "dili_risk": {"threshold": 0.95, "operator": "<"},
        "herg_risk": {"threshold": 0.95, "operator": "<"},
        "molecular_weight": {"min": 300, "max": 600},
        "tpsa": {"min": 40, "max": 140},
        "logp": {"min": 0, "max": 5},
    }
```

#### Delta Scoring Mechanism

Delta Scoring is used to dynamically adjust the strictness of toxicity evaluation during ToT search:

```python
    DELTA_SCORING = {
        "enabled": True,
        "toxicity_relaxation": 0.05,
        "max_iterations": 3,
    }

    def apply_delta_scoring(self, current_result, parent_result, depth):
        delta_dili = current_result.dili_risk - parent_result.dili_risk
        if delta_dili < self.DELTA_SCORING["toxicity_relaxation"]:
            relaxation_factor = min(depth * 0.02, 0.1)
            adjusted_dili = parent_result.dili_risk + (delta_dili * (1 - relaxation_factor))
            return adjusted_dili, True
        return current_result.dili_risk, False
```

## 3.6 ToT Three-Stage Search Algorithm

### 3.6.1 Algorithm Overview

Tree-of-Thoughts (ToT) search algorithm models the molecular generation process as a multi-way tree, exploring chemical space through Breadth-First Search (BFS) across three progressive stages:

```
                    [ROOT: GBM Target Selection]
                              |
                    [Stage 1: Scaffold Proposal]
                    BFS width=3, select Top-1
                              |
                    [Stage 2: Assembly Strategy]
                    BFS width=3, select Top-1
                              |
                    [Stage 3: SMILES Generation]
                    Sample N SMILES
                              |
                    [Physical Evaluation]
                    Vina + ADMET + MPO
                              |
                    [Hard Cutoffs + Ranking]
```

### 3.6.2 Algorithm Pseudocode

```python
def tot_search(
    target_name: str,
    constraints: Dict,
    max_depth: int = 3,
    breadth_per_stage: int = 3,
    num_smiles_per_leaf: int = 5,
    physical_evaluator: GBMPhysicalEvaluator,
    prompt_generator: GBMPromptGenerator
) -> List[Tuple[str, PhysicalEvaluationResult]]:

    # Stage 1: Scaffold Proposal
    scaffold_prompt = prompt_generator.build_tot_propose_prompt(
        domain_prompt, scaffold_state, "scaffold"
    )
    scaffold_responses = model.sample(scaffold_prompt, num_samples=breadth_per_stage)
    scaffolds = [parse_scaffold_from_response(resp) for resp in scaffold_responses]
    best_scaffold = select_top_k(scaffolds, k=1)[0]

    # Stage 2: Assembly Strategy
    assembly_prompt = prompt_generator.build_tot_propose_prompt(
        domain_prompt, assembly_state, "assembly"
    )
    assembly_responses = model.sample(assembly_prompt, num_samples=breadth_per_stage)
    assembly_strategies = [parse_assembly_from_response(resp) for resp in assembly_responses]
    best_strategy = select_top_k(assembly_strategies, k=1)[0]

    # Stage 3: SMILES Generation
    smiles_prompt = prompt_generator.build_tot_propose_prompt(
        domain_prompt, smiles_state, "smiles"
    )
    smiles_responses = model.sample(smiles_prompt, num_samples=num_smiles_per_leaf)
    all_smiles = []
    for resp in smiles_responses:
        parsed_smiles = parse_smiles_from_response(resp)
        all_smiles.extend(parsed_smiles)

    # Physical Evaluation
    for smiles in all_smiles:
        eval_result = physical_evaluator.evaluate(smiles)
        if eval_result.passed_hard_cutoffs:
            all_candidates.append((smiles, eval_result))

    # Ranking
    ranked_candidates = sorted(all_candidates, key=lambda x: x[1].mpo_reward, reverse=True)
    return ranked_candidates
```

## 3.7 Chapter Summary

This chapter has provided a detailed exposition of the core methods for the GBM candidate molecule generation and optimization system based on multi-modal large language models. First, a multi-modal knowledge base covering 7 key GBM therapeutic targets was constructed, integrating target structural information, clinical treatment data, and approved drug molecular structures, providing the model with rich domain knowledge. Second, a specialized prompt generator was designed to support CoT and ToT dual-mode reasoning, capable of dynamically constructing high-quality molecular design prompts based on target specificity and constraint conditions. Third, the Llamole multi-modal large language model was adapted to the domain through LoRA technology, enabling it to generate molecular SMILES meeting GBM drug design requirements. Finally, a complete physics evaluation pipeline was constructed, including AutoDock Vina molecular docking, ADMET prediction, MPO Reward comprehensive scoring, hard pruning mechanism, and Delta Scoring strategy, achieving closed-loop optimization from generation to evaluation. The ToT three-stage search algorithm achieves effective balance between molecular generation quality and diversity through progressive exploration of scaffold proposal, assembly strategy, and SMILES generation stages.

---

# Chapter 4 Experiments and Results

## 4.1 Introduction

This chapter conducts systematic experimental validation of the proposed method on three representative GBM targets: EGFR, VEGF/VEGFR, and IDH1/IDH2. First, the experimental environment, evaluation metrics, and comparison baselines are introduced. Then, main experimental results are presented, analyzing the performance of generated molecules on DrugClip Score, Vina docking score, BBB permeability, and other metrics. Comparison with existing methods is then discussed, followed by ablation experiments quantifying the contribution of each module. Finally, in-depth discussion of the results is provided.

## 4.2 Experimental Setup

### 4.2.1 Experimental Environment

**Hardware Environment**:
- GPU: RTX 3090, 24GB

**Software Environment**:
- Python 3.11
- PyTorch (with BF16 inference support)
- RDKit (molecular processing and descriptor calculation)
- Transformers (HuggingFace model loading)
- PEFT (LoRA fine-tuning)

### 4.2.2 Model Configuration

**Base Model**: Qwen2-7B-Instruct

**LoRA Fine-tuning Adapter**: saves/Llamole-Qwen2-7B-Instruct-Adapter-gbm-with-graph-models

**Model Loading Configuration**:
```yaml
model_name_or_path: models/Qwen2-7B-Instruct
adapter_name_or_path: saves/Llamole-Qwen2-7B-Instruct-Adapter-gbm-with-graph-models
graph_lm_connector_path: saves/Llamole-Qwen2-7B-Instruct-Adapter/connector
graph_encoder_path: saves/graph_encoder
graph_predictor_path: saves/graph_predictor
flash_attn: disabled
bf16: true
```

**Generation Hyperparameters**:
| Parameter | Value | Description |
|-----------|-------|-------------|
| max_length | 1024 | Maximum input+output token count |
| max_new_tokens | 256 | Maximum generated tokens |
| temperature | 0.7 | Sampling temperature |
| top_p | 0.9 | Nucleus sampling |
| do_sample | True | Enable sampling |
| num_return_sequences | 5 | Candidate count per generation |

### 4.2.3 Experimental Targets and Data

**Validation Targets** (3 representative GBM targets):
| Target | Description |
|--------|-------------|
| EGFR | Most common GBM target (~40%), tyrosine kinase inhibitor design |
| VEGF/VEGFR | Anti-angiogenesis strategy |
| IDH1/IDH2 | Metabolic target (~10% GBM) |

**Molecules per target**: Approximately 100 per target

**Molecular Constraint Conditions**:
| Constraint | Minimum | Maximum | Description |
|------------|---------|---------|-------------|
| MW | 200 | 700 | Molecular weight range |
| LogP | 1.0 | 4.5 | Lipid-water partition coefficient |
| TPSA | 40 | 120 | Topological polar surface area |
| HBD | 1 | 5 | Hydrogen bond donor count |
| HBA | 3 | 10 | Hydrogen bond acceptor count |
| Rotatable bonds | - | 8 | Conformational flexibility |

### 4.2.4 Evaluation Metrics

**Comprehensive Score (TotalScore)**: The core evaluation metric for generated molecules, calculated as:

```
DrugClip_norm = max(DrugClip, 0) / 3.0
Comprehensive Score = 0.30 × DrugClip_norm + 0.30 × vina_norm
                    + 0.15 × bbb_norm + 0.08 × (1-dili_prob)
                    + 0.07 × (1-pains_alert_score) + 0.05 × (1-herg_prob)
                    + 0.05 × SA_score
```

**DrugClip Score**: A molecular-target binding activity prediction model based on contrastive learning. Score range approximately [-3, +3], positive values indicate predicted activity.

**Vina Dock Score**: AutoDock Vina docking score (kcal/mol), more negative indicates stronger binding:
- Vina < -7.0 kcal/mol: Strong binding
- -7.0 ≤ Vina < -5.0 kcal/mol: Moderate binding
- Vina ≥ -5.0 kcal/mol: Weak binding (triggers hard pruning)

**Scoring Layers**:
| Layer | Score Range | Meaning |
|-------|-------------|---------|
| High Priority | Score ≥ 80 | Excellent comprehensive metrics, recommend priority validation |
| Suggest Validation | 70 ≤ Score < 80 | Good comprehensive metrics, worth further study |
| Retain for Observation | 60 ≤ Score < 70 | Basically meets requirements, needs further optimization |
| Priority Elimination | Score < 60 | Multiple indicators substandard |

## 4.3 Main Experimental Results

### 4.3.1 Comprehensive Score Distribution

Figure 4.1 shows the comprehensive score distribution histogram of generated molecules. The results indicate:
- EGFR and IDH1 comprehensive score means are both approximately 58 points
- VEGF mean is closer to 64 points, with median also higher than other groups, indicating more balanced overall candidate quality
- EGFR and IDH1 have narrower distribution widths, while VEGF has larger distribution span

### 4.3.2 DrugClip Score Distribution

Figure 4.2 shows the DrugClip Score distribution histogram for the three targets. The red dashed line marks the DrugClip=0 threshold.

| Target | DrugClip>0 Molecules | Mean | Std Dev | Max Value |
|--------|---------------------|------|---------|-----------|
| EGFR | 43 | 0.57 | 0.37 | 1.61 |
| IDH1 | 67 | 0.93 | 0.60 | 2.47 |
| VEGF | 48 | 0.62 | 0.38 | 1.74 |
| **Total** | **158** | **0.74** | - | - |

Key finding: IDH1 has the highest DrugClip mean (0.93), the most outstanding predicted activity among the three targets. The highest DrugClip value (2.47) also appears at the IDH1 target.

### 4.3.3 Vina Docking Score Distribution

Figure 4.3 shows the Vina docking score box plot for DrugClip>0 molecules. The horizontal dashed line marks the Vina=-6.0 kcal/mol strong binding reference line.

| Target | Vina Mean (kcal/mol) | Std Dev | Vina<-7.0 Ratio |
|--------|----------------------|---------|-----------------|
| EGFR | -5.78 | 2.67 | - |
| IDH1 | -7.82 | 1.73 | - |
| VEGF | -5.34 | 4.88 | - |

Key finding: IDH1 Vina mean (-7.82 kcal/mol) has crossed the -7.0 strong binding reference line, the only target achieving strong binding standard at the physical binding level. VEGF Vina mean is the weakest (-5.34 kcal/mol) with extremely large fluctuation (std dev 4.88), showing obvious positive outliers.

### 4.3.4 DrugClip vs Vina Correlation Analysis

Figure 4.4 shows a 2D scatter plot with DrugClip Score as x-axis and Vina docking score as y-axis for DrugClip>0 molecules.

Key findings:
- Most active molecules fall in the upper-left region (high DrugClip + strong Vina), indicating the model can generate molecules with both predicted activity and physical binding affinity
- Weak negative correlation exists between DrugClip and Vina; higher DrugClip scores tend to cluster toward more negative (better) Vina ranges
- IDH1 orange points are most concentrated in the "high DrugClip + strong binding" region
- EGFR target has relatively lower DrugClip mean (0.57), indicating some gap between generating parseable molecules and generating predicted active molecules

### 4.3.5 BBB Permeability Pass Rate

Figure 4.5 shows the BBB permeability pass rate for DrugClip>0 molecules using a stacked bar chart.

| Target | DrugClip>0 Molecules | BBB Passed | BBB Pass Rate |
|--------|---------------------|------------|---------------|
| EGFR | 99 | 47 | 47.5% |
| IDH1_IDH2 | 100 | 45 | 45.0% |
| VEGF_VEGFR | 99 | 55 | 55.6% |
| **Total** | **298** | **147** | - |

Key finding: VEGF_VEGFR has the highest BBB pass rate (55.6%), possibly related to the relatively smaller MW and TPSA of molecules generated for this target. None of the three targets exceed 60% BBB pass rate, indicating BBB penetration remains the core challenge in GBM molecular design.

### 4.3.6 TPSA vs LogP Physicochemical Space Distribution

Figure 4.6 shows scatter plots with TPSA as x-axis and Consensus LogP as y-axis for DrugClip>0 molecules.

Key findings:
- BBB-passing molecules are mainly concentrated in the ideal CNS drug physicochemical space (TPSA 40-90 Å², LogP 1-4)
- Molecules falling in the TPSA>140 Å² "death line" region are few, indicating the physics evaluation module effectively constrains overly polar molecules
- EGFR target molecules have overall smaller physicochemical properties (TPSA mean 63.96), favorable for BBB penetration

### 4.3.7 Comprehensive Score Layer Distribution

Figure 4.7 shows the scoring layer distribution for generated molecules using a stacked bar chart.

Key findings:
- EGFR target: Score≥80 ~0%, Score≥70 22%
- IDH1 target: Score≥80 10%, Score≥70 21%
- VEGF target: Score≥80 12%, Score≥70 42%
- VEGF has the highest Score≥70 ratio (~42%), with more balanced overall candidate quality

## 4.4 Ablation Experiment Results

### 4.4.1 Ablation Experiment Design

Six ablation experiments were designed to quantify the contribution of each core module:

| Exp ID | Experiment Name | Ablation Content |
|--------|-----------------|------------------|
| Ablation-1 | No CoT | Remove CoT reasoning template, use only basic prompts |
| Ablation-2 | No Feedback | Remove physics engine feedback, ToT search without feedback closed loop |
| Ablation-3 | No Physical Eval | Remove entire physics evaluation |
| Ablation-4 | No Domain Knowledge | Remove GBM knowledge base injection |
| Ablation-5 | No GBM LoRA | Use only general Llamole adapter, without GBM LoRA |
| Ablation-6 | Baseline | No adapters, directly use Qwen2-7B-Instruct |

### 4.4.2 DrugClip>0 Molecule Ratio Comparison

Figure 4.12 shows the DrugClip>0 molecule ratio comparison.

| Method | DrugClip>0 Molecule Ratio |
|--------|---------------------------|
| Llamole (This thesis) | 53.0% |
| LlaSMol-Mistral-7B | 39% |
| Top-K | 38% |

### 4.4.3 Top Molecule Quality Analysis

Figures 4.9 and 4.10 show ablation experiment results.

Key findings:
1. Removing physics evaluation (32.4%), removing CoT reasoning (25.7%), removing CoT&ToT (24.3%) cause the most significant damage
2. Baseline model (no adapters) DrugClip mean is only 0.40, a 45.9% decrease from Full Model's 0.74, indicating domain adaptation is crucial
3. IDH1 is most sensitive to physics evaluation removal (0.93→0.47, 49.5% decrease), indicating this target highly depends on physics evaluation for screening and guidance

## 4.5 Comparison Experiments

For validation of the proposed method's effectiveness, comparison with three representative molecular generation methods:

| Comparison Method | Description |
|-------------------|-------------|
| LlaSMol-Mistral-7B | Mistral-7B-based molecular generation model |
| PlayMolecule AI | Large language model-based molecular design system |
| Molecule-oriented LLaMA-7b | LLaMA model fine-tuned for molecular description→SMILES task |
| Llamole Qwen-7b | Large language model trained for retrosynthesis path planning |

Experimental results:
- PlayMolecule AI generates molecules with scores higher than the proposed method in some cases, but generated molecules show obvious scaffold convergence. Although candidate molecules are all valid and non-repetitive SMILES, their Murcko scaffold is highly consistent, with main differences concentrated in terminal substituents and aromatic ring substitution positions, indicating the model tends to perform local structure fine-tuning around known EGFR inhibitor-like scaffolds
- Molecule-oriented LLaMA-7b mainly trains on "molecular description→SMILES" tasks, with poor ability for direct target-based molecular generation
- Llamole Qwen-7b focuses on training molecular retrosynthesis path planning, with poor ability for direct target-based molecular generation

## 4.6 Result Analysis

### 4.6.1 Analysis of Domain Knowledge Injection Effect

1. Contribution to DrugClip mean: After removing domain knowledge, three-target DrugClip mean decreases from 0.74 to 0.67 (9.5% decrease). Notably, EGFR DrugClip mean increases from 0.57 to 0.60 after removing domain knowledge, indicating target-specific contribution of domain knowledge to DrugClip score.

2. Contribution to Vina binding ability: After removing domain knowledge, EGFR Vina mean changes from -5.78 to -6.12 kcal/mol (slight improvement), VEGF from -5.34 to -5.16 kcal/mol (slight degradation).

3. Differential impact on different targets: IDH1 is most sensitive to domain knowledge removal (DrugClip mean decreases 16.1%), while VEGF shows almost no change (0.0%).

### 4.6.2 Analysis of CoT-ToT Reasoning Strategy Effect

1. Role of CoT reasoning: After removing CoT reasoning, DrugClip mean decreases from 0.74 to 0.55 (25.7% decrease), indicating CoT reasoning effectively improves DrugClip activity prediction scores of generated molecules through 7 structured steps.

2. Role of ToT feedback closed loop: After removing physics feedback, DrugClip mean decreases from 0.74 to 0.59 (20.3% decrease), with VEGF DrugClip mean significantly decreasing from 0.62 to 0.55 (11.3%).

3. Synergistic effect of CoT and ToT: CoT more affects "what structure to generate," while ToT feedback more affects "whether generated structure meets physical constraints." Only their collaboration achieves the leap from "can generate" to "can generate high-quality DrugClip activity."

### 4.6.3 Analysis of Physics Evaluation System Necessity

1. Severe degradation after removing physics evaluation: After removing physics evaluation, DrugClip mean significantly decreases from 0.74 to 0.50 (32.4% decrease, maximum). Vina mean degrades on all three targets.

2. Differential impact on different targets: IDH1 is most sensitive to physics evaluation removal (DrugClip mean decreases 49.5%).

3. Delta Scoring mechanism: Allows molecules with toxicity delta ≤0.05 to temporarily not trigger hard pruning, preserving exploration opportunities. This design expands chemical space for molecular generation while maintaining safety bottom line.

### 4.6.4 Analysis of GBM LoRA Fine-tuning Effect

1. Contribution to DrugClip mean: After removing GBM LoRA, DrugClip mean decreases from 0.74 to 0.66 (10.8% decrease).

2. Differential impact on different targets: EGFR is most sensitive to GBM LoRA removal (DrugClip mean decreases 13.2%), followed by VEGF (decreases 4.8%).

3. Validating importance of LoRA fine-tuning: GBM LoRA adapter introduction increases DrugClip mean by approximately 10.8%, indicating GBM domain LoRA fine-tuning plays an important role in the model's ability to generate high-quality GBM candidate molecules.

## 4.7 Chapter Summary

This chapter has conducted systematic experimental validation of the GBM candidate molecule generation framework on three representative GBM targets: EGFR, VEGF/VEGFR, and IDH1/IDH2. Main conclusions are as follows:

**Main Experimental Results** (DrugClip>0 molecules):
- IDH1 performs best: DrugClip mean 0.93 (highest), Vina mean -7.82 kcal/mol (only one crossing -7.0 strong binding line)
- VEGF BBB advantage is significant: BBB pass rate 55.6% (highest), favorable for CNS drug design
- Three-target comprehensive: DrugClip mean 0.74, Vina mean -6.31 kcal/mol, BBB pass rate 49.3%

**Ablation Experiment Quantitative Contributions** (DrugClip mean decrease %):
- No physical evaluation: 24.0% decrease (largest)
- No feedback: 20.1% decrease
- No CoT reasoning: 15.8% decrease
- No GBM LoRA: 10.8% decrease
- No domain knowledge: 9.5% decrease
- Baseline model: 45.9% decrease

These results indicate that the proposed multi-module collaborative framework can effectively integrate domain knowledge, reasoning capabilities, and physics verification, thereby generating GBM candidate molecules with good drug-likeness potential.

---

# Chapter 5 Summary and Outlook

## 5.1 Research Summary

### 5.1.1 Core Work Summary

This thesis focuses on the core question of "how to efficiently generate candidate drug molecules with GBM therapeutic potential using multi-modal large language models," conducting systematic research from three dimensions: multi-modal knowledge integration, controllable molecular generation, and physics verification closed loop.

**Work 1: GBM Multi-Modal Knowledge Base Construction and Injection**

This thesis constructed a GBM multi-modal knowledge base covering target biology information, clinical treatment experience, and molecular structure knowledge, achieving efficient knowledge transfer to generation models through domain LoRA fine-tuning. Compared to general-purpose molecular generation models, this knowledge base enables the model to perform target-specific molecular design for GBM molecular features such as EGFR amplification and IDH mutations.

**Work 2: CoT-ToT Collaborative Reasoning and Physics Verification Closed-Loop Multi-Stage Generation Strategy**

This thesis innovatively combined Chain-of-Thought reasoning with Tree-of-Thoughts search, constructing a closed-loop optimization system with quantitative feedback from the physics engine. CoT reasoning guides the model through 7 structured steps, while ToT search achieves efficient exploration of generation space through three-stage BFS (scaffold proposal → fragment assembly → SMILES generation). Physics evaluation feedback injects quantitative indicators including Vina docking, DILI/hERG risk, and BBB scores into the next round of prompts, achieving data-driven structure optimization guidance.

**Work 3: Rapid Drug-Likeness Evaluation System Based on Structural Alert Detection and Clark Formula**

This thesis designed a drug-likeness evaluation system that can run without external model APIs. Structural alert detection (based on SMARTS pattern matching) can complete toxicity group screening at millisecond level, Clark formula with heuristic penalties can complete BBB permeability prediction at microsecond level, and MPO Reward achieves multi-objective balance through weighted geometric mean.

### 5.1.2 Main Conclusions

Through systematic experimental validation on three representative GBM targets—EGFR, VEGF/VEGFR, and IDH1/IDH2—this thesis draws the following main conclusions:

**Conclusion 1**: The proposed multi-modal LLM-based GBM candidate molecule generation framework can effectively generate structurally novel candidate drug molecules with some BBB permeability and target binding activity. Under Full Model configuration, for DrugClip>0 molecules (predicted active): DrugClip mean 0.74, Vina mean -5.74 kcal/mol, BBB pass rate 49.3%.

**Conclusion 2**: IDH1 target performs most outstandingly. DrugClip mean reaches 0.93 (highest among three targets), Vina mean reaches -7.35 kcal/mol, 74.0% of molecules reach strong binding standard (Vina<-6.0 kcal/mol), reflecting mature application potential of the model on IDH1 target.

**Conclusion 3**: VEGF target has significant BBB permeability advantage. BBB pass rate reaches 55.6% (highest among three targets), favorable for CNS drug design requirements.

**Conclusion 4**: Ablation experiments quantify contributions of each core module. After removing physics evaluation, DrugClip mean decreases 24.0% (largest decrease); removing feedback decreases 20.1%; removing CoT reasoning decreases 15.8%, validating necessity of physics evaluation feedback, feedback closed loop, and CoT reasoning modules.

**Conclusion 5**: GBM domain LoRA fine-tuning is crucial for maintaining Vina binding ability. After removing GBM LoRA, Vina mean degrades from -5.74 to positive value 0.23 kcal/mol, indicating LoRA fine-tuning plays an irreplaceable role in maintaining molecular physics binding ability.

**Conclusion 6 (Comprehensive Positioning)**: Core advantages of this framework: ① Strong binding ability on IDH1 target (-7.82 kcal/mol); ② BBB permeability advantage on VEGF target (55.6% pass rate); ③ Multi-objective collaborative optimization capability (DrugClip, Vina, BBB three-dimensional overall balance).

## 5.2 Research Limitations

### 5.2.1 Experimental Scale Limitations

- **Limited target coverage**: This thesis only validated on three targets—EGFR, VEGF/VEGFR, and IDH1/IDH2—without covering all GBM-related targets (e.g., GBM stem cells, PD-1/PD-L1, PI3K/AKT/mTOR, etc.)
- **Limited generated molecule quantity**: Only approximately 100 candidate molecules per target were generated, with small sample size and statistical significance needing improvement
- **Lack of wet lab validation**: Generated candidate molecules were not validated through in vitro cell experiments or animal experiments, with clinical transformation potential unknown

### 5.2.2 Technical Limitations

- **Limited prediction accuracy in ToT process**: Clark formula is an empirical model based on descriptors, with prediction accuracy inferior to deep learning-based BBB prediction models
- **Lack of molecular novelty evaluation**: Quantitative analysis of structural novelty between generated molecules and known drugs was not performed
- **Multi-target collaborative design not implemented**: Current framework only supports single-target molecular generation, without implementing multi-target collaborative inhibitor molecule design

### 5.2.3 Methodological Limitations

- **Limited ToT search depth**: Current ToT search only supports 3 stages, with possibly insufficient exploration depth for complex molecular structures
- **Fixed Delta Scoring thresholds**: The toxicity delta ≤0.05 exemption threshold in Delta Scoring is empirically set, possibly not applicable to all molecular types
- **Physics engine feedback delay**: Physics evaluation is only executed after SMILES generation, without evaluation guidance introduction at scaffold proposal and fragment assembly stages

## 5.3 Future Research Outlook

### 5.3.1 Experimental Scale Expansion

**Expanding target coverage**: Conduct validation on all GBM-related targets (including GBM stem cells, PD-1/PD-L1, PI3K/AKT/mTOR, p53/MDM2), and explore multi-target collaborative inhibitor molecule generation methods.

**Increasing generated molecule scale**: Increase generated molecules per target to 800-1000, accumulate sufficient data for statistical significance analysis, and establish candidate molecule database.

**Introducing wet lab validation**: Collaborate with School of Pharmacy to perform in vitro cell activity experiments (IC50 determination), hERG blockade experiments, and BBB permeability experiments on top candidate molecules to verify computational prediction results.

### 5.3.2 Technical Improvement

**Introducing deep learning BBB prediction models**: Replace Clark formula with CNN or GNN-based BBB prediction models to improve prediction accuracy. References include ADMETlab, pkADMET, and other tools.

**Molecular novelty metrics**: Introduce molecular scaffold novelty and BRICS fragment novelty analysis to quantitatively evaluate structural differences between generated molecules and training data.

**Multi-target collaborative design**: Explore multi-target molecular design methods based on multi-constraint conditions, or design dual-specificity molecular generation strategies.

### 5.3.3 Methodological Innovation

**Adaptive ToT search**: Dynamically adjust ToT search width and depth based on physics evaluation feedback, investing more computational resources in high-quality branches.

**Reinforcement learning strategy optimization**: Use MPO Reward as reinforcement learning reward signal, optimize LLM generation distribution using PPO and other algorithms to further improve high-score molecule generation ratio.

**Retrosynthesis planning integration**: Integrate Llamole's A* search retrosynthesis planning capability to assess synthetic feasibility and design routes for top candidate molecules, improving synthetizability of candidate molecules.

### 5.3.4 Clinical Transformation

**Establishing candidate molecule database**: Construct GBM candidate molecule database, recording all generation results and evaluation data to support subsequent candidate molecule screening and priority ranking.

---

# References

[A1] Stupp R, Mason W P, Van Den Bent M J, et al. Radiotherapy plus concomitant and adjuvant temozolomide for glioblastoma[J]. New England Journal of Medicine, 2005, 352(10): 987-996.

[A2] Fu M, Xue B, Miao X, et al. Overcoming immunotherapy resistance in glioblastoma: challenges and emerging strategies[J]. Frontiers in Pharmacology, 2025, 16: 1584688.

[A3] Norden A D, Drappatz J, Wen P Y. Malignant gliomas in adults[M]//Blue Books of Neurology. Butterworth-Heinemann, 2010, 36: 99-120.

[A4] Ferraris C, Cavalli R, Panciani P P, et al. Overcoming the blood–brain barrier: successes and challenges in developing nanoparticle-mediated drug delivery systems for the treatment of brain tumours[J]. International Journal of Nanomedicine, 2020: 2999-3022.

[A5] Hanahan D, Weinberg R A. Hallmarks of cancer: the next generation[J]. Cell, 2011, 144(5): 646-674.

[A6] O'Rourke D M, Nasrallah M L P, Desai A, et al. A single dose of peripherally infused EGFRvIII-directed CAR T cells mediates antigen loss and induces adaptive resistance in patients with recurrent glioblastoma[J]. Science Translational Medicine, 2017, 9(399): eaaa0984.

[A7] Heffron T P, Ndubaku C O, Salphati L, et al. Discovery of clinical development candidate GDC-0084, a brain penetrant inhibitor of PI3K and mTOR[J]. ACS Medicinal Chemistry Letters, 2016, 7(4): 351-356.

[B1] Grisoni F. Chemical language models for de novo drug design: Challenges and opportunities[J]. Current Opinion in Structural Biology, 2023, 79: 102527.

[B2] Xu M, Powers A S, Dror R O, et al. Geometric latent diffusion models for 3D molecule generation[C]//International Conference on Machine Learning. PMLR, 2023: 38592-38610.

[B3] Bagal V, Aggarwal R, Vinod P K, et al. MolGPT: molecular generation using a transformer-decoder model[J]. Journal of Chemical Information and Modeling, 2021, 62(9): 2064-2076.

[B4] Blaschke T, Arús-Pous J, Chen H, et al. REINVENT 2.0: an AI tool for de novo drug design[J]. Journal of Chemical Information and Modeling, 2020, 60(12): 5918-5922.

[B5] Zhou Z, Kearnes S, Li L, et al. Optimization of molecules via deep reinforcement learning[J]. Scientific Reports, 2019, 9(1): 10752.

[B6] Liu G, Xu J, Luo T, et al. Graph Diffusion Transformers for Multi-Conditional Molecular Generation[C]//NeurIPS, 2024.

[C1] Liu G, Sun M, Matusik W, et al. Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning[J]. arXiv:2410.04223, 2024.

[C2] Jin C, Guo S, Zhou S, et al. Effective and Explainable Molecular Property Prediction by Chain-of-Thought Enabled Large Language Models and Multi-Modal Molecular Information Fusion[J]. Journal of Chemical Information and Modeling, 2025, 65(11): 5438-5455.

[C3] Liu P, Ren Y, Tao J, et al. Git-mol: A multi-modal large language model for molecular science with graph, image, and text[J]. Computers in Biology and Medicine, 2024, 171: 108073.

[C4] Li J, Zhang D, Wang X, et al. ChemVLM: Exploring the power of multimodal large language models in chemistry area[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2025, 39(1): 415-423.

[C5] Li K, Wu Z, Wang S, et al. DrugPilot: LLM-based parameterized reasoning agent for drug discovery[J]. arXiv:2505.13940, 2025.

[D1] Trott O, Olson A J. AutoDock Vina: improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading[J]. Journal of Computational Chemistry, 2010, 31(2): 455-461.

[D2] Clark D E, Delany S G. Rapid calculation of polar surface area[C]//ACS Publications, 2000.

[D3] Wu Z, Ramsundar B, Feinberg E N, et al. MoleculeNet: a benchmark for molecular machine learning[J]. Chemical Science, 2018, 9(2): 513-530.

[D4] Daina A, Michielin O, Zoete V. SwissADME: a free web tool to evaluate pharmacokinetics, drug-likeness and medicinal chemistry friendliness of small molecules[J]. Scientific Reports, 2017, 7(1): 42717.

[E1] Wei J, Wang X, Schuurmans D, et al. Chain-of-thought prompting elicits reasoning in large language models[C]//NeurIPS, 2022.

[E2] Yao S, Yu D, Zhao J, et al. Tree of thoughts: Deliberate problem solving with large language models[C]//NeurIPS, 2023.

[F1] Hu E J, Shen Y, Wallis P, et al. LoRA: Low-rank adaptation of large language models[J]. arXiv:2106.09685, 2021.

[F2] Dettmers T, Pagnoni A, Holtzman A, et al. QLoRA: Efficient finetuning of quantized LLMs[C]//NeurIPS, 2023.

[G1] Xu K, Hu W, Leskovec J, et al. How powerful are graph neural networks?[C]//ICLR, 2019.

[G2] Gilmer J, Schoenholz S S, Riley P F, et al. Neural message passing for quantum chemistry[C]//ICML, 2017: 1263-1272.

---

*End of English Version*
