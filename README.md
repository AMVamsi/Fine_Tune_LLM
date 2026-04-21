# LLM Fine-Tuning for Knowledge Graph Extraction & Clinical Summarization

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace">
  <img src="https://img.shields.io/badge/Mistral-7B-6A0DAD?style=for-the-badge" alt="Mistral-7B">
  <img src="https://img.shields.io/badge/LLaMA2-13B-0467DF?style=for-the-badge&logo=meta&logoColor=white" alt="LLaMA2">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/LoRA-Fine--tuning-00897B?style=flat-square" alt="LoRA">
  <img src="https://img.shields.io/badge/QLoRA-4--bit Quantization-FF6F00?style=flat-square" alt="QLoRA">
  <img src="https://img.shields.io/badge/Unsloth-Efficient Training-7B68EE?style=flat-square" alt="Unsloth">
  <img src="https://img.shields.io/badge/WandB-Experiment Tracking-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black" alt="WandB">
  <img src="https://img.shields.io/badge/Neo4j-Knowledge Graph-008CC1?style=flat-square&logo=neo4j&logoColor=white" alt="Neo4j">
  <img src="https://img.shields.io/badge/spaCy-NLP-09A3D5?style=flat-square&logo=spacy&logoColor=white" alt="spaCy">
  <img src="https://img.shields.io/badge/MIMIC--IV-Clinical NLP-C62828?style=flat-square" alt="MIMIC-IV">
</p>

---

> **Research project exploring LLM fine-tuning across two domains: (1) knowledge graph triple extraction from news articles using Mistral-7B + LoRA, and (2) clinical note summarization (Brief Hospital Course) using the MIMIC-IV-BHC dataset with LLaMA2, GPT-4, and Clinical-T5.**

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Experimental Results](#experimental-results)
- [Technical Details](#technical-details)
- [Quick Start](#quick-start)
- [Dependencies](#dependencies)
- [Key Findings](#key-findings)
- [Future Work](#future-work)

---

## Overview

This project investigates parameter-efficient fine-tuning (PEFT) techniques — specifically **LoRA** and **QLoRA** — applied to two distinct NLP tasks:

| Task | Domain | Model | Dataset |
|------|--------|-------|---------|
| **Knowledge Graph Triple Extraction** | News / General NLP | Mistral-7B-Instruct | NewsKG21 |
| **Medical Entity Extraction** | Healthcare / NLP | MedCAT + SNOMED CT | Clinical text |
| **Clinical Note Summarization** | Healthcare / Clinical NLP | LLaMA2-13B, GPT-4, Clinical-T5 | MIMIC-IV-BHC |

---

## Project Structure

```
Fine_Tune_LLM/
├── Task1/
│   └── Task1.py                  # MedCAT entity extraction + Neo4j/RDF integration
├── Task2/
│   ├── newsfinetune1.py          # Case 1 — Gold Standard triples fine-tuning
│   ├── newfinetune2.py           # Case 2 — spaCy-filtered gold triples fine-tuning
│   └── newsfinetune3.py          # Case 3 — Combined gold + spaCy triples fine-tuning
├── data_preprocessing/
│   └── newKG21.py                # Preprocessing pipeline for NewsKG21 (all 3 cases)
├── mimic-iv-bhc.py               # Clinical BHC summarization pipeline
├── requirements.txt
└── TM2_Final_Report.pdf          # Full technical report
```

### Task 1 — MedCAT & SNOMED CT Integration

`Task1/Task1.py` builds a **medical entity extraction and knowledge graph storage** pipeline:

- Loads a **MedCAT** model to identify SNOMED CT-coded entities in clinical text
- Exports entity relationships into **Neo4j** via the **RDFLib** framework
- Designed for downstream clinical knowledge graph construction

### Task 2 — LLM Fine-Tuning for Knowledge Graph Extraction

Three experimental cases, each fine-tuning **Mistral-7B-Instruct** with **LoRA** (4-bit quantization):

| File | Case | Training Data |
|------|------|---------------|
| `newsfinetune1.py` | Case 1 — Gold Standard | Pure manually annotated (gold) triples |
| `newfinetune2.py` | Case 2 — Filtered | spaCy-filtered subset of gold triples |
| `newsfinetune3.py` | Case 3 — Combined | Gold triples + spaCy-extracted triples |

All cases use an **Alpaca-style instruction-input-response** prompt format and are tracked via **Weights & Biases**.

### Clinical Summarization Pipeline — MIMIC-IV-BHC

`mimic-iv-bhc.py` benchmarks **Brief Hospital Course (BHC) summarization** across:

- **Adaptation strategies**: Zero-shot prompting → Prefix prompting → QLoRA fine-tuning
- **Models compared**: Clinical-T5-Base, LLaMA2-13B-Chat, GPT-4 (with Flan-T5 as fallback)
- **Evaluation**: BERT Score (F1) across short, medium, and long clinical note bins
- **Data split**: Stratified 95% train / 5% test per context-length bin

### Data Preprocessing

`data_preprocessing/newKG21.py` processes the raw **NewsKG21** dataset into three prompt formats:

- Parses and cleans raw triple annotations
- Applies spaCy NLP for subject extraction and noise filtering
- Generates subject-conditioned prompts matched to each experimental case

---

## Experimental Results

### Knowledge Graph Triple Extraction (NewsKG21)

Fine-tuning **Mistral-7B-Instruct** on 1,501 training samples (80/20 split; 375 test samples) with evaluation on predicate-object pair extraction:

| Case | Approach | Precision | Recall | F1 Score | Characteristic |
|:----:|----------|:---------:|:------:|:--------:|----------------|
| 1 | Gold Standard | 0.2071 | 0.2057 | **0.2064** | Balanced; limited by dataset size |
| 2 | spaCy Filtered | **0.2245** | 0.1892 | 0.2058 | Highest precision; conservative recall |
| 3 | Combined | 0.2120 | 0.2099 | **0.2110** | Best overall F1; balanced approach |

> **Best F1**: Case 3 (Combined) — `0.2110`  
> **Best Precision**: Case 2 (spaCy Filtered) — `0.2245`

### Clinical Summarization (MIMIC-IV-BHC) — BERT F1 Scores

| Model | Zero-shot | Prefix Prompting | QLoRA Fine-tuning |
|-------|:---------:|:----------------:|:-----------------:|
| Clinical-T5-Base | 0.584 | 0.601 | **0.647** |
| LLaMA2-13B-Chat | 0.612 | 0.629 | **0.683** |
| GPT-4 | 0.667 | **0.673** | N/A |

> **Key finding**: QLoRA fine-tuning yields substantial gains — **+6.3%** for Clinical-T5, **+7.1%** for LLaMA2.  
> LLaMA2-13B-Chat with QLoRA (`0.683`) outperforms the domain-specialized Clinical-T5 model.

---

## Technical Details

### Knowledge Graph Extraction Stack

| Component | Choice |
|-----------|--------|
| Base model | Mistral-7B-Instruct |
| Quantization | 4-bit (bitsandbytes) |
| PEFT method | LoRA (Low-Rank Adaptation) |
| Training framework | Unsloth + TRL SFTTrainer |
| Experiment tracking | Weights & Biases |
| Evaluation metric | Precision / Recall / F1 |

### Clinical Summarization Stack

| Component | Choice |
|-----------|--------|
| Models evaluated | Clinical-T5-Base, LLaMA2-13B-Chat, GPT-4 |
| PEFT method | QLoRA (4-bit quantization) |
| Training framework | Unsloth |
| Evaluation metric | BERT Score (semantic F1) |
| Dataset | MIMIC-IV-BHC (2,000 train / 100 test per context bin) |

---

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> **GPU required**: CUDA-capable GPU with ≥16 GB VRAM recommended for Mistral-7B / LLaMA2.

---

### Task 1 — MedCAT Entity Extraction

```bash
cd Task1
python Task1.py
```

Requires a local MedCAT model pack (SNOMED CT). Update `model_path` in the script before running.

---

### Task 2 — Knowledge Graph Fine-Tuning

```bash
# Step 1: Preprocess the NewsKG21 dataset
cd data_preprocessing
python newKG21.py

# Step 2: Run fine-tuning experiments
cd ../Task2
python newsfinetune1.py   # Case 1 — Gold Standard
python newfinetune2.py    # Case 2 — spaCy Filtered
python newsfinetune3.py   # Case 3 — Combined
```

---

### Clinical Summarization — MIMIC-IV-BHC

```bash
# Full pipeline (preprocessing + training + evaluation)
python mimic-iv-bhc.py --data-file mimic-iv-bhc.csv

# Inference only (skip training)
python mimic-iv-bhc.py --skip-training --context-bin all

# Specific context-length bin
python mimic-iv-bhc.py --context-bin short --output-dir results/short
python mimic-iv-bhc.py --context-bin medium --output-dir results/medium
python mimic-iv-bhc.py --context-bin long --output-dir results/long
```

> **Note**: Access to the MIMIC-IV dataset requires a [PhysioNet credentialed account](https://physionet.org/settings/credentialing/).

---

## Dependencies

```bash
# Core ML & LLM
pip install torch transformers datasets peft trl bitsandbytes
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"

# Clinical & NLP
pip install medcat spacy nltk bert-score openai

# Knowledge Graphs
pip install rdflib rdflib-neo4j neo4j

# Experiment tracking
pip install wandb huggingface-hub

# Utilities
pip install pandas numpy scikit-learn tqdm
python -m spacy download en_core_web_sm
```

Full pinned versions are in [`requirements.txt`](requirements.txt).

---

## Key Findings

✅ **Data filtering improves precision**: Case 2 (spaCy Filtered) achieved the highest precision (`0.2245`), demonstrating that removing noisy triples yields more conservative but accurate extractions.

✅ **Data augmentation improves F1**: Case 3 (Combined) achieved the best overall F1 (`0.2110`), showing that augmenting gold triples with spaCy-extracted ones improves recall without sacrificing precision.

✅ **QLoRA fine-tuning consistently outperforms prompting**: Both Clinical-T5 and LLaMA2 gained 6–7% BERT F1 after QLoRA fine-tuning versus their zero-shot baselines.

✅ **General-purpose LLMs can match domain-specific models**: LLaMA2-13B-Chat with QLoRA (`0.683`) surpasses Clinical-T5-Base (`0.647`) despite not being trained on clinical text.

---

## Future Work

- **Enhanced evaluation**: Fuzzy matching and semantic similarity metrics beyond strict string matching
- **Data cleaning**: Systematic correction of spelling errors and entity normalization in NewsKG21
- **Stratified splits**: Balanced train/test distributions across relation types
- **Architecture exploration**: Relation-specific BERT-based models for structured extraction
- **Data augmentation**: Synthetic training example generation via GPT-4
- **Multi-task learning**: Joint NER + relation extraction training
- **Target performance**: F1 improvement from ~0.21 to 0.35–0.40 for triple extraction

---

## Dataset Information

| Dataset | Task | Train Size | Test Size | Split |
|---------|------|:----------:|:---------:|:-----:|
| NewsKG21 (Cases 1 & 3) | KG Triple Extraction | 1,501 | 375 | 80/20 |
| NewsKG21 (Case 2) | KG Triple Extraction | 1,274 | 318 | 80/20 |
| MIMIC-IV-BHC (per bin) | Clinical Summarization | 2,000 | 100 | 95/5 |

---

## Report

The full technical report is available in [`TM2_Final_Report.pdf`](TM2_Final_Report.pdf).

---

*Author: Mohan Adluru · June 2025*

