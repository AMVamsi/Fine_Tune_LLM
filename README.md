# LLM Fine-Tuning for Knowledge Graph Triple Extraction and Clinical Summarization

This repository contains comprehensive studies on Large Language Model (LLM) fine-tuning for two distinct healthcare and NLP tasks: knowledge graph triple extraction from news articles and clinical note summarization using the MIMIC-IV-BHC dataset.

## Project Structure

### Task 1: MedCAT SNOMED CT Integration
- **`Task1/Task1.py`**: MedCAT entity extraction and Neo4j integration
  - Extracts medical entities from text using MedCAT models
  - Stores entities in Neo4j using RDF framework

### Task 2: LLM Fine-Tuning Experiments
- **`Task2/newsfinetune1.py`**: **Case 1 (Gold Standard)** - Fine-tuning on pure gold-labeled triples
- **`Task2/newfinetune2.py`**: **Case 2 (Filtered)** - Fine-tuning on spaCy-filtered gold triples  
- **`Task2/newsfinetune3.py`**: **Case 3 (Combined)** - Fine-tuning on gold + spaCy-extracted triples

### Clinical Summarization Pipeline
- **`mimic-iv-bhc.py`**: **MIMIC-IV Brief Hospital Course (BHC) Summarization Pipeline**
  - Reproduces and benchmarks clinical note summarization using MIMIC-IV-BHC dataset
  - Compares general-purpose LLMs (LLaMA2-13B-Chat, GPT-4) vs healthcare-adapted models (Clinical-T5)
  - Implements zero-shot, prefix prompting, and QLoRA fine-tuning strategies
  - Evaluates across context length bins (short, medium, long clinical notes)

### Data Preprocessing
- **`data_preprocessing/newKG21.py`**: Data preprocessing pipeline for all three experimental cases
  - Processes raw newsKG21 dataset into three different training formats
  - Implements spaCy NLP processing for subject extraction and filtering
  - Generates subject-conditioned prompts for each approach

##  Experimental Results

### Knowledge Graph Triple Extraction (NewsKG21)

| Case | Approach | Precision | Recall | F1 Score | Key Characteristics |
|------|----------|-----------|--------|----------|-------------------|
| 1 | Gold Standard | 0.2071 | 0.2057 | **0.2064** | Balanced, limited by dataset size |
| 2 | spaCy Filtered | **0.2245** | 0.1892 | 0.2058 | Highest precision, conservative |
| 3 | Combined | 0.2120 | 0.2099 | **0.2110** | Best overall F1, balanced approach |

### Clinical Summarization (MIMIC-IV-BHC) - BERT F1 Scores

| Model | Zero-shot | Prefix Prompting | QLoRA Fine-tuning |
|-------|-----------|------------------|-------------------|
| **Clinical-T5-Base** | 0.584 | 0.601 | **0.647** |
| **LLaMA2-13B-Chat** | 0.612 | 0.629 | **0.683** |
| **GPT-4** | 0.667 | **0.673** | N/A |

*Key Finding: QLoRA fine-tuning provides significant improvements (+6.3% for Clinical-T5, +7.1% for LLaMA2), with LLaMA2 outperforming specialized clinical models.*

##  Technical Details

### Knowledge Graph Extraction
- **Model**: Mistral-7B-Instruct with 4-bit quantization
- **Fine-tuning**: LoRA (Low-Rank Adaptation) for efficient parameter updates
- **Evaluation**: Precision, Recall, F1 on predicate-object pair extraction
- **Data**: NewsKG21 dataset with manually annotated triples

### Clinical Summarization
- **Models**: Clinical-T5, LLaMA2-13B-Chat, GPT-4 (with Flan-T5 fallback)
- **Framework**: Unsloth for efficient fine-tuning and inference
- **Evaluation**: BERT Score (precision, recall, F1) for semantic similarity
- **Data**: MIMIC-IV-BHC with stratified 95%/5% train/test splits across context bins

## Key Challenges

- **Data Quality Issues**: Inconsistent entity extraction, spelling errors, name variations
- **Evaluation Limitations**: Strict string matching without semantic equivalence
- **Dataset Constraints**: Limited size (1,501 samples), suboptimal train/test splits
- **Task Complexity**: Multi-hop reasoning, coreference resolution, context dependency
- **Model Architecture**: Mistral-7B sequence-to-sequence limitations for structured output

## Future Work

### Immediate Improvements
- **Enhanced Evaluation**: Implement fuzzy matching and semantic similarity metrics
- **Data Cleaning**: Systematic correction of spelling errors and entity inconsistencies
- **Better Splits**: Implement stratified sampling for balanced train/test distributions

### Advanced Enhancements
- **Model Architecture**: Explore relation-specific BERT-based architectures
- **Data Augmentation**: Generate synthetic training examples using GPT-4
- **Multi-task Learning**: Joint training on NER + relation extraction
- **Active Learning**: Iterative model improvement with human feedback

### Expected Impact
- **Target Performance**: F1 score improvement from ~0.21 to 0.35-0.40
- **Robustness**: Better handling of complex sentences and rare relations
- **Generalization**: Improved performance across different domains and text types

##  Dataset Information

- **Training Size**: 1,501 samples (Case 1 & 3), 1,274 samples (Case 2)
- **Test Size**: 375 samples (Case 1 & 3), 318 samples (Case 2)
- **Split Ratio**: 80:20 train/test
- **Source**: NewsKG21 with manually annotated entity-relationship triples

## Dependencies

```bash
pip install torch transformers datasets
pip install unsloth trl wandb
pip install spacy pandas ast
pip install medcat rdflib neo4j rdflib-neo4j
python -m spacy download en_core_web_sm
##  Validation of Hypotheses

**Data Filtering Improves Precision**: Case 2 achieved highest precision (0.2245)  
**Data Augmentation Improves F1**: Case 3 achieved best overall F1 (0.2110)  
**Quality vs Quantity Trade-off**: Results demonstrate precision-recall balance

## 🏃‍♂️ Quick Start

### Task 1: MedCAT Integration
```bash
cd Task1
python Task1.py
```

### Task 2: LLM Fine-Tuning (Knowledge Graph)
```bash
# Preprocess data
cd data_preprocessing
python newKG21.py

# Run experiments
cd ../Task2
python newsfinetune1.py  # Case 1
python newfinetune2.py   # Case 2 
python newsfinetune3.py  # Case 3
```

### Clinical Summarization (MIMIC-IV-BHC)
```bash
# Run full pipeline with training
python mimic-iv-bhc.py --data-file mimic-iv-bhc.csv

# Run inference only (skip training)
python mimic-iv-bhc.py --skip-training --context-bin all

# Run specific context length
python mimic-iv-bhc.py --context-bin short --output-dir short_results
```

---

