#!/usr/bin/env python3
"""
Brief Hospital Course (BHC) Summarization Pipeline using MIMIC-IV-BHC Dataset

Brief hospital course (BHC) summaries are critical clinical documents that concisely summarize 
a patient's hospital stay, including key medical events, treatments administered, and patient 
outcomes. While large language models (LLMs) demonstrate remarkable capabilities in automating 
real-world tasks, their effectiveness for healthcare applications such as synthesizing BHCs 
from clinical notes remains underexplored.

This pipeline introduces and evaluates the MIMIC-IV-BHC dataset, a novel pre-processed dataset 
encapsulating clinical note and brief hospital course (BHC) pairs specifically designed to 
adapt LLMs for BHC synthesis. The benchmark compares summarization performance across:

- Two general-purpose LLMs: LLaMA2-13B-Chat, GPT-4
- Healthcare-adapted LLMs: Clinical-T5, Flan-T5-Base (fallback)

Key Features:
- Multi-context evaluation: Short, medium, and long clinical notes
- Adaptation strategies: Zero-shot prompting, prefix prompting, and QLoRA fine-tuning
- Comprehensive evaluation: BERT Score for summarization quality assessment
- Clinical domain focus: Specialized for hospital course synthesis from discharge summaries
- Scalable architecture: Supports multiple model types and training strategies

Dataset: MIMIC-IV-BHC (2000 train/ 100 test pairs) for each context length bin (short, medium, long)

Author: Mohan Adluru
Created: June 2025
"""

import argparse
import os
# Configure cache directories to avoid disk space issues
os.environ["HF_HOME"] = "/cfs/earth/scratch/adlurmoh/track2/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/cfs/earth/scratch/adlurmoh/track2/hf_cache" 
os.environ["WANDB_CACHE_DIR"] = "/cfs/earth/scratch/adlurmoh/track2/wandb_cache"
os.environ["TRITON_CACHE_DIR"] = "/cfs/earth/scratch/adlurmoh/track2/triton_cache"

# ========= HUGGING FACE AUTHENTICATION =========
# Set the Hugging Face token
os.environ["HUGGINGFACE_TOKEN"] = "REMOVED_FOR_SECURITY"  # Replace with your actual token

# ========= WANDB SETUP =========
os.environ["WANDB_API_KEY"] = "REMOVED_FOR_SECURITY"  # Replace with your actual WandB API key
os.environ["OPENAI_API_KEY"] = "REMOVED_FOR_SECURITY"  # Replace with your actual OpenAI API key

import sys
import logging
from pathlib import Path
from datetime import datetime
import json
import pickle
import random
import re
import traceback
from typing import Dict, List, Tuple, Any, Optional

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# NLP and ML libraries
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import torch
import torch.nn as nn

# Unsloth imports (like newsfinetune1.py)
from unsloth import FastLanguageModel
from trl import SFTTrainer

# Standard transformers imports
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, TrainingArguments
)

# Import specific modules
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.trainer import Trainer
from transformers.data.data_collator import DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Define TaskType constants
class TaskType:
    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
    CAUSAL_LM = "CAUSAL_LM"

from datasets import Dataset
from sklearn.model_selection import train_test_split
import openai
from bert_score import score as bert_score
from huggingface_hub import login
import wandb

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# PIPELINE FUNCTIONS
# ============================================================================

def setup_hf_authentication():
    """Setup Hugging Face authentication for gated models like LLaMA2."""
    try:
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if hf_token:
            login(token=hf_token)
            logger.info("Hugging Face authentication successful")
        else:
            logger.warning("HUGGINGFACE_TOKEN not set. You may not be able to access gated models.")
    except Exception as e:
        logger.error(f"Failed to authenticate with Hugging Face: {e}")

def setup_environment():
    """Setup environment, download NLTK resources, and configure GPU."""
    logger.info("Setting up environment...")
    
    # Setup HF authentication first
    setup_hf_authentication()
    
    # Initialize WandB (like newsfinetune1.py)
    try:
        wandb.init(project="clinical-bhc-summarization", name="llama2-clinical", config={"epochs": 3})
        logger.info("WandB initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize WandB: {e}")
    
    # Check GPU availability (like newsfinetune1.py)
    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")
    logger.info(f"Using device: {torch.cuda.get_device_name(0)}")
    
    # Log GPU capabilities
    major_version, minor_version = torch.cuda.get_device_capability()
    logger.info(f"CUDA compute capability: {major_version}.{minor_version}")
    
    # Download NLTK resources
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        logger.info("NLTK resources downloaded successfully")
    except Exception as e:
        logger.warning(f"Could not download NLTK resources: {e}")
    
    logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    logger.info("Environment setup complete")

def extract_bhc_section(text: str) -> str:
    """Extract Brief Hospital Course section from clinical note."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Pattern to match BHC section
    bhc_patterns = [
        r'BRIEF HOSPITAL COURSE[:]\s*\n(.*?)(?=\n[A-Z][A-Z\s]+[:]\s*\n|$)',
        r'Brief Hospital Course[:]\s*\n(.*?)(?=\n[A-Z][A-Z\s]+[:]\s*\n|$)',
        r'HOSPITAL COURSE[:]\s*\n(.*?)(?=\n[A-Z][A-Z\s]+[:]\s*\n|$)',
        r'Hospital Course[:]\s*\n(.*?)(?=\n[A-Z][A-Z\s]+[:]\s*\n|$)'
    ]
    
    for pattern in bhc_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            bhc_text = match.group(1).strip()
            # Clean up the text
            bhc_text = re.sub(r'\n+', ' ', bhc_text)
            bhc_text = re.sub(r'\s+', ' ', bhc_text)
            return bhc_text
    
    # If no BHC section found, return first 500 characters
    return text[:500] if text else ""

def clean_clinical_text(text: str) -> str:
    """Clean and normalize clinical text."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common clinical artifacts
    text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed content
    text = re.sub(r'___+', '', text)     # Remove underscores
    text = re.sub(r'={3,}', '', text)    # Remove equal signs
    text = re.sub(r'-{3,}', '', text)    # Remove dashes
    
    # Fix common abbreviations
    text = re.sub(r'\bpt\b', 'patient', text, flags=re.IGNORECASE)
    text = re.sub(r'\bhx\b', 'history', text, flags=re.IGNORECASE)
    text = re.sub(r'\bdx\b', 'diagnosis', text, flags=re.IGNORECASE)
    
    return text.strip()

def get_context_length_bin(text: str, tokenizer) -> str:
    """Determine context length bin for a text."""
    if not text:
        return "short"
    
    token_count = len(tokenizer.encode(text))
    
    if token_count <= 1024:
        return "short"
    elif token_count <= 2048:
        return "medium"
    else:
        return "long"

def load_and_preprocess_data(data_file: str, output_dir: str) -> Tuple[pd.DataFrame, Dict[str, Dataset]]:
    """Load and preprocess the MIMIC-IV BHC dataset."""
    logger.info(f"Loading data from {data_file}")
    
    # Load data
    try:
        df = pd.read_csv(data_file)
        logger.info(f"Loaded {len(df)} records")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise e
    
    # Check required columns - UPDATE THIS PART
    required_cols = ['note_id', 'input', 'target']  # Changed to match your data
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Map your column names to expected names
    df = df.rename(columns={
        'note_id': 'hadm_id',
        'input': 'discharge_summary', 
        'target': 'brief_hospital_course'
    })
    
    # Extract and clean BHC sections
    logger.info("Processing clinical text...")
    # Since your 'input' is already the source text, use it directly
    df['source_text'] = df['discharge_summary'].apply(clean_clinical_text)
    df['target_text'] = df['brief_hospital_course'].apply(clean_clinical_text)
    
    # Filter out empty texts
    initial_count = len(df)
    df = df[(df['source_text'].str.len() > 20) & (df['target_text'].str.len() > 10)]
    logger.info(f"Filtered from {initial_count} to {len(df)} records")
    
    # Split into train/test
    train_df, test_df = train_test_split(df, test_size=0.05, random_state=42)
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Create context length bins using a tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    
    # Add context bins
    train_df['context_bin'] = train_df['source_text'].apply(
        lambda x: get_context_length_bin(x, tokenizer)
    )
    test_df['context_bin'] = test_df['source_text'].apply(
        lambda x: get_context_length_bin(x, tokenizer)
    )
    
    # Log bin distribution
    logger.info("Context bin distribution:")
    logger.info(f"Train: {train_df['context_bin'].value_counts().to_dict()}")
    logger.info(f"Test: {test_df['context_bin'].value_counts().to_dict()}")
    
    # Create binned datasets
    binned_datasets = {}
    for bin_name in ['short', 'medium', 'long']:
        train_bin = train_df[train_df['context_bin'] == bin_name]
        test_bin = test_df[test_df['context_bin'] == bin_name]
        
        if len(train_bin) > 0 and len(test_bin) > 0:
            binned_datasets[bin_name] = {
                'train': Dataset.from_pandas(train_bin),
                'test': Dataset.from_pandas(test_bin)
            }
            logger.info(f"{bin_name.capitalize()} bin - Train: {len(train_bin)}, Test: {len(test_bin)}")
    
    # Save processed data
    processed_file = os.path.join(output_dir, 'processed_data.pkl')
    with open(processed_file, 'wb') as f:
        pickle.dump({
            'train_df': train_df,
            'test_df': test_df,
            'binned_datasets': binned_datasets
        }, f)
    
    logger.info(f"Processed data saved to {processed_file}")
    return df, binned_datasets

def setup_models() -> Dict[str, Any]:
    """Setup and load all models using unsloth approach."""
    logger.info("Setting up models...")
    
    models = {}
    
    # Clinical T5 model (actual clinical domain model)
    try:
        logger.info("Loading Clinical-T5...")
        # Use an actual clinical model like ClinicalT5 or Bio-T5
        clinical_t5_tokenizer = AutoTokenizer.from_pretrained("luqh/ClinicalT5-base")
        clinical_t5_model = AutoModelForSeq2SeqLM.from_pretrained("luqh/ClinicalT5-base")
        
        models['clinical_t5'] = {
            'tokenizer': clinical_t5_tokenizer,
            'model': clinical_t5_model,
            'type': 'seq2seq'
        }
        logger.info("Clinical-T5 loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Clinical-T5: {e}")
        # Fallback to Flan-T5 if clinical model fails
        try:
            logger.info("Falling back to Flan-T5-Base...")
            flan_t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            flan_t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
            
            models['clinical_t5'] = {
                'tokenizer': flan_t5_tokenizer,
                'model': flan_t5_model,
                'type': 'seq2seq'
            }
            logger.info("Flan-T5-Base loaded as fallback")
        except Exception as e2:
            logger.error(f"Failed to load fallback model: {e2}")
def prepare_training_data_unsloth(dataset: Dataset, tokenizer) -> Dataset:
    """Prepare dataset for unsloth training (like newsfinetune1.py)."""
    
    # Define instruction prompt template (similar to newsfinetune1.py)
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Summarize the following clinical discharge summary into a brief hospital course.

### Input:
{}

### Response:
{}"""

    EOS_TOKEN = tokenizer.eos_token

    def format_prompt(example):
        """Format example into instruction prompt format with proper tokens"""
        return {
            "text": alpaca_prompt.format(example["source_text"], example["target_text"]) + EOS_TOKEN
        }

    # Apply formatting to training data
    return dataset.map(format_prompt)

def prepare_training_data(dataset: Dataset, tokenizer, model_type: str) -> Dataset:
    """Prepare dataset for training."""
    def preprocess_function(examples):
        if model_type == 'seq2seq':
            # For T5-style models
            inputs = [f"summarize: {text}" for text in examples['source_text']]
            model_inputs = tokenizer(
                inputs, 
                max_length=1024, 
                truncation=True, 
                padding=True
            )
            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    examples['target_text'], 
                    max_length=256, 
                    truncation=True, 
                    padding=True
                )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        elif model_type == 'causal':
            # For LLaMA2-13B Chat style models with proper chat format
            inputs = []
            for source, target in zip(examples['source_text'], examples['target_text']):
                prompt = f"""<s>[INST] <<SYS>>
You are a helpful medical assistant. Summarize the following clinical discharge summary into a brief hospital course.
<</SYS>>

{source} [/INST] {target}</s>"""
                inputs.append(prompt)
            
            model_inputs = tokenizer(
                inputs,
                max_length=2048,
                truncation=True,
                padding=True
            )
            
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            return model_inputs
    
    return dataset.map(preprocess_function, batched=True)

def train_models(binned_datasets: Dict[str, Dataset], output_dir: str):
    """Train models using unsloth approach (like newsfinetune1.py)."""
    logger.info("Starting training...")
    
    # Load base models for training
    models = setup_models()
    
    for model_name in ['llama2']:  # Focus on LLaMA2 with unsloth like newsfinetune1.py
        if model_name not in models:
            logger.warning(f"Model {model_name} not available for training")
            continue
        
        logger.info(f"Training {model_name}...")
        
        model_info = models[model_name]
        tokenizer = model_info['tokenizer']
        model = model_info['model']
        max_seq_length = model_info.get('max_seq_length', 2048)
        
        # Setup LoRA configuration using unsloth (like newsfinetune1.py)
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
            use_rslora=False,
            loftq_config=None,
        )
        
        # Train on each context bin
        for bin_name, dataset_dict in binned_datasets.items():
            logger.info(f"Training {model_name} on {bin_name} context bin...")
            
            # Get the training dataset
            train_data = dataset_dict['train']
            
            # Prepare dataset for unsloth training
            train_dataset = prepare_training_data_unsloth(train_data, tokenizer)
            
            # Training arguments (similar to newsfinetune1.py)
            training_args = TrainingArguments(
                output_dir=os.path.join(output_dir, f"{model_name}_{bin_name}_outputs"),
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                num_train_epochs=3,
                learning_rate=2e-4,
                logging_steps=10,
                report_to="wandb",
                run_name=f"{model_name}_clinical_{bin_name}",
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                optim="adamw_8bit",
                lr_scheduler_type="linear",
            )
            
            # Trainer using unsloth
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                dataset_text_field="text",
                args=training_args,
                packing=False,
                max_seq_length=max_seq_length,
            )
            
            # Train
            trainer.train()
            
            # Save the trained model
            model_save_path = os.path.join(output_dir, f"{model_name}_{bin_name}_finetuned")
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            logger.info(f"Model saved to {model_save_path}")

def format_prompt(text: str, strategy: str, model_type: str) -> str:
    """Format input text based on adaptation strategy."""
    if strategy == "zero_shot":
        if model_type == "seq2seq":
            return f"summarize: {text}"
        elif model_type == "causal":
            # Updated prompt format for LLaMA2-13B Chat
            prompt = f"""<s>[INST] <<SYS>>
You are a helpful medical assistant. Summarize the following clinical discharge summary into a brief hospital course.
<</SYS>>

{text} [/INST]"""
            return prompt
        else:  # API models
            return text
    
    elif strategy == "prefix":
        prefix = "As a clinical expert, provide a concise summary focusing on key medical events, treatments, and outcomes: "
        if model_type == "seq2seq":
            return f"summarize: {prefix}{text}"
        elif model_type == "causal":
            # Updated prompt format for LLaMA2-13B Chat with prefix
            prompt = f"""<s>[INST] <<SYS>>
You are a helpful medical assistant. {prefix}
<</SYS>>

{text} [/INST]"""
            return prompt
        else:  # API models
            return f"{prefix}\n\n{text}"
    
    return text

def run_inference_pipeline(binned_datasets: Dict[str, Dataset], models: Dict[str, Any], 
                         context_bin: str, output_dir: str) -> Dict[str, Any]:
    """Run inference pipeline for all models and strategies."""
    logger.info("Starting inference pipeline...")
    
    results = {}
    strategies = ["zero_shot", "prefix"]  # Remove qlora from strategies as it's handled in training
    
    # Determine which bins to process
    bins_to_process = [context_bin] if context_bin != 'all' else list(binned_datasets.keys())
    
    for bin_name in bins_to_process:
        if bin_name not in binned_datasets:
            logger.warning(f"Context bin {bin_name} not available")
            continue
        
        logger.info(f"Processing {bin_name} context bin...")
        test_dataset = binned_datasets[bin_name]['test']
        
        for model_name, model_info in models.items():
            logger.info(f"Running inference with {model_name}...")
            
            for strategy in strategies:
                logger.info(f"Strategy: {strategy}")                
                predictions = []
                references = []
                
                for i, example in enumerate(test_dataset):
                    source_text = example['source_text']
                    target_text = example['target_text']
                    
                    # Format prompt
                    formatted_input = format_prompt(source_text, strategy, model_info['type'])
                    
                    try:
                        # Generate prediction
                        if model_info['type'] == 'api':
                            # GPT-4 API call
                            from openai import OpenAI
                            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                            
                            response = client.chat.completions.create(
                                model="gpt-4",
                                messages=[
                                    {"role": "system", "content": "You are a clinical expert. Provide concise summaries of medical cases."},
                                    {"role": "user", "content": formatted_input}
                                ],
                                max_tokens=256,
                                temperature=0.1
                            )
                            prediction = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
                        
                        elif model_info['type'] == 'seq2seq':
                            # T5-style models
                            tokenizer = model_info['tokenizer']
                            model = model_info['model']
                            
                            inputs = tokenizer(formatted_input, return_tensors="pt", max_length=1024, truncation=True)
                            
                            with torch.no_grad():
                                outputs = model.generate(
                                    **inputs,
                                    max_length=256,
                                    num_beams=4,
                                    length_penalty=1.0,
                                    early_stopping=True
                                )
                            
                            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        elif model_info['type'] == 'causal':
                            # LLaMA2-13B Chat style models using unsloth
                            tokenizer = model_info['tokenizer']
                            model = model_info['model']
                            
                            # Enable inference mode for unsloth
                            FastLanguageModel.for_inference(model)
                            
                            inputs = tokenizer(formatted_input, return_tensors="pt", max_length=2048, truncation=True)
                            
                            with torch.no_grad():
                                outputs = model.generate(
                                    **inputs,
                                    max_new_tokens=256,
                                    do_sample=True,
                                    temperature=0.1,
                                    pad_token_id=tokenizer.eos_token_id
                                )
                            
                            # Extract only the new tokens (response part)
                            response_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                            prediction = tokenizer.decode(response_tokens, skip_special_tokens=True)
                        
                        else:
                            prediction = "Unknown model type"
                        
                        predictions.append(prediction)
                        references.append(target_text)
                        
                        if (i + 1) % 10 == 0:
                            logger.info(f"Processed {i + 1}/{len(test_dataset)} examples")
                    
                    except Exception as e:
                        logger.error(f"Error processing example {i}: {e}")
                        predictions.append("")
                        references.append(target_text)
                
                # Store results
                result_key = f"{model_name}_{strategy}_{bin_name}"
                results[result_key] = {
                    'predictions': predictions,
                    'references': references,
                    'model': model_name,
                    'strategy': strategy,
                    'context_bin': bin_name
                }
                
                # Save intermediate results
                result_file = os.path.join(output_dir, f"inference_results_{result_key}.pkl")
                with open(result_file, 'wb') as f:
                    pickle.dump(results[result_key], f)
                
                logger.info(f"Results saved for {result_key}")
    
    # Save all results
    all_results_file = os.path.join(output_dir, "all_inference_results.pkl")
    with open(all_results_file, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"All inference results saved to {all_results_file}")
    return results

def run_evaluation_pipeline(inference_results: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Run evaluation pipeline using BERT Score."""
    logger.info("Starting evaluation pipeline...")
    
    evaluation_results = {}
    
    for result_key, result_data in inference_results.items():
        logger.info(f"Evaluating {result_key}...")
        
        predictions = result_data['predictions']
        references = result_data['references']
        
        # Filter out empty predictions
        valid_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip()]
        
        if not valid_pairs:
            logger.warning(f"No valid predictions for {result_key}")
            continue
        
        valid_predictions, valid_references = zip(*valid_pairs)
        
        # Calculate BERT Score
        try:
            P, R, F1 = bert_score(valid_predictions, valid_references, lang="en", verbose=False)
            
            bert_scores = {
                'precision': P.mean().item(),
                'recall': R.mean().item(),
                'f1': F1.mean().item(),
                'precision_std': P.std().item(),
                'recall_std': R.std().item(),
                'f1_std': F1.std().item(),
                'num_examples': len(valid_pairs)
            }
            
            evaluation_results[result_key] = bert_scores
            
            logger.info(f"{result_key} - BERT F1: {bert_scores['f1']:.4f} ± {bert_scores['f1_std']:.4f}")
        
        except Exception as e:
            logger.error(f"Error calculating BERT Score for {result_key}: {e}")
    
    # Create summary report
    summary_report = []
    for result_key, scores in evaluation_results.items():
        model, strategy, context_bin = result_key.split('_', 2)
        summary_report.append({
            'Model': model,
            'Strategy': strategy,
            'Context_Bin': context_bin,
            'BERT_F1': scores['f1'],
            'BERT_F1_Std': scores['f1_std'],
            'BERT_Precision': scores['precision'],
            'BERT_Recall': scores['recall'],
            'Num_Examples': scores['num_examples']
        })
    
    # Save evaluation results
    eval_file = os.path.join(output_dir, "evaluation_results.pkl")
    with open(eval_file, 'wb') as f:
        pickle.dump(evaluation_results, f)
    
    # Save summary as CSV
    summary_df = pd.DataFrame(summary_report)
    summary_file = os.path.join(output_dir, "evaluation_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    
    # Create visualization
    create_evaluation_plots(summary_df, output_dir)
    
    logger.info(f"Evaluation results saved to {eval_file}")
    logger.info(f"Summary report saved to {summary_file}")
    
    return evaluation_results

def create_evaluation_plots(summary_df: pd.DataFrame, output_dir: str):
    """Create evaluation plots and visualizations."""
    logger.info("Creating evaluation plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Plot 1: BERT F1 scores by model and strategy
    plt.figure(figsize=(12, 8))
    
    # Pivot data for better plotting
    pivot_df = summary_df.pivot_table(
        index=['Model', 'Strategy'], 
        columns='Context_Bin', 
        values='BERT_F1'
    )
    
    # Create heatmap
    sns.heatmap(
        pivot_df, 
        annot=True, 
        fmt='.3f', 
        cmap='viridis',
        cbar_kws={'label': 'BERT F1 Score'}
    )
    plt.title('BERT F1 Scores by Model, Strategy, and Context Length')
    plt.xlabel('Context Length Bin')
    plt.ylabel('Model and Strategy')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bert_f1_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Evaluation plots saved to output directory")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main pipeline execution function."""
    parser = argparse.ArgumentParser(description='Clinical Summarization Pipeline')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file')
    parser.add_argument('--skip-training', action='store_true', help='Skip training')
    parser.add_argument('--context-bin', type=str, choices=['short', 'medium', 'long', 'all'], 
                       default='all', help='Context length bin to process')
    parser.add_argument('--data-file', type=str, default='mimic-iv-bhc.csv', help='Input data file')
    parser.add_argument('--output-dir', type=str, default=f'output_{datetime.now().strftime("%Y%m%d_%H%M%S")}', help='Output directory')
    
    args = parser.parse_args()
    
    logger.info("Starting Clinical Summarization Pipeline")
    logger.info(f"Arguments: {args}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Section 1: Environment Setup
        logger.info("Section 1: Environment Setup")
        setup_environment()
        
        # Section 2: Data Loading and Preprocessing
        logger.info("Section 2: Data Loading and Preprocessing")
        processed_data, binned_datasets = load_and_preprocess_data(args.data_file, args.output_dir)
        
        if not binned_datasets:
            logger.error("No valid context bins created. Exiting.")
            return 1
        
        # Section 3: Model Setup
        logger.info("Section 3: Model Setup")
        loaded_models = setup_models()
        
        if not loaded_models:
            logger.error("No models loaded successfully. Exiting.")
            return 1
        
        # Section 4: Training (if not skipped)
        if not args.skip_training:
            logger.info("Section 4: Training")
            train_models(binned_datasets, args.output_dir)
        else:
            logger.info("Section 4: Training skipped")
        
        # Section 5: Inference
        logger.info("Section 5: Inference Pipeline")
        inference_results = run_inference_pipeline(
            binned_datasets, loaded_models, args.context_bin, args.output_dir
        )
        
        if not inference_results:
            logger.error("No inference results generated. Exiting.")
            return 1
        
        # Section 6: Evaluation
        logger.info("Section 6: Evaluation")
        evaluation_results = run_evaluation_pipeline(
            inference_results, args.output_dir
        )
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")
        
        # Cleanup
        wandb.finish()
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)