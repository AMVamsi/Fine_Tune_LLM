"""
LLM Fine-tuning for Knowledge Graph Triple Extraction - Case 3

This script fine-tunes a Mistral-7B language model for extracting knowledge graph triples
from news articles using a combined training approach. The model is trained on both gold
standard triples and spaCy-extracted triples, providing a hybrid dataset that combines
manually annotated high-quality triples with automatically extracted triples from spaCy
NLP processing. This approach aims to leverage both precision of gold labels and coverage
of automated extraction.

Key Features:
- Uses combined gold standard + spaCy-extracted triples for training
- Implements hybrid triple extraction with comprehensive training data
- Employs LoRA (Low-Rank Adaptation) for efficient parameter updates
- Evaluates model performance on predicate-object pair extraction
- Integrates with Weights & Biases for experiment tracking

Training Data: Combined train/test sets with both gold and spaCy-extracted triples
             (generated by newKG21.py preprocessing pipeline)
Model: Mistral-7B-Instruct with 4-bit quantization and LoRA adapters
Evaluation: Precision, Recall, and F1 scores on hybrid triple extraction task
Approach: Hybrid approach combining manual annotation quality with automated extraction coverage
"""

import os


os.environ["HF_HOME"] = "/cfs/earth/scratch/adlurmoh/track2/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/cfs/earth/scratch/adlurmoh/track2/hf_cache"
os.environ["WANDB_CACHE_DIR"] = "/cfs/earth/scratch/adlurmoh/track2/wandb_cache"
os.environ["TRITON_CACHE_DIR"] = "/cfs/earth/scratch/adlurmoh/track2/triton_cache"

import re
import torch
from unsloth import FastLanguageModel
import pandas as pd
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
import json
import ast
import logging
import wandb

# ========= GPU ASSERT ========= #
assert torch.cuda.is_available(), "CUDA not available"
device = torch.device("cuda")
print(f"Using device: {torch.cuda.get_device_name(0)}")

# ========= Login to wandb ========= #
os.environ["WANDB_API_KEY"] = "REMOVED_FOR_SECURITY" 
wandb.init(project="newsKG21-triple-extraction", name="mistral-finetune", config={"epochs": 3})

# ========= Logging Setup ========= #
logging.basicConfig(
    filename="train_log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_print(msg):
    print(msg)
    logging.info(msg)

major_version, minor_version = torch.cuda.get_device_capability()
log_print(f"CUDA compute capability: {major_version}.{minor_version}")

# ========= Load Model ========= #
max_seq_length = 512
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

# ========= Apply LoRA Adapters ========= #
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

# ========= Load Train/Test Data ========= #
try:
    data_train = pd.read_csv('/cfs/earth/scratch/adlurmoh/track2/case3/combined_trained.csv')
    data_test = pd.read_csv('/cfs/earth/scratch/adlurmoh/track2/case3/combined_tested.csv')
    log_print(data_train.head())
except FileNotFoundError:
    log_print(" Error: 'train.csv' or 'test.csv' not found.")
    raise

dataset_train = Dataset.from_pandas(data_train)
dataset_test = Dataset.from_pandas(data_test)


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def format_prompt(example):
    return {
        "text": alpaca_prompt.format(example["instruction"], example["input"], example["output"]) + EOS_TOKEN
    }

datasetTrain = dataset_train.map(format_prompt)

# ========= Fine-Tuning ========= #
training_args = TrainingArguments(
    output_dir="news_llm_outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    report_to="wandb",
    run_name="mistral_newsKG21",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    lr_scheduler_type="linear",
    # save_steps=50,
    # save_total_limit=2,
    # evaluation_strategy="no",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=datasetTrain,
    dataset_text_field="text",
    args=training_args,
    packing=False,
    max_seq_length=max_seq_length,
)

trainer.train()

model.save_pretrained("newsKG21_finetuned_model3")
tokenizer.save_pretrained("newsKG21_finetuned_model3")
log_print(" Model and tokenizer saved to 'newsKG21_finetuned_model3'")

# ========= Test Inference ========= #
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="newsKG21_finetuned_model3",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)


def generate_prediction(example):
    prompt = alpaca_prompt.format(example["instruction"], example["input"], "")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=128)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("### Response:")[-1].strip()

log_print(" Generating predictions...")
predictions = [generate_prediction(example) for example in dataset_test]
ground_truths = [example["output"] for example in dataset_test]

# ========= Safe Parse Utility ========= #

def safe_parse(text):
    """
    Parse string representation of triples into Python data structure,
    handling common formatting issues
    """
    try:
        # Normalize smart quotes
        text = text.replace("’", "'").replace("“", '"').replace("”", '"')

        # Try parsing string
        parsed = ast.literal_eval(text)

        # CASE 1: A single triple
        if isinstance(parsed, list) and len(parsed) == 3 and all(isinstance(x, str) for x in parsed):
            return {tuple(parsed)}

        # CASE 2: List of triples
        if isinstance(parsed, list) and all(isinstance(t, list) and len(t) == 3 for t in parsed):
            return set(tuple(t) for t in parsed)

        return set()

    except Exception as e:
        print(f"\nParse error:\n{text}\nError: {e}")
        return set()



parsed_preds = [safe_parse(pred) for pred in predictions]
parsed_truths = [safe_parse(gt) for gt in ground_truths]

# Helper function to extract predicate-object pairs from triples
def po_only(triple_set):
    """Extract predicate-object pairs from triples, ignoring subjects"""
    return set((p, o) for (_, p, o) in triple_set)

# ========= Calculate Metrics ========= #
tp = fp = fn = 0
print("\n--- Evaluation Samples ---")
for idx, (pred_set, truth_set) in enumerate(zip(parsed_preds, parsed_truths)):
    pred_po = po_only(pred_set)
    truth_po = po_only(truth_set)

    tp += len(pred_po & truth_po)  # True positives: correctly predicted
    fp += len(pred_po - truth_po)  # False positives: incorrectly predicted
    fn += len(truth_po - pred_po)  # False negatives: missed predictions

# Calculate precision, recall, and F1 score
precision = tp / (tp + fp + 1e-8)  # Add small epsilon to avoid division by zero
recall = tp / (tp + fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)

# Print results
print("\n=== Final Evaluation Metrics (on Predicate + Object Only) ===")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Log metrics to WandB
wandb.log({"precision": precision, "recall": recall, "f1": f1})

# ========= Save Predictions ========= #
with open("newsfinetune3.jsonl", "w") as f:
    for example, pred in zip(dataset_test, predictions):
        f.write(json.dumps({
            "instruction": example["instruction"],
            "input": example["input"],
            "true_output": example["output"],
            "predicted_output": pred
        }) + "\n")

log_print(" Predictions saved to finetune3.jsonl")

wandb.finish()
