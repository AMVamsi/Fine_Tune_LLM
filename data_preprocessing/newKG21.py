"""
Knowledge Graph Triple Extraction - Data Preprocessing Pipeline

This script processes raw newsKG21 dataset and creates three different training datasets
for comparative LLM fine-tuning experiments on knowledge graph triple extraction from
news articles. Each approach represents a different strategy for training data preparation
and subject entity handling.

Experimental Cases:
1. **Gold Standard Triples (Case 1)**: 
   - Pure gold-labeled triples with subject-conditioned prompts
   - Output: gold_prompt_train.csv, gold_prompt_test.csv

2. **spaCy-Filtered Gold Triples (Case 2)**:
   - Gold triples filtered using spaCy subject extraction and matching
   - Output: gold_filter_train.csv, gold_filter_test.csv

3. **Combined Gold + spaCy Extracted (Case 3)**:
   - Combination of gold triples and spaCy-extracted triples
   - Output: combined_trained.csv, combined_tested.csv

Key Features:
- Data validation and quality checks for triple format consistency
- spaCy NLP processing for dependency parsing and subject extraction
- Subject-conditioned prompt engineering for each approach
- Compound noun phrase handling for complex entities
- Comprehensive triple extraction rules (active/passive voice, conjuncts)

Input Data: train.txt, test.txt (raw newsKG21 format)
Output Data: Six CSV files for three different training approaches
Dependencies: spaCy (en_core_web_sm), datasets, ast

This preprocessing pipeline enables systematic comparison of different training data
preparation strategies for knowledge graph triple extraction tasks.
"""

import ast
from datasets import Dataset
import spacy

def load_txt_to_dict_list(filepath):
    data = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # Safely evaluate the string to a Python dictionary
                record = ast.literal_eval(line)

                # Extract 'sentence' and 'triple' keys
                sentence = record.get('sentence', '')
                triples = record.get('triple', [])

                data.append({
                    # "instruction": "Extract all subject–predicate–object triples from the sentence.",
                    "input": sentence,
                    "output": str(triples)
                })

            except Exception as e:
                print(f" Error on line:\n{line}\n{e}")

    return data
   
#----------------- Loading the Dataset ----------------------------------------   
#  Usage
file_path_train = "train.txt"
file_path_test = "test.txt" 
formatted_data_train = load_txt_to_dict_list(file_path_train)
formatted_data_test = load_txt_to_dict_list(file_path_test)

print('---------------Loading Dataset----------------')
#check a sample
print("sample formatted Train data:\n",formatted_data_train[0])
print("Sample formatted Test data:\n",formatted_data_test[0])


dataset_train = Dataset.from_list(formatted_data_train)
dataset_test = Dataset.from_list(formatted_data_test)

print('dataset_train:\n',dataset_train)
print('dataset_test:\n',dataset_test)

#---------------------- Data Validation ---------------------------------------


def validate_dataset(dataset):
    invalid_entries = []
    multiple_triple_count = 0

    for idx, item in enumerate(dataset):
        try:
            # Check required keys
            if not all(k in item for k in ['input', 'output']):
                invalid_entries.append((idx, "Missing required keys"))
                continue

            output_str = item['output'].strip()

            # Check if output string starts and ends with valid brackets
            if not (output_str.startswith("[") and output_str.endswith("]")):
                invalid_entries.append((idx, "Output format invalid or not properly closed"))
                continue

            # Parse the output string
            triples = ast.literal_eval(output_str)

            if not isinstance(triples, list):
                invalid_entries.append((idx, "Output is not a list"))
                continue

            # Check if each element is a valid triple
            if not all(isinstance(triple, list) and len(triple) == 3 for triple in triples):
                print(triples)
                invalid_entries.append((triples, "Each triple must be a list of 3 elements"))
                continue

            # Count multiple triples
            if len(triples) > 1:
                multiple_triple_count += 1

        except Exception as e:
            invalid_entries.append((idx, f"Error parsing output: {e}"))

    print(f"\n Validation completed.")
    print(f"Total entries: {len(dataset)}")
    print(f"Total entries with multiple triples: {multiple_triple_count}")
    print(f" Total invalid entries: {len(invalid_entries)}")

    if invalid_entries:
        print("\n Sample invalid entries:")
        for i, msg in invalid_entries[:5]:  # Display first 5 issues
            print(f" - Row {i}: {msg}")
    else:
        print(" All entries are valid.")

print('---------------Data Validation----------------')
print(validate_dataset(dataset_train))
print(validate_dataset(dataset_test))

#----------------------- 1. Prompt from Gold Triple -----------------------------

def build_subject_conditioned_dataset_from_gold_triples(dataset):
    """
    Given a dataset where each row contains a sentence and a list of gold SPO triples,
    generate a new Alpaca-style prompt for each triple using its subject entity.
    """
    new_rows = []

    for row in dataset:
        sentence = row["input"]
        try:
            gold_triples = ast.literal_eval(row["output"])
        except Exception as e:
            print(f"Skipping malformed output: {row['output']} ({e})")
            continue

        for triple in gold_triples:
            if len(triple) != 3:
                continue  # skip malformed triples
            subj, pred, obj = triple

            instruction = f"Given the subject entity '{subj}', extract the full '{subj}'–predicate–object triples from the sentence."
            new_rows.append({
                "instruction": instruction,
                "input": sentence,
                "output": str([subj, pred, obj])
            })

    return Dataset.from_list(new_rows)

gold_prompt_train = build_subject_conditioned_dataset_from_gold_triples(dataset_train)
gold_prompt_test = build_subject_conditioned_dataset_from_gold_triples(dataset_test)

print('--------------- 1. golden prompt ----------------')

for row in gold_prompt_test:
    print(row)
    break

#------------------------- 2. Prompt from common subjects gold triple && spacy extract -------------

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# --------- Extract all subject candidates from the sentence ---------
def extract_all_subjects(sentence):
    doc = nlp(sentence)
    subjects = []
    for token in doc:
        if token.dep_ == "nsubj":
            subject = get_compound_subject(token)
            subjects.append(subject)
    return subjects

# --------- Handle compound/complex subjects ---------
def get_compound_subject(token):
    parts = []
    for child in token.lefts:
        if child.dep_ in ("compound", "amod"):
            parts.append(child.text)
    parts.append(token.text)
    return " ".join(parts)

# --------- Filter dataset & create subject-specific prompts ---------
def generate_subject_conditioned_prompts(dataset):
    updated = []

    for row in dataset:
        sentence = row["input"]
        try:
            gold_triples = ast.literal_eval(row["output"])
        except Exception:
            continue  # skip malformed rows

        # Extract subjects from the sentence using spaCy
        spacy_subjects = extract_all_subjects(sentence)
        found_triples = []

        for triple in gold_triples:
            subj, pred, obj = triple
            for s_subj in spacy_subjects:
                if s_subj.lower() in subj.lower() or subj.lower() in s_subj.lower():
                    instruction = (
                        f"Given the subject entity '{subj}', extract the full '{subj}'–predicate–object triples from the sentence."
                    )
                    found_triples.append({
                        "instruction": instruction,
                        "input": sentence,
                        "output": str([subj, pred, obj])
                    })
                    break  # avoid duplicates if multiple subjects match

        # Add only if any matching triple found
        if found_triples:
            updated.extend(found_triples)
    return Dataset.from_list(updated)


gold_filter_train = generate_subject_conditioned_prompts(dataset_train)
gold_filter_test = generate_subject_conditioned_prompts(dataset_test)

print('--------------- 2. gold + filter + spacy ----------------')
for row in gold_filter_train:
    print(row)
    break


#--------------------------- 3. gold triple + spacy ------------------------------------------------

#3 - gold + spacy extracted triple

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# -------- Utility: Reconstruct compound noun phrases --------
def get_compound_noun(token):
    compound_parts = []
    for child in token.lefts:
        if child.dep_ in ("compound", "amod"):
            compound_parts.append(child.text)
    compound_parts.append(token.text)
    return " ".join(compound_parts)

# -------- Triple Extraction Function (with conjunct handling) --------
def extract_clean_triples(sentence):
    doc = nlp(sentence)
    triples = []

    for token in doc:
        # Rule 1: subject → verb → object
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            subject = get_compound_noun(token)
            predicate = token.head.lemma_

            obj_token = None
            for child in token.head.children:
                if child.dep_ in ("dobj", "attr", "oprd"):
                    obj_token = child
                elif child.dep_ == "prep":
                    for subchild in child.children:
                        if subchild.dep_ == "pobj":
                            obj_token = subchild

            if obj_token:
                obj_main = get_compound_noun(obj_token)
                triples.append([subject, predicate, obj_main])

                # NEW: Handle "conj" objects like "Budweiser"
                for sibling in obj_token.conjuncts:
                    obj_alt = get_compound_noun(sibling)
                    triples.append([subject, predicate, obj_alt])

        # Rule 2: passive voice agent pattern
        if token.dep_ == "agent" and token.head.dep_ == "acl":
            for child in token.children:
                if child.pos_ in ("NOUN", "PROPN"):
                    subject = get_compound_noun(child)
                    predicate = f"{token.head}_by"
                    obj = token.head.head.text
                    triples.append([subject, predicate, obj])

    return triples

# -------- Final Dataset Builder: Combine Extracted + Gold Triples --------

def build_combined_triple_dataset(dataset):
    examples = []

    for row in dataset:
        sentence = row["input"]

        # Get gold triples (if available)
        try:
            gold_triples = ast.literal_eval(row["output"])
        except:
            gold_triples = []

        gold_set = set(tuple(trip) for trip in gold_triples)

        # Add gold-labeled triples
        for subj, pred, obj in gold_triples:
            examples.append({
                "instruction": f"Given the subject entity '{subj}', extract the full triple from the sentence.",
                "input": sentence,
                "output": str([subj, pred, obj])
            })

        # Extracted triples using spaCy
        extracted_triples = extract_clean_triples(sentence)
        for subj, pred, obj in extracted_triples:
            if (subj, pred, obj) not in gold_set:
                examples.append({
                    "instruction": f"Given the subject entity '{subj}', extract the full triple from the sentence.",
                    "input": sentence,
                    "output": str([subj, pred, obj])
                })

    return Dataset.from_list(examples)

combined_trained = build_combined_triple_dataset(dataset_train)
combined_tested = build_combined_triple_dataset(dataset_test)

print('--------------- 3. gold + spacy ----------------')
print('combined train\n',combined_trained)
print('combined test\n',combined_tested)

# === SAVE TO CSV FORMAT === 
gold_prompt_train.to_csv("gold_prompt_train.csv")
gold_prompt_test.to_csv("gold_prompt_test.csv")

gold_filter_train.to_csv("gold_filter_train.csv")
gold_filter_test.to_csv("gold_filter_test.csv")

combined_trained.to_csv("combined_trained.csv")
combined_tested.to_csv("combined_tested.csv")