#!/usr/bin/env python3
"""
Token Tag Waterfall Processor
Processes CoNLL files maintaining token-level granularity and applying a waterfall tagging approach.
"""

import os
import logging
import argparse
import json
import spacy
import random
import numpy as np
from typing import List, Dict, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load Spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.info("Downloading Spacy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def read_conll_file(file_path: str) -> List[List[Tuple[str, str]]]:
    """
    Read a CoNLL formatted file and return paragraphs.
    Each paragraph is a list of lines, maintaining token-level granularity.
    Each line is a tuple of (token, tag).
    """
    paragraphs = []
    current_paragraph = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:  # Blank line indicates paragraph break
                if current_paragraph:
                    paragraphs.append(current_paragraph)
                    current_paragraph = []
            else:
                # Split on whitespace and take first two elements (token and tag)
                parts = line.split()
                if len(parts) >= 2:
                    current_paragraph.append((parts[0], parts[1]))
                else:
                    current_paragraph.append((parts[0], "O"))  # Default tag if missing

    if current_paragraph:  # Add last paragraph if not empty
        paragraphs.append(current_paragraph)

    return paragraphs

def select_tag(tag_options):
    """
    Select the most appropriate tag based on the waterfall priority:
    1. Original BIO tags (B-* or I-*)
    2. Spacy BIO tags
    3. Original non-BIO tags
    4. Default to O
    """
    # First check for original BIO tags
    for tag_opt in tag_options:
        if tag_opt["source"] == "conll" and (tag_opt["tag"].startswith("B-") or tag_opt["tag"].startswith("I-")):
            return tag_opt["tag"]
    
    # Then check for Spacy BIO tags
    for tag_opt in tag_options:
        if tag_opt["source"] == "spacy" and (tag_opt["tag"].startswith("B-") or tag_opt["tag"].startswith("I-")):
            return tag_opt["tag"]
    
    # Then use original non-BIO tag if it's not O
    for tag_opt in tag_options:
        if tag_opt["source"] == "conll" and tag_opt["tag"] != "O":
            return tag_opt["tag"]
    
    # Default to O
    return "O"

def process_token(token: str, original_tag: str, spacy_doc: spacy.tokens.Doc = None) -> Dict:
    """
    Process a single token through the tagging waterfall.
    Returns a dictionary with the token and its possible tags.
    """
    token_info = {
        "token": token,
        "tag_options": [],
        "selected_tag": None
    }
    
    # Add original CoNLL tag as first option
    token_info["tag_options"].append({
        "source": "conll",
        "tag": original_tag,
        "pos": None
    })
    
    # Add Spacy tags
    if spacy_doc:
        # Check if token is part of any entity
        is_part_of_entity = False
        for ent in spacy_doc.ents:
            if token in ent.text:
                is_part_of_entity = True
                # Convert Spacy tag to BIO format if it's part of an entity
                bio_tag = f"B-{ent.label_}" if token == ent.text.split()[0] else f"I-{ent.label_}"
                token_info["tag_options"].append({
                    "source": "spacy",
                    "tag": bio_tag,
                    "pos": None
                })
        
        # If token is not part of any entity, add Spacy O tag
        if not is_part_of_entity:
            token_info["tag_options"].append({
                "source": "spacy",
                "tag": "O",
                "pos": None
            })
    
    # Select tag based on waterfall priority
    token_info["selected_tag"] = select_tag(token_info["tag_options"])
    
    return token_info

def process_paragraph(paragraph: List[Tuple[str, str]], spacy_doc=None) -> Dict:
    """
    Process a paragraph maintaining token-level granularity.
    Returns a dictionary with tokens and their tags.
    """
    # Process each token
    tokens = []
    has_pii = False  # Flag to track if paragraph contains PII
    
    for token, original_tag in paragraph:
        # Skip empty tokens or tags that are actually tokens
        if not token or token == original_tag:
            continue
        
        # Check if this token has a non-trivial tag (non-"O")
        if original_tag != "O":
            has_pii = True
            
        token_info = process_token(token, original_tag, spacy_doc)
        tokens.append(token_info)
    
    return {
        "tokens": tokens,
        "pii": has_pii  # Add pii flag to the paragraph
    }

def process_file(file_path: str, output_dir: str) -> None:
    """
    Process a single CoNLL file, applying the tagging waterfall approach.
    """
    # Read and parse the CoNLL file
    paragraphs = read_conll_file(file_path)
    
    # Run Spacy on all paragraphs at once
    all_texts = []
    for paragraph in paragraphs:
        tokens = [token for token, _ in paragraph]
        all_texts.append(" ".join(tokens))
    
    # Process all texts with Spacy - THIS IS THE ONLY PLACE WE USE SPACY
    spacy_docs = [nlp(text) for text in all_texts]
    
    # Process each paragraph with its corresponding Spacy doc
    processed_paragraphs = []
    for paragraph, spacy_doc in zip(paragraphs, spacy_docs):
        processed_paragraph = process_paragraph(paragraph, spacy_doc)
        processed_paragraphs.append(processed_paragraph)
    
    # Save outputs
    basename = os.path.basename(file_path)
    
    # Save original text
    with open(os.path.join(output_dir, f"{basename}_original.txt"), "w") as f:
        for paragraph in paragraphs:
            tokens = [token for token, _ in paragraph]
            f.write(" ".join(tokens) + "\n")
    
    # Save token tags in JSON format
    output_json = {
        "paragraphs": processed_paragraphs
    }
    with open(os.path.join(output_dir, f"{basename}_token_tags.json"), "w") as f:
        json.dump(output_json, f, indent=2)
    
    # Save human-readable format
    with open(os.path.join(output_dir, f"{basename}_readable.txt"), "w") as f:
        for para in processed_paragraphs:
            f.write(f"Paragraph (pii: {para['pii']}):\n")
            for token_info in para["tokens"]:
                f.write(f"Token: {token_info['token']}")
                f.write("\n")
                f.write("  Tag options:\n")
                for opt in token_info["tag_options"]:
                    f.write(f"    - {opt['source']}: {opt['tag']}\n")
                f.write(f"  Selected tag: {token_info['selected_tag']}\n")
            f.write("\n")

def main():
    """Main function to process files."""
    parser = argparse.ArgumentParser(description="Process CoNLL formatted files with token-level tagging")
    parser.add_argument("--dataset_dir", type=str, 
                       default="synthetic_data_generation/dataset_txt/dataset_txt/training",
                       help="Directory containing CoNLL formatted files")
    parser.add_argument("--num_samples", type=int, default=1,
                       help="Number of samples to process")
    parser.add_argument("--output_dir", type=str, default="examples",
                       help="Output directory for processed files")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--specific_file", type=str, default=None,
                       help="Process a specific file instead of random sampling")
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Get list of files to process
    if args.specific_file:
        files_to_process = [args.specific_file]
    else:
        # Get all files from training directory
        training_dir = "synthetic_data_generation/dataset_txt/dataset_txt/training"
        files_to_process = [f for f in os.listdir(training_dir) if f.startswith("annot_csv_") and f.endswith(".txt")]
        if args.num_samples:
            files_to_process = random.sample(files_to_process, min(args.num_samples, len(files_to_process)))
    
    # Process each file
    for file_name in files_to_process:
        logger.info(f"Processing file: {file_name}")
        file_path = os.path.join(training_dir, file_name)
        process_file(file_path, args.output_dir)

if __name__ == "__main__":
    main() 