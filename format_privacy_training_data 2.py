#!/usr/bin/env python3
"""
Format Privacy Training Data
Reformats the JSON files in the examples directory into a single CSV file for training.
"""

import os
import json
import csv
import glob
import logging
from typing import List, Dict, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def join_tokens_with_smart_spacing(tokens: List[str]) -> str:
    """
    Join tokens with smart spacing around punctuation.
    
    Args:
        tokens: List of tokens to join
        
    Returns:
        Joined text with appropriate spacing
    """
    if not tokens:
        return ""
    
    # Define punctuation marks that should not have a space before them
    punctuation = {'.', ',', '!', '?', ';', ':', ')', ']', '}', '"', "'"}
    
    result = tokens[0]
    
    for i in range(1, len(tokens)):
        current_token = tokens[i]
        # Don't add space if current token is punctuation
        if current_token in punctuation:
            result += current_token
        else:
            result += " " + current_token
    
    return result

def process_json_file(json_file_path: str, writer: csv.writer) -> None:
    """
    Process a single JSON file and write its data to the CSV writer.
    
    Args:
        json_file_path: Path to the JSON file
        writer: CSV writer object
    """
    logger.info(f"Processing {json_file_path}")
    
    # Load the JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process each paragraph
    for paragraph in data['paragraphs']:
        # Extract tokens and selected tags
        tokens = [token_info['token'] for token_info in paragraph['tokens']]
        selected_tags = [token_info['selected_tag'] for token_info in paragraph['tokens']]
        
        # Create the original paragraph by joining tokens with smart spacing
        original_paragraph = join_tokens_with_smart_spacing(tokens)
        
        # Create the annotations as a list of (token, tag) pairs
        annotations = [(token, tag) for token, tag in zip(tokens, selected_tags)]
        
        # Get the pii flag
        pii = paragraph.get('pii', False)
        
        # Write the row to the CSV
        writer.writerow([original_paragraph, str(annotations), str(pii).lower()])

def main():
    """Main function to process all JSON files in the examples directory."""
    # Get all JSON files in the examples directory
    examples_dir = "examples"
    json_files = glob.glob(os.path.join(examples_dir, "*_token_tags.json"))
    
    if not json_files:
        logger.error(f"No JSON files found in {examples_dir}")
        return
    
    logger.info(f"Found {len(json_files)} JSON files")
    
    # Create output directory if it doesn't exist
    output_dir = "formatted_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output files
    output_csv_path = os.path.join(output_dir, "all_paragraphs.csv")
    output_json_path = os.path.join(output_dir, "all_paragraphs.json")
    
    # Open the CSV file for writing
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        # Write the header
        writer.writerow(['paragraph', 'annotations', 'pii'])
        
        # Process each JSON file and collect paragraphs for JSON output
        all_paragraphs = []
        
        # Process each JSON file
        for json_file in json_files:
            logger.info(f"Processing {json_file}")
            
            # Load the JSON data
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Add paragraphs to the list for JSON output
            for paragraph in data['paragraphs']:
                # Create a simplified paragraph with just tokens and selected tags
                simplified_paragraph = {
                    'tokens': [{'token': token_info['token'], 'selected_tag': token_info['selected_tag']} 
                              for token_info in paragraph['tokens']],
                    'pii': paragraph.get('pii', False)
                }
                all_paragraphs.append(simplified_paragraph)
            
            # Process each paragraph for CSV output
            for paragraph in data['paragraphs']:
                # Extract tokens and selected tags
                tokens = [token_info['token'] for token_info in paragraph['tokens']]
                selected_tags = [token_info['selected_tag'] for token_info in paragraph['tokens']]
                
                # Create the original paragraph by joining tokens with smart spacing
                original_paragraph = join_tokens_with_smart_spacing(tokens)
                
                # Create the annotations as a list of (token, tag) pairs
                annotations = [(token, tag) for token, tag in zip(tokens, selected_tags)]
                
                # Get the pii flag
                pii = paragraph.get('pii', False)
                
                # Write the row to the CSV
                writer.writerow([original_paragraph, str(annotations), str(pii).lower()])
        
        # Write all paragraphs to JSON file
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump({'paragraphs': all_paragraphs}, f, indent=2)
    
    logger.info(f"Saved all paragraphs to {output_csv_path} and {output_json_path}")
    logger.info("All files processed successfully")

if __name__ == "__main__":
    main() 