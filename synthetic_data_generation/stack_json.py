#!/usr/bin/env python3
"""
Script to combine multiple JSON files containing medical entity records into a single
JSON file and CSV file.
"""

import os
import json
import csv
import argparse
import logging
from typing import List, Dict, Any, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_json_file(file_path: str) -> List[List[Any]]:
    """
    Read a JSON file containing records.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of records, where each record is [text, ner_pairs, has_med]
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return []

def combine_json_files(input_dir: str) -> List[List[Any]]:
    """
    Combine all JSON files in the input directory and its subdirectories.
    
    Args:
        input_dir: Directory containing JSON files
        
    Returns:
        Combined list of records
    """
    all_records = []
    
    # Get all JSON files in the directory and subdirectories
    json_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('_entities.json'):
                json_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    # Process each file
    for file_path in json_files:
        records = read_json_file(file_path)
        # Filter out any non-list items (stray entity types)
        records = [r for r in records if isinstance(r, list) and len(r) == 3]
        all_records.extend(records)
    
    logger.info(f"Combined {len(all_records)} total records")
    return all_records

def save_json_output(records: List[List[Any]], output_path: str):
    """
    Save records in JSON format.
    
    Args:
        records: List of records to save
        output_path: Path to save the JSON file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2)
        logger.info(f"Saved JSON output to {output_path}")
    except Exception as e:
        logger.error(f"Error saving JSON output: {e}")

def save_csv_output(records: List[List[Any]], output_path: str):
    """
    Save records in CSV format.
    
    Args:
        records: List of records to save
        output_path: Path to save the CSV file
    """
    try:
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)  # Quote all fields
            # Write header
            writer.writerow(['Original Text', 'NER Tag Pairs', 'Has Medical Info'])
            # Write records
            for record in records:
                # Format NER pairs as a list of tuples without double quotes
                ner_pairs = record[1]
                # Format as [('tag','token'), ('tag','token'), ...] with single quotes
                ner_pairs_str = '[' + ','.join([f"('{tag}','{token}')" for tag, token in ner_pairs]) + ']'
                writer.writerow([
                    record[0],  # Original text
                    ner_pairs_str,  # NER pairs as list of tuples
                    str(record[2]).lower()  # Boolean as lowercase string
                ])
        logger.info(f"Saved CSV output to {output_path}")
    except Exception as e:
        logger.error(f"Error saving CSV output: {e}")

def main():
    """Main function to combine JSON files and create outputs."""
    parser = argparse.ArgumentParser(description="Combine JSON files and create JSON/CSV outputs")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing JSON files to combine")
    parser.add_argument("--output_dir", type=str, default="combined_output",
                       help="Directory to save output files")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Combine records from all JSON files
    all_records = combine_json_files(args.input_dir)
    
    if not all_records:
        logger.error("No records found to process")
        return
    
    # Save outputs
    json_path = os.path.join(args.output_dir, "combined_records.json")
    csv_path = os.path.join(args.output_dir, "combined_records.csv")
    
    save_json_output(all_records, json_path)
    save_csv_output(all_records, csv_path)
    
    # Print sample record
    logger.info("\nSample record:")
    logger.info("-" * 50)
    sample = all_records[0]
    logger.info(f"Original text: {sample[0]}")
    logger.info(f"NER pairs: {sample[1]}")
    logger.info(f"Has medical info: {sample[2]}")
    logger.info("-" * 50)

if __name__ == "__main__":
    main() 