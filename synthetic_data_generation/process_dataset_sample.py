#!/usr/bin/env python3
"""
Script to randomly sample and process files from dataset_txt directories.
"""

import os
import random
import argparse
import logging
import subprocess
from typing import List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_all_files(base_dir: str) -> List[Tuple[str, str]]:
    """
    Get all txt files from dataset_txt directories.
    
    Args:
        base_dir: Base directory containing dataset_txt
        
    Returns:
        List of (file_path, dataset_type) tuples
    """
    all_files = []
    dataset_types = ['eval', 'test', 'training']
    
    for dataset_type in dataset_types:
        dir_path = os.path.join(base_dir, 'dataset_txt', 'dataset_txt', dataset_type)
        if not os.path.exists(dir_path):
            logger.warning(f"Directory not found: {dir_path}")
            continue
            
        files = [f for f in os.listdir(dir_path) if f.endswith('.txt')]
        for file in files:
            all_files.append((os.path.join(dir_path, file), dataset_type))
    
    return all_files

def sample_files(all_files: List[Tuple[str, str]], num_samples: int) -> List[Tuple[str, str]]:
    """
    Randomly sample files without replacement.
    
    Args:
        all_files: List of (file_path, dataset_type) tuples
        num_samples: Number of files to sample
        
    Returns:
        List of sampled (file_path, dataset_type) tuples
    """
    return random.sample(all_files, min(num_samples, len(all_files)))

def process_files(sampled_files: List[Tuple[str, str]], output_base_dir: str, model: str):
    """
    Process sampled files using the injector script.
    
    Args:
        sampled_files: List of (file_path, dataset_type) tuples
        output_base_dir: Base directory for outputs
        model: Model to use for injection
    """
    # Group files by dataset type
    files_by_type = {}
    for file_path, dataset_type in sampled_files:
        if dataset_type not in files_by_type:
            files_by_type[dataset_type] = []
        files_by_type[dataset_type].append(file_path)
    
    # Process each dataset type
    for dataset_type, files in files_by_type.items():
        # Create output directory for this dataset type
        output_dir = os.path.join(output_base_dir, dataset_type)
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the dataset directory path
        dataset_dir = os.path.dirname(files[0])
        
        # Run the injector script
        cmd = [
            'python', 'medical_entity_injector_llm_new.py',
            '--model', model,
            '--dataset_dir', dataset_dir,
            '--num_samples', str(len(files)),
            '--output_dir', output_dir
        ]
        
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully processed {len(files)} files from {dataset_type}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error processing files from {dataset_type}: {e}")

def main():
    """Main function to sample and process files."""
    parser = argparse.ArgumentParser(description="Sample and process files from dataset_txt")
    parser.add_argument("--base_dir", type=str, required=True,
                       help="Base directory containing dataset_txt")
    parser.add_argument("--num_samples", type=int, required=True,
                       help="Number of files to sample")
    parser.add_argument("--model", type=str, default="mistral",
                       help="Model to use for injection")
    parser.add_argument("--output_base_dir", type=str, default="processed_samples",
                       help="Base directory for outputs")
    args = parser.parse_args()
    
    # Get all files
    logger.info("Getting all files from dataset directories...")
    all_files = get_all_files(args.base_dir)
    logger.info(f"Found {len(all_files)} total files")
    
    # Sample files
    logger.info(f"Sampling {args.num_samples} files...")
    sampled_files = sample_files(all_files, args.num_samples)
    
    # Process files
    logger.info("Processing sampled files...")
    process_files(sampled_files, args.output_base_dir, args.model)
    
    # Run stacking script
    logger.info("Running stacking script...")
    stack_cmd = [
        'python', 'stack_json.py',
        '--input_dir', args.output_base_dir,
        '--output_dir', os.path.join(args.output_base_dir, 'combined')
    ]
    
    try:
        subprocess.run(stack_cmd, check=True)
        logger.info("Successfully completed stacking")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during stacking: {e}")

if __name__ == "__main__":
    main() 