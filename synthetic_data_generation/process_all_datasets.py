#!/usr/bin/env python3
"""
Dataset Processing Coordinator

This script coordinates the processing of all datasets (training, test, eval)
using the medical_entity_injector_llm_new.py script.
"""

import os
import subprocess
import argparse
import logging
import time
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_directory(data_dir, output_dir, model, num_files=None, script_path="medical_entity_injector_llm_new.py", batch_size=100):
    """
    Process a single dataset directory with the medical entity injector.
    
    Args:
        data_dir: Path to input dataset directory
        output_dir: Path to output directory
        model: LLM model to use
        num_files: Number of files to process (None for all)
        script_path: Path to the medical entity injector script
        batch_size: Number of files to process in each batch
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Count files in directory to process
    files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    total_files = len(files)
    
    if num_files is None:
        num_files = total_files
    else:
        num_files = min(num_files, total_files)
    
    logger.info(f"Processing {num_files} files from {data_dir} in batches of {batch_size}")
    
    # Process in batches
    success_count = 0
    batches = (num_files + batch_size - 1) // batch_size  # Ceiling division
    
    for batch in range(batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, num_files)
        batch_size_actual = end_idx - start_idx
        
        # Create temporary directory with symbolic links to files for this batch
        batch_dir = os.path.join(output_dir, f"batch_{batch}_input")
        os.makedirs(batch_dir, exist_ok=True)
        
        # Create symbolic links to the actual files
        for i in range(start_idx, end_idx):
            src_file = os.path.join(data_dir, files[i])
            dst_link = os.path.join(batch_dir, files[i])
            
            # Create a copy instead of a symlink to avoid permission issues
            with open(src_file, 'r') as src:
                with open(dst_link, 'w') as dst:
                    dst.write(src.read())
        
        # Build command for this batch
        batch_output_dir = os.path.join(output_dir, f"batch_{batch}_output")
        os.makedirs(batch_output_dir, exist_ok=True)
        
        cmd = [
            "python", script_path,
            "--dataset_dir", batch_dir,
            "--num_samples", str(batch_size_actual),
            "--model", model,
            "--output_dir", batch_output_dir
        ]
        
        # Run command
        logger.info(f"Processing batch {batch+1}/{batches} ({start_idx+1}-{end_idx}/{num_files})")
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                logger.info(f"Successfully processed batch {batch+1}/{batches}")
                success_count += 1
                
                # Copy results from batch output to main output
                for output_file in os.listdir(batch_output_dir):
                    src_path = os.path.join(batch_output_dir, output_file)
                    dst_path = os.path.join(output_dir, output_file)
                    
                    if os.path.isfile(src_path):
                        with open(src_path, 'r') as src:
                            with open(dst_path, 'w') as dst:
                                dst.write(src.read())
            else:
                logger.error(f"Failed to process batch {batch+1}/{batches}")
                logger.error(f"Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"Batch {batch+1}/{batches} timed out after 1 hour")
        except Exception as e:
            logger.error(f"Error processing batch {batch+1}/{batches}: {e}")
        
        # Clean up temporary directories
        import shutil
        try:
            shutil.rmtree(batch_dir)
            shutil.rmtree(batch_output_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directories: {e}")
    
    return success_count == batches

def main():
    """Main function to coordinate processing of all dataset directories."""
    parser = argparse.ArgumentParser(description="Process all dataset directories")
    parser.add_argument("--input_base_dir", type=str, 
                       default="dataset_txt/dataset_txt",
                       help="Base directory containing dataset subdirectories")
    parser.add_argument("--output_base_dir", type=str, 
                       default="processed_dataset",
                       help="Base directory for output")
    parser.add_argument("--model", type=str, 
                       default="mistral",
                       help="Ollama model to use")
    parser.add_argument("--files_per_dir", type=int, 
                       default=100,
                       help="Number of files to process per directory (None for all)")
    parser.add_argument("--batch_size", type=int,
                       default=10,
                       help="Number of files to process in each batch")
    parser.add_argument("--script_path", type=str,
                       default="medical_entity_injector_llm_new.py",
                       help="Path to medical entity injector script")
    
    args = parser.parse_args()
    
    # Directory statistics
    dir_sizes = {
        "training": 20039,
        "test": 307,
        "eval": 2744
    }
    
    # Print dataset information
    logger.info("Dataset Information:")
    for dir_name, size in dir_sizes.items():
        logger.info(f"  {dir_name}: {size} files")
    
    # Confirm with user
    if args.files_per_dir is None:
        total_files = sum(dir_sizes.values())
    else:
        total_files = min(args.files_per_dir, dir_sizes["training"]) + \
                     min(args.files_per_dir, dir_sizes["test"]) + \
                     min(args.files_per_dir, dir_sizes["eval"])
    
    logger.info(f"Will process approximately {total_files} files in total")
    logger.info(f"Using model: {args.model}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Get start time
    start_time = time.time()
    
    # Process each dataset directory
    dataset_dirs = ["training", "test", "eval"]
    success_count = 0
    
    for dir_name in dataset_dirs:
        input_dir = os.path.join(args.input_base_dir, dir_name)
        output_dir = os.path.join(args.output_base_dir, dir_name)
        
        if os.path.exists(input_dir):
            dir_start_time = time.time()
            logger.info(f"Starting processing of {dir_name} directory")
            
            success = process_directory(
                input_dir, 
                output_dir, 
                args.model, 
                args.files_per_dir,
                args.script_path,
                args.batch_size
            )
            
            dir_end_time = time.time()
            dir_duration = dir_end_time - dir_start_time
            
            logger.info(f"Completed processing of {dir_name} directory in {dir_duration:.2f} seconds")
            
            if success:
                success_count += 1
        else:
            logger.warning(f"Directory {input_dir} does not exist, skipping")
    
    # Get end time and calculate duration
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info(f"Processing complete. Successfully processed {success_count}/{len(dataset_dirs)} directories")
    logger.info(f"Total processing time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    # Print summary of output files
    logger.info("Output Summary:")
    total_output_files = 0
    for dir_name in dataset_dirs:
        output_dir = os.path.join(args.output_base_dir, dir_name)
        if os.path.exists(output_dir):
            file_count = len([f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))])
            total_output_files += file_count
            logger.info(f"  {dir_name}: {file_count} files")
    
    logger.info(f"Total output files: {total_output_files}")

if __name__ == "__main__":
    main() 