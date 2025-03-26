#!/usr/bin/env python3
"""
Download and prepare the Wikiner dataset for ML training.
This script downloads the pre-processed dataset and sets it up in the correct format
for training a Named Entity Recognition model.
"""

import os
import wget
import tarfile
import logging
from torch.utils import data
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NerDataset(data.Dataset):
    """Dataset class for loading NER data in CoNLL format."""
    def __init__(self, fpath):
        """
        Initialize the dataset from a file path.
        Args:
            fpath: Path to the CoNLL format text file
        """
        entries = open(fpath, 'r').read().strip().split("\n\n")
        sents, tags_li = [], []  # list of lists
        for entry in entries:
            lines = entry.splitlines()
            words = [line.split()[0] for line in entry.splitlines() if len(line.split()) > 1]
            tags = ([line.split()[-1] for line in entry.splitlines() if len(line.split()) > 1])
            if not (len(tags) != 0 and tags.count(tags[0]) == len(tags)):
                sents.append(["[CLS]"] + words + ["[SEP]"])
                tags_li.append(["<PAD>"] + tags + ["<PAD>"])
        self.sents, self.tags_li = sents, tags_li

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx]
        return words, tags

    def append(self, other):
        """Append another dataset to this one."""
        self.sents.extend(other.sents)
        self.tags_li.extend(other.tags_li)

def download_dataset():
    """Download the Wikiner dataset if it doesn't exist."""
    dataset_url = 'https://archive.org/download/wikiner_dataset_csv.tar/wikiner_dataset_txt.tar.gz'
    dataset_file = 'wikiner_dataset_txt.tar.gz'
    
    if not os.path.exists(dataset_file):
        logger.info("Downloading Wikiner dataset...")
        wget.download(dataset_url)
        logger.info("\nDownload complete!")
    else:
        logger.info("Dataset file already exists.")

def extract_dataset():
    """Extract the downloaded dataset."""
    dataset_file = 'wikiner_dataset_txt.tar.gz'
    if os.path.exists(dataset_file):
        logger.info("Extracting dataset...")
        with tarfile.open(dataset_file, mode='r') as tar:
            tar.extractall('./dataset_txt')
        logger.info("Extraction complete!")
    else:
        logger.error("Dataset file not found. Please run download_dataset() first.")

def get_dataset_splits():
    """Get the training, validation, and test datasets."""
    logger.info("Loading dataset splits...")
    
    # Get all annotation files
    paths_annot = sorted([os.path.join(f[0], name) for f in os.walk('./dataset_txt')
                    if len(f[2])!=0 for name in f[2] if os.path.splitext(name)[-1] == '.txt' and name.split('_')[0]=='annot'],
                   key=lambda path: int(path.split('_')[-1].split('.')[0]))
    
    # Split into train/val/test
    training_files = [path for path in paths_annot if 'training' in path.split('/')]
    eval_files = [path for path in paths_annot if 'eval' in path.split('/')]
    test_files = [path for path in paths_annot if 'test' in path.split('/')]
    
    # Create datasets
    train_dataset = NerDataset(training_files[0])
    for train_file in training_files[1:]:
        train_dataset.append(NerDataset(train_file))
    
    eval_dataset = NerDataset(eval_files[0])
    for eval_file in eval_files[1:]:
        eval_dataset.append(NerDataset(eval_file))
    
    test_dataset = NerDataset(test_files[0])
    for test_file in test_files[1:]:
        test_dataset.append(NerDataset(test_file))
    
    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}, Test: {len(test_dataset)}")
    return train_dataset, eval_dataset, test_dataset

def main():
    """Main function to download and prepare the dataset."""
    # Create necessary directories
    os.makedirs('./dataset_txt', exist_ok=True)
    
    # Download and extract dataset
    download_dataset()
    extract_dataset()
    
    # Get dataset splits
    train_dataset, eval_dataset, test_dataset = get_dataset_splits()
    
    logger.info("Dataset preparation complete!")
    return train_dataset, eval_dataset, test_dataset

if __name__ == "__main__":
    main() 