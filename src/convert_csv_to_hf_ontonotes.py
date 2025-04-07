import os
import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel, Features, Sequence, Value
from sklearn.model_selection import train_test_split
import ast
import argparse

def rename_columns(df):
    if df.shape[1] < 3:
        raise ValueError("DataFrame must have at least 3 columns to rename.")
    df.columns.values[0] = "text"
    df.columns.values[1] = "bio_tags"
    df.columns.values[2] = "privacy_class_label"
    return df

def load_and_prep_df(args):
    # Build full path to dataset
    dataset_path = f"{args.dataset}"
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset not found at path: {dataset_path}")
    # Load the dataset
    print(f"Loading CSV data set ...")
    df = pd.read_csv(dataset_path)

    if args.rename_columns:
        print("Renaming first three columns...")
    
    print("Preview of loaded DataFrame:")
    print(df.head())
    print(df.columns)
    return df

def prepare_hf_dataset(args, df):
    # Ensure Correct Columns ===
    assert "text" in df.columns
    assert "bio_tags" in df.columns
    assert "privacy_class_label" in df.columns

    # Keep required columns ===
    keep_cols = ["text", "bio_tags", "privacy_class_label"]
    df = df[keep_cols].reset_index(drop=True)

    # Split into train/valid/test (75/15/10) ===
    train_df, temp_df = train_test_split(df, test_size=0.25, stratify=df["privacy_class_label"], random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.4, stratify=temp_df["privacy_class_label"], random_state=42)

    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)

    dataset = DatasetDict({
        "train": train_dataset,
        "validation": valid_dataset,
        "test": test_dataset
    })
    print(f"Saving HF dataset ....")
    dataset.save_to_disk(f"{args.target}")
    return dataset

def validate_bio_tags(hf_dataset):
    print(f"Validating HF dataset")
    unique_tags = set(tag for row in hf_dataset["train"]["bio_tags"] for tag, _ in ast.literal_eval(row))
    label_list = sorted(unique_tags)
    print(f"Unique BIO Tags extracted: {label_list}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Loads NER dataset in csv form  and converts to hf Dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="CSV dataset file name")
    parser.add_argument("--target", type=str, required=True, help="Target HuggingFace dataset name in the dataset directoy")
    parser.add_argument("--validate", action="store_true", help="Validate the BIO tags in dataset")
    parser.add_argument("--rename_columns", action="store_true", help="Rename the columns for the required name.")

    args = parser.parse_args()
    print("CSV to HF Convertor and Validator")
    print("----------------------------------")
    print(f"Source CSV file   : {args.dataset}")
    print(f"Target dir for HF : {args.target}")
    return args

if __name__ == "__main__":
    args = parse_arguments()
    df = load_and_prep_df(args)
    hf_dataset = prepare_hf_dataset(args, df)
    if args.validate:
        validate_bio_tags(hf_dataset)
