import os
import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel, Features, Sequence, Value
from sklearn.model_selection import train_test_split
import ast
import argparse
import re
import hashlib

PERSONAL_TAGS = {"B-PERSON", "I-PERSON"}

def hash_prefix(text, prefix_len):
    base = text[:prefix_len].lower().strip()
    return hashlib.md5(base.encode()).hexdigest()

def get_label_pairs(df, prefix_len=100):
    df.loc[:, "prefix_hash"] = df["text"].apply(lambda x: hash_prefix(x, prefix_len))
    grouped = df.groupby("prefix_hash")
    pair_indices = []
    for _, group in grouped:
        labels = group["privacy_class_label"].unique()
        if set(labels) == {0, 1}:
            pair_indices.extend(group.index.tolist())
    df_pairs = df.loc[pair_indices].drop(columns=["prefix_hash"]).reset_index(drop=True)
    return df_pairs

def verify_bio_tags(df):
    all_tags = set()
    for row in df["bio_tags"]:
        try:
            all_tags.update(tag for tag, _ in ast.literal_eval(row))
        except:
            continue
    return sorted(all_tags)

def derive_pii_tags(all_tags):
    return {
        tag for tag in all_tags
        if tag.startswith(("B-", "I-"))
        and tag not in PERSONAL_TAGS
        and tag != "O"
    }

def compute_tag_stats(bio_tags_str, pii_tags):
    try:
        tags = [tag for tag, _ in ast.literal_eval(bio_tags_str)]
    except Exception:
        return 0, 0, 0, 0.0
    personal_count = sum(1 for tag in tags if tag in PERSONAL_TAGS)
    pii_count = sum(1 for tag in tags if tag in pii_tags)
    non_o_tags = sum(1 for tag in tags if tag != "O")
    tag_ratio = (personal_count + pii_count) / len(tags) if tags else 0.0
    return personal_count, pii_count, non_o_tags, tag_ratio

def sample_high_quality_normal_text(df, df_pairs, pii_tags, n=10000, min_words=50, min_tags=3):
    used_texts = set(df_pairs["text"])
    df_0 = df[df["privacy_class_label"] == 0].copy()
    df_0 = df_0[~df_0["text"].isin(used_texts)]
    # Compute word and tag stats
    df_0["word_count"] = df_0["text"].apply(lambda x: len(x.split()))
    tag_stats = df_0["bio_tags"].apply(lambda x: compute_tag_stats(x, pii_tags))
    df_0[["personal_tags", "pii_tags", "tag_count", "tag_ratio"]] = pd.DataFrame(tag_stats.tolist(), index=df_0.index)
    # Filter based on thresholds
    #print("Before tag filter:", len(df_0))
    #print(f"After word_count >= {min_words}:", len(df_0[df_0['word_count'] >= min_words]))
    #print("After tag_count >= 3:", len(df_0[df_0['tag_count'] >= 3]))

    df_0 = df_0[
        (df_0["word_count"] >= min_words) &
        (df_0["tag_count"] >= min_tags)
    ]
    # Sample if needed
    df_sample = df_0.sample(n=n, random_state=42) if len(df_0) > n else df_0
    return df_sample.reset_index(drop=True)

def rename_columns(df):
    if df.shape[1] < 3:
        raise ValueError("DataFrame must have at least 3 columns to rename.")
    df.columns.values[0] = "text"
    df.columns.values[1] = "bio_tags"
    df.columns.values[2] = "privacy_class_label"
    return df

def is_english_like(text):
    if not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < 10:
        return False
    # Must contain at least one English alphabet letter
    return bool(re.search(r'[A-Za-z]', text))

def clean_dataset(df, drop_exact=True):  
    orig_size = len(df) 
    print(f"Original size: {orig_size}")
    
    df_clean = df[df["text"].apply(is_english_like)]
    print(f"✔️ Retained {len(df_clean)} clean rows out of {len(df)} total")
    df = df_clean

    if drop_exact:
        df = df.drop_duplicates(subset=["text", "privacy_class_label"])
        print(f"After removing exact duplicates: {len(df)}")
    pairs = get_label_pairs(df)
    bios = verify_bio_tags(df)
    PII_TAGS = {tag for tag in bios if tag.startswith(("B-", "I-")) and tag not in PERSONAL_TAGS and tag != "O"}
    label0_df = sample_high_quality_normal_text(df, pairs, pii_tags=PII_TAGS, n=20000)

    df_label1_all = df[df["privacy_class_label"] == 1]
    used_texts = set(pairs["text"])
    label1_df = df_label1_all[~df_label1_all["text"].isin(used_texts)].copy()

    print(f"Privacy class 0/1 pairs              : {len(pairs)}")
    print(f"Privacy class 0 High Quality Sampled : {len(label0_df)}")
    print(f"Privacy class 1 non paired           : {len(label1_df)}")
    df = pd.concat([pairs, label0_df, label1_df])

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
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
        df = rename_columns(df)
    
    df = clean_dataset(df)
    df['text_len'] = df['text'].apply(lambda x: len(x))
    print(f"Max input text: {max(df['text_len'])}")
    print(f"95th percentile: {df['text_len'].quantile(0.95)}")
    print(f"90th percentile: {df['text_len'].quantile(0.90)}")
    print(f"50th percentile: {df['text_len'].quantile(0.50)}")
    df = df[["text", "bio_tags", "privacy_class_label"]].copy()
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    print("Preview of loaded DataFrame:")
    print(df.head())
    return df

def prepare_hf_dataset(args, df):
    # Ensure Correct Columns ===
    assert "text" in df.columns
    assert "bio_tags" in df.columns
    assert "privacy_class_label" in df.columns

    # Keep required columns ===
    keep_cols = ["text", "bio_tags", "privacy_class_label"]
    df = df[keep_cols].reset_index(drop=True)
    train_df, temp_df = train_test_split(df, train_size=args.train_test_split_ratio, stratify=df["privacy_class_label"], random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=args.test_valid_split_ratio, stratify=temp_df["privacy_class_label"], random_state=42)

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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Loads NER dataset in csv form  and converts to hf Dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="CSV dataset file name including path")
    parser.add_argument("--target", type=str, required=True, help="Target path for creating (must not exist) for the HF Dataset")
    parser.add_argument("--train_test_split_ratio", type=float, default=0.8, help="Split the dataset into train test split ration")
    parser.add_argument("--test_valid_split_ratio", type=float, default=0.5, help="Split the test dataset into test valid split ration")
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
        bios = verify_bio_tags(df)
        print(f"List of bio tag: {bios}")

