import argparse
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from privacy_ner_attend_pii_train import PrivacyDetectionModel
from transformers import AutoTokenizer
from tqdm import tqdm

from privacy_ner_attend_pii_train import parse_arguments
from privacy_ner_attend_pii_train import load_hf_dataset
from privacy_ner_attend_pii_train import get_base_tokenizer
from privacy_ner_attend_pii_train import prepare_dataset, custom_collate_fn

# Load model from disk
def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model_args = checkpoint['model_args']
    print(model_args)
    model = PrivacyDetectionModel(**model_args)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

# Evaluate model on test dataset
def evaluate_model(model, dataset, batch_size, device, compute_metrics):
    predictions = []
    labels = []
    test_loader = DataLoader(
        dataset["test"],  
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn)

    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs['logits_privacy'].argmax(dim=-1).cpu().tolist()
            predictions.extend(preds)
            labels.extend(batch["privacy_labels"])

    if compute_metrics:
        correct = sum(p == l for p, l in zip(predictions, labels))
        accuracy = correct / len(labels)
        print(f"Accuracy: {accuracy:.8f}")

    return predictions,labels

# Main testing function
def main():
    MODEL   = "privacy_attend_ner_model"
    args = parse_arguments(MODEL)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Loading model from {args.model_file}")
    model = load_model(args.model_file, device)

    print(f"Loading dataset from {args.hf_dataset}")

    dataset = load_hf_dataset(args)
    tokenizer = get_base_tokenizer(args)
    ( 
        personal_label2id, 
        personal_id2label, 
        pii_label2id, 
        pii_id2label,
        tokenized_dataset
    ) =  prepare_dataset(args, tokenizer, dataset)

    print("Starting evaluation...")
    predictions, labels = evaluate_model(model, tokenized_dataset, args.batch_size, device, args.compute_metrics)

    print(f"Predictions (first 10): {predictions[:10]}")
    print(f"Predictions (Lables 10): {labels[:10]}")

if __name__ == "__main__":
    main()
