#!/usr/bin/env python3
"""
Medical Entity Injector for PII Detection Training

This script demonstrates a simplified approach to creating synthetic PII data
by injecting medical entities into Wikipedia text. It uses the existing 
dataset infrastructure but adds medical entities in a controlled way.
"""

import os
import random
import logging
import spacy
import pandas as pd
import re
from download_wikiner import download_dataset, extract_dataset, NerDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    logger.info("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Domain-Specific Named Entities (DS-NER) for medical domain
MEDICAL_CONDITIONS = [
    "type 1 diabetes", "asthma", "hypertension", "depression", 
    "anxiety disorder", "juvenile myoclonic epilepsy", "multiple sclerosis",
    "rheumatoid arthritis", "Crohn's disease", "fibromyalgia"
]

MEDICATIONS = [
    "Lisinopril", "Metformin", "Prozac", "Lexapro", "Depakote",
    "Adderall", "Synthroid", "Humira", "Enbrel", "Wellbutrin"
]

HOSPITALS = [
    "Mayo Clinic", "Cleveland Clinic", "Johns Hopkins Hospital",
    "Massachusetts General Hospital", "UCLA Medical Center",
    "Cedars-Sinai Medical Center", "Mount Sinai Hospital",
    "Stanford Health Care", "NYU Langone Hospitals", "Duke University Hospital"
]

# Patterns to inject medical entities naturally
INJECTION_PATTERNS = [
    "At age {age}, {pronoun} was diagnosed with {condition}.",
    "In {year}, {pronoun} began treatment for {condition}.",
    "Following a health scare, {pronoun} was treated at {hospital} for {condition}.",
    "{pronoun} has been taking {medication} for {condition} since {year}.",
    "After experiencing symptoms for several months, {pronoun} received a diagnosis of {condition}.",
    "{pronoun} spent two weeks at {hospital} recovering from complications related to {condition}.",
    "Doctors at {hospital} prescribed {medication} for {pronoun}'s {condition}."
]

def load_wiki_text():
    """Load text from Wikipedia dataset files to use for injection."""
    # Make sure the dataset exists
    if not os.path.exists("dataset_txt"):
        logger.info("Dataset not found. Downloading...")
        download_dataset()
        extract_dataset()
    
    # Get list of files
    text_files = []
    for root, dirs, files in os.walk("dataset_txt"):
        for file in files:
            if file.endswith(".txt") and not file.startswith("annot"):
                text_files.append(os.path.join(root, file))
    
    # Sample a few files
    if not text_files:
        logger.error("No text files found in dataset")
        return []
    
    sampled_files = random.sample(text_files, min(5, len(text_files)))
    
    # Read the content
    wiki_texts = []
    for file in sampled_files:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Split into paragraphs and sample a few
            paragraphs = content.split('\n\n')
            if paragraphs:
                wiki_texts.append(random.choice(paragraphs))
    
    return wiki_texts

def detect_person_mentions(text):
    """
    Find mentions of people in the text to contextualize injections.
    Returns a dict with pronoun information.
    """
    doc = nlp(text)
    
    # Look for PERSON entities
    persons = [ent for ent in doc.ents if ent.label_ == "PERSON"]
    
    # Default context
    context = {
        "pronoun": "they",
        "possessive": "their",
        "object": "them",
        "name": None
    }
    
    if persons:
        # Get the first person mentioned
        person = persons[0].text
        context["name"] = person
        
        # Check for gendered pronouns
        for token in doc:
            if token.lower_ in ["he", "his", "him"] and token.i > persons[0].start:
                context["pronoun"] = "he"
                context["possessive"] = "his"
                context["object"] = "him"
                break
            elif token.lower_ in ["she", "her", "hers"] and token.i > persons[0].start:
                context["pronoun"] = "she"
                context["possessive"] = "her"
                context["object"] = "her"
                break
    
    # Add random age and year
    context["age"] = random.randint(20, 60)
    context["year"] = random.randint(2000, 2020)
    
    return context

def inject_medical_entity(text):
    """
    Inject medical entities into text in a natural way.
    Returns modified text and entity information.
    """
    # Find a good place to inject (after a sentence)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) < 2:
        return text, None
    
    # Choose an injection point
    injection_point = random.randint(1, min(len(sentences) - 1, 5))
    
    # Get person context
    context = detect_person_mentions(text)
    
    # Choose random entities to inject
    condition = random.choice(MEDICAL_CONDITIONS)
    medication = random.choice(MEDICATIONS)
    hospital = random.choice(HOSPITALS)
    
    # Select and format injection pattern
    pattern = random.choice(INJECTION_PATTERNS)
    injected_sentence = pattern.format(
        pronoun=context["pronoun"].capitalize(),
        condition=condition,
        medication=medication,
        hospital=hospital,
        age=context["age"],
        year=context["year"]
    )
    
    # Insert the sentence
    sentences.insert(injection_point, injected_sentence)
    modified_text = " ".join(sentences)
    
    # Return the modified text and entity details
    entities = {
        "MED": [condition] if "{condition}" in pattern else [],
        "DRUG": [medication] if "{medication}" in pattern else [],
        "HSP": [hospital] if "{hospital}" in pattern else [],
        "injected_sentence": injected_sentence,
        "injection_point": injection_point
    }
    
    return modified_text, entities

def create_conll_format(text, entities):
    """
    Convert text with injected entities to CoNLL format with BIO tags.
    Returns list of tokens and their corresponding tags.
    """
    # Process the text with spaCy for proper tokenization
    doc = nlp(text)
    
    # Initialize lists for tokens and tags
    tokens = []
    tags = []
    
    # Process each sentence
    for sent in doc.sents:
        sent_tokens = [token.text for token in sent]
        sent_tags = ["O"] * len(sent_tokens)
        
        # Check for entities in this sentence
        sent_text = sent.text
        
        # Tag medical conditions
        for condition in entities.get("MED", []):
            condition_spans = find_spans(sent_text.lower(), condition.lower())
            for start, end in condition_spans:
                # Map character spans to token indices
                token_indices = char_span_to_token_indices(sent, start, end)
                if token_indices:
                    sent_tags[token_indices[0]] = "B-MED"
                    for i in token_indices[1:]:
                        sent_tags[i] = "I-MED"
        
        # Tag medications
        for medication in entities.get("DRUG", []):
            medication_spans = find_spans(sent_text.lower(), medication.lower())
            for start, end in medication_spans:
                # Map character spans to token indices
                token_indices = char_span_to_token_indices(sent, start, end)
                if token_indices:
                    sent_tags[token_indices[0]] = "B-DRUG"
                    for i in token_indices[1:]:
                        sent_tags[i] = "I-DRUG"
        
        # Tag hospitals
        for hospital in entities.get("HSP", []):
            hospital_spans = find_spans(sent_text.lower(), hospital.lower())
            for start, end in hospital_spans:
                # Map character spans to token indices
                token_indices = char_span_to_token_indices(sent, start, end)
                if token_indices:
                    sent_tags[token_indices[0]] = "B-HSP"
                    for i in token_indices[1:]:
                        sent_tags[i] = "I-HSP"
        
        # Add sentence tokens and tags
        tokens.extend(sent_tokens)
        tokens.append("")  # Empty line between sentences
        tags.extend(sent_tags)
        tags.append("")  # Empty line between sentences
    
    return tokens, tags

def find_spans(text, substring):
    """Find all spans of a substring in text."""
    spans = []
    start = 0
    while True:
        start = text.find(substring, start)
        if start == -1:
            break
        spans.append((start, start + len(substring)))
        start += 1
    return spans

def char_span_to_token_indices(doc_span, start_char, end_char):
    """Convert character spans to token indices within a spaCy span."""
    indices = []
    for i, token in enumerate(doc_span):
        # Get character offsets within the sentence
        token_start = token.idx - doc_span.start_char
        token_end = token_start + len(token.text)
        
        # Check for overlap
        if (token_start <= end_char and token_end > start_char):
            indices.append(i)
    
    return indices

def save_to_conll_file(tokens, tags, output_file):
    """Save tokens and tags to a file in CoNLL format."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for token, tag in zip(tokens, tags):
            if token:
                f.write(f"{token} {tag}\n")
            else:
                f.write("\n")  # Empty line for sentence boundaries

def main():
    """Main function to demonstrate synthetic medical entity injection."""
    # Create output directory
    os.makedirs("medical_synthetic_data", exist_ok=True)
    
    # Load Wikipedia text
    logger.info("Loading Wikipedia text...")
    wiki_texts = load_wiki_text()
    
    if not wiki_texts:
        logger.error("No text found to process")
        return
    
    # Process each text
    all_tokens = []
    all_tags = []
    
    for i, text in enumerate(wiki_texts):
        logger.info(f"Processing text {i+1}/{len(wiki_texts)}")
        
        # Inject medical entity
        modified_text, entities = inject_medical_entity(text)
        
        if entities:
            # Convert to CoNLL format
            tokens, tags = create_conll_format(modified_text, entities)
            
            # Save individual samples
            output_file = f"medical_synthetic_data/sample_{i+1}.txt"
            save_to_conll_file(tokens, tags, output_file)
            
            # Add to combined dataset
            all_tokens.extend(tokens)
            all_tags.extend(tags)
            
            # Print example
            if i == 0:
                injected = entities.get("injected_sentence", "No injection")
                logger.info(f"Example injection: {injected}")
                
                # Show a sample of the CoNLL format
                sample = zip(tokens[:50], tags[:50])
                sample_str = "\n".join([f"{t} {tag}" for t, tag in sample if t])
                logger.info(f"CoNLL format sample:\n{sample_str}")
    
    # Save combined dataset
    if all_tokens:
        combined_file = "medical_synthetic_data/combined_dataset.txt"
        save_to_conll_file(all_tokens, all_tags, combined_file)
        logger.info(f"Created {len(wiki_texts)} samples in medical_synthetic_data/")
        logger.info(f"Combined dataset saved to {combined_file}")
    else:
        logger.warning("No data was generated")

if __name__ == "__main__":
    main() 