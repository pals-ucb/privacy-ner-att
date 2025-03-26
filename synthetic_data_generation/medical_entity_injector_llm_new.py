#!/usr/bin/env python3
"""
New LLM-based Medical Entity Injector

This script reads CoNLL formatted files from dataset_txt folder and
processes them for medical entity injection.
"""

import os
import logging
import argparse
import random
import re
import json
import requests
from typing import Dict, List, Tuple, Optional, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMClient:
    """Base class for LLM clients."""
    
    def generate(self, prompt: str) -> str:
        """Generate text based on prompt."""
        raise NotImplementedError

class OllamaClient(LLMClient):
    """Client for Ollama API."""
    
    def __init__(self, model: str = "mistral"):
        self.model = model
        self.api_url = "http://localhost:11434/api/generate"
        logger.info(f"Initialized Ollama client with model: {model}")
        
        # Define diverse medical entities for fallback
        self.conditions = [
            "Type 1 Diabetes", "Type 2 Diabetes", "Hypertension", "Asthma", "Migraine", 
            "Rheumatoid Arthritis", "Crohn's Disease", "Multiple Sclerosis", "Lupus", 
            "Fibromyalgia", "Chronic Fatigue Syndrome", "Psoriasis", "Eczema", 
            "Celiac Disease", "Hypothyroidism", "Hyperthyroidism", "ADHD", 
            "Generalized Anxiety Disorder", "Depression", "Bipolar Disorder", 
            "Schizophrenia", "PTSD", "OCD", "Epilepsy", "Parkinson's Disease", 
            "Alzheimer's Disease", "Coronary Artery Disease", "Heart Failure", 
            "Stroke", "COPD", "Anemia", "Osteoporosis", "Osteoarthritis", 
            "Gout", "Chronic Kidney Disease", "Sleep Apnea", "Glaucoma", 
            "Macular Degeneration", "Cataracts", "Hepatitis", "Cirrhosis"
        ]
        self.medications = [
            "Lisinopril", "Metformin", "Atorvastatin", "Levothyroxine", "Albuterol", 
            "Omeprazole", "Amlodipine", "Metoprolol", "Losartan", "Gabapentin", 
            "Hydrochlorothiazide", "Sertraline", "Simvastatin", "Montelukast", 
            "Escitalopram", "Fluoxetine", "Pantoprazole", "Prednisone", "Ibuprofen", 
            "Acetaminophen", "Aspirin", "Furosemide", "Tramadol", "Amoxicillin", 
            "Azithromycin", "Insulin Glargine", "Insulin Lispro", "Citalopram", 
            "Duloxetine", "Venlafaxine", "Alprazolam", "Clonazepam", "Trazodone", 
            "Warfarin", "Carvedilol", "Clopidogrel", "Meloxicam", "Cyclobenzaprine", 
            "Methylphenidate", "Bupropion"
        ]
        self.hospitals = [
            "Mayo Clinic", "Cleveland Clinic", "Johns Hopkins Hospital", "Massachusetts General Hospital",
            "New York-Presbyterian Hospital", "UCLA Medical Center", "Stanford Health Care",
            "Mount Sinai Hospital", "Cedars-Sinai Medical Center", "University of Michigan Hospitals",
            "UCSF Medical Center", "Northwestern Memorial Hospital", "NYU Langone Hospitals",
            "Brigham and Women's Hospital", "Barnes-Jewish Hospital", "Houston Methodist Hospital",
            "University of Pennsylvania Hospital", "Rush University Medical Center", "Yale New Haven Hospital",
            "Baylor St. Luke's Medical Center", "City of Hope Medical Center", "UC San Diego Health",
            "Duke University Hospital", "Emory University Hospital", "Vanderbilt University Medical Center",
            "Kaiser Permanente Los Angeles Medical Center", "St. Jude Children's Research Hospital",
            "Memorial Sloan Kettering Cancer Center", "MD Anderson Cancer Center", "Boston Children's Hospital",
            "Great Ormond Street Hospital", "Royal London Hospital", "St Thomas' Hospital",
            "Auckland City Hospital", "Sydney Children's Hospital", "Toronto General Hospital",
            "The Ottawa Hospital", "Montreal General Hospital", "Singapore General Hospital",
            "Tokyo Medical University Hospital"
        ]
        self.templates = [
            " In recent years, they were diagnosed with [MED]{condition}[/MED] and prescribed [DRUG]{medication}[/DRUG]. They received treatment at [HSP]{hospital}[/HSP].",
            " They have been managing [MED]{condition}[/MED] for several years with [DRUG]{medication}[/DRUG], regularly visiting [HSP]{hospital}[/HSP] for check-ups.",
            " After experiencing symptoms of [MED]{condition}[/MED], they consulted specialists at [HSP]{hospital}[/HSP] who prescribed [DRUG]{medication}[/DRUG].",
            " Their health history includes [MED]{condition}[/MED], for which they take [DRUG]{medication}[/DRUG] and receive care at [HSP]{hospital}[/HSP].",
            " During a routine visit to [HSP]{hospital}[/HSP], doctors discovered they had [MED]{condition}[/MED] and recommended [DRUG]{medication}[/DRUG] as treatment.",
            " They were referred to [HSP]{hospital}[/HSP] after showing signs of [MED]{condition}[/MED] and were subsequently prescribed [DRUG]{medication}[/DRUG] to manage their symptoms.",
            " Following a diagnosis of [MED]{condition}[/MED] at [HSP]{hospital}[/HSP], they began a treatment regimen including [DRUG]{medication}[/DRUG] with positive results.",
            " Their medical records at [HSP]{hospital}[/HSP] indicate they have been treated for [MED]{condition}[/MED] with regular doses of [DRUG]{medication}[/DRUG].",
            " Despite being diagnosed with [MED]{condition}[/MED], they've maintained their activities with help from [DRUG]{medication}[/DRUG] prescribed by specialists at [HSP]{hospital}[/HSP]."
        ]
    
    def generate(self, prompt: str) -> str:
        """Generate text using Ollama API."""
        try:
            # First, try the simplified approach using command line
            import subprocess
            cmd = ["ollama", "run", self.model, prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return result.stdout.strip()
            
            logger.warning(f"Command-line Ollama failed: {result.stderr}")
            
            # Fall back to API if command-line fails
            response = requests.post(
                self.api_url,
                json={"model": self.model, "prompt": prompt},
                stream=False
            )
            
            if response.status_code == 200:
                try:
                    return response.json().get("response", "")
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing JSON from Ollama API: {e}")
            
            # If all attempts fail, use fallback with diverse entities
            logger.warning("Using fallback response with diverse entities")
            
            # Try to extract original paragraph from prompt
            orig_para = ""
            try:
                orig_para = prompt.split('"')[1]
            except:
                pass
            
            # Generate diverse fallback
            condition = random.choice(self.conditions)
            medication = random.choice(self.medications)
            hospital = random.choice(self.hospitals)
            template = random.choice(self.templates)
            
            fallback = template.format(condition=condition, medication=medication, hospital=hospital)
            return orig_para + fallback
            
        except Exception as e:
            logger.error(f"Failed to generate text: {e}")
            # Create a fallback response with diverse entities
            try:
                orig_para = prompt.split('"')[1]
            except:
                orig_para = "Original paragraph extraction failed."
            
            condition = random.choice(self.conditions)
            medication = random.choice(self.medications)
            hospital = random.choice(self.hospitals)
            template = random.choice(self.templates)
            
            fallback = template.format(condition=condition, medication=medication, hospital=hospital)
            return orig_para + fallback

def read_conll_file(file_path: str) -> List[List[str]]:
    """
    Read a CoNLL formatted file and returns a list of lines grouped by paragraphs.
    
    Args:
        file_path: Path to the CoNLL formatted file
        
    Returns:
        List of paragraphs, where each paragraph is a list of lines
    """
    paragraphs = []
    current_paragraph = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:  # Blank line → New paragraph
                if current_paragraph:
                    paragraphs.append(current_paragraph)
                    current_paragraph = []
            else:
                current_paragraph.append(line)

    # Append last paragraph if not empty
    if current_paragraph:
        paragraphs.append(current_paragraph)

    return paragraphs

def reconstruct_paragraphs(conll_paragraphs: List[List[str]]) -> List[str]:
    """
    Reconstruct text paragraphs from CoNLL formatted paragraphs.
    
    Args:
        conll_paragraphs: List of paragraphs, where each paragraph is a list of CoNLL lines
        
    Returns:
        List of reconstructed text paragraphs
    """
    reconstructed = []
    
    for paragraph in conll_paragraphs:
        words = [line.split()[0] for line in paragraph]
        reconstructed.append(" ".join(words))
    
    return reconstructed

def inject_medical_entities(text_paragraphs: List[str], llm_client: LLMClient) -> Tuple[List[str], Dict[str, List[Dict[str, Any]]]]:
    """
    Inject medical entities into text paragraphs using an LLM.
    
    Args:
        text_paragraphs: List of text paragraphs
        llm_client: LLM client to use for generation
        
    Returns:
        Tuple of (modified paragraphs, entity data)
    """
    modified_paragraphs = []
    entity_data = {
        "MED": [],  # Medical conditions
        "DRUG": [], # Medications
        "HSP": []   # Hospitals
    }
    
    # Define known entities
    known_entities = {
        "MED": [
            "Type 1 Diabetes", "Type 2 Diabetes", "Hypertension", "Asthma", "Migraine", 
            "Rheumatoid Arthritis", "Crohn's Disease", "Multiple Sclerosis", "Lupus", 
            "Fibromyalgia", "Chronic Fatigue Syndrome", "Psoriasis", "Eczema", 
            "Celiac Disease", "Hypothyroidism", "Hyperthyroidism", "ADHD", 
            "Generalized Anxiety Disorder", "Depression", "Bipolar Disorder", 
            "Schizophrenia", "PTSD", "OCD", "Epilepsy", "Parkinson's Disease", 
            "Alzheimer's Disease", "Coronary Artery Disease", "Heart Failure", 
            "Stroke", "COPD", "Anemia", "Osteoporosis", "Osteoarthritis", 
            "Gout", "Chronic Kidney Disease", "Sleep Apnea", "Glaucoma", 
            "Macular Degeneration", "Cataracts", "Hepatitis", "Cirrhosis"
        ],
        "DRUG": [
            "Lisinopril", "Metformin", "Atorvastatin", "Levothyroxine", "Albuterol", 
            "Omeprazole", "Amlodipine", "Metoprolol", "Losartan", "Gabapentin", 
            "Hydrochlorothiazide", "Sertraline", "Simvastatin", "Montelukast", 
            "Escitalopram", "Fluoxetine", "Pantoprazole", "Prednisone", "Ibuprofen", 
            "Acetaminophen", "Aspirin", "Furosemide", "Tramadol", "Amoxicillin", 
            "Azithromycin", "Insulin Glargine", "Insulin Lispro", "Citalopram", 
            "Duloxetine", "Venlafaxine", "Alprazolam", "Clonazepam", "Trazodone", 
            "Warfarin", "Carvedilol", "Clopidogrel", "Meloxicam", "Cyclobenzaprine", 
            "Methylphenidate", "Bupropion"
        ],
        "HSP": [
            "Mayo Clinic", "Cleveland Clinic", "Johns Hopkins Hospital", "Massachusetts General Hospital",
            "New York-Presbyterian Hospital", "UCLA Medical Center", "Stanford Health Care",
            "Mount Sinai Hospital", "Cedars-Sinai Medical Center", "University of Michigan Hospitals",
            "UCSF Medical Center", "Northwestern Memorial Hospital", "NYU Langone Hospitals",
            "Brigham and Women's Hospital", "Barnes-Jewish Hospital", "Houston Methodist Hospital",
            "University of Pennsylvania Hospital", "Rush University Medical Center", "Yale New Haven Hospital",
            "Baylor St. Luke's Medical Center", "City of Hope Medical Center", "UC San Diego Health",
            "Duke University Hospital", "Emory University Hospital", "Vanderbilt University Medical Center",
            "Kaiser Permanente Los Angeles Medical Center", "St. Jude Children's Research Hospital",
            "Memorial Sloan Kettering Cancer Center", "MD Anderson Cancer Center", "Boston Children's Hospital",
            "Great Ormond Street Hospital", "Royal London Hospital", "St Thomas' Hospital",
            "Auckland City Hospital", "Sydney Children's Hospital", "Toronto General Hospital",
            "The Ottawa Hospital", "Montreal General Hospital", "Singapore General Hospital",
            "Tokyo Medical University Hospital"
        ]
    }
    
    # Choose a random paragraph for injection
    if len(text_paragraphs) > 1:
        target_idx = random.randint(0, len(text_paragraphs) - 1)
    else:
        target_idx = 0
    
    for i, paragraph in enumerate(text_paragraphs):
        if i == target_idx:
            # Select random medical entities for the prompt
            random_condition = random.choice(known_entities["MED"])
            random_drug = random.choice(known_entities["DRUG"])
            random_hospital = random.choice(known_entities["HSP"])

            # Construct prompt for the LLM with the random entities as examples
            prompt = f"""
            Here is a paragraph about a person:
            
            "{paragraph}"
            
            Rewrite this paragraph to include one or more of:
            1. A medical condition they have (tag as MED)
            2. A medication they take (tag as DRUG)
            3. A hospital they visited (tag as HSP)
            
            Be creative and diverse in your choice of medical conditions, medications, and hospitals.
            Choose from a wide variety of common and specific medical terms.
            
            Keep the original information intact. Add 1-2 sentences naturally.
            Format your response as a single paragraph with entities marked like this:
            They were diagnosed with [MED]{random_condition}[/MED] and prescribed [DRUG]{random_drug}[/DRUG]. They were treated at [HSP]{random_hospital}[/HSP].
            
            Only return the paragraph, no quotation marks, no other text.
            """
            
            # Generate modified paragraph
            response = llm_client.generate(prompt)
            
            # Clean up response (in case the LLM adds extra commentary)
            response = response.strip()
            # Remove quotation marks at the beginning and end
            if response.startswith('"') or response.startswith('"') or response.startswith("'"):
                response = response[1:]
            if response.endswith('"') or response.endswith('"') or response.endswith("'"):
                response = response[:-1]
            modified_paragraph = response
            
            # Extract entities using regex patterns
            med_entities = re.findall(r'\[MED\](.*?)\[/MED\]', modified_paragraph)
            drug_entities = re.findall(r'\[DRUG\](.*?)\[/DRUG\]', modified_paragraph)
            hsp_entities = re.findall(r'\[HSP\](.*?)\[/HSP\]', modified_paragraph)
            
            # Store entity data with start and end positions
            for entity_type, entities in [("MED", med_entities), ("DRUG", drug_entities), ("HSP", hsp_entities)]:
                for entity in entities:
                    start_idx = modified_paragraph.find(f"[{entity_type}]{entity}[/{entity_type}]")
                    
                    # Only add if we found the entity
                    if start_idx >= 0:
                        entity_data[entity_type].append({
                            "text": entity,
                            "paragraph_idx": i,
                            "span": (start_idx, start_idx + len(entity)),
                            "type": entity_type
                        })
            
            # If no tagged entities were found, try to detect known entities
            if not any(entity_data.values()):
                logger.info("No tagged entities found, trying to detect known entities")
                for entity_type, entities in known_entities.items():
                    for entity in entities:
                        if entity.lower() in modified_paragraph.lower():
                            # Find the exact position with case-insensitive search
                            start_idx = re.search(entity, modified_paragraph, re.IGNORECASE)
                            if start_idx:
                                actual_text = modified_paragraph[start_idx.start():start_idx.end()]
                                entity_data[entity_type].append({
                                    "text": actual_text,
                                    "paragraph_idx": i,
                                    "span": (start_idx.start(), start_idx.end()),
                                    "type": entity_type
                                })
                                logger.info(f"Detected untagged entity: {actual_text} ({entity_type})")
            
            # Remove the tags for the final text
            modified_paragraph = re.sub(r'\[MED\](.*?)\[/MED\]', r'\1', modified_paragraph)
            modified_paragraph = re.sub(r'\[DRUG\](.*?)\[/DRUG\]', r'\1', modified_paragraph)
            modified_paragraph = re.sub(r'\[HSP\](.*?)\[/HSP\]', r'\1', modified_paragraph)
            
            modified_paragraphs.append(modified_paragraph)
        else:
            # Keep original paragraph
            modified_paragraphs.append(paragraph)
    
    return modified_paragraphs, entity_data

def create_conll_output(original_conll: List[List[str]], modified_paragraphs: List[str], 
                       entity_data: Dict[str, List[Dict[str, Any]]]) -> List[List[str]]:
    """
    Create CoNLL formatted output with injected entities tagged.
    
    Args:
        original_conll: Original CoNLL paragraphs
        modified_paragraphs: Modified text paragraphs
        entity_data: Entity data
        
    Returns:
        Modified CoNLL formatted paragraphs
    """
    # Find paragraph with injected entities
    injected_paragraph_idx = -1
    for entity_type in entity_data:
        for entity in entity_data[entity_type]:
            injected_paragraph_idx = entity["paragraph_idx"]
            break
        if injected_paragraph_idx >= 0:
            break
    
    if injected_paragraph_idx < 0:
        logger.warning("No injected entities found")
        return original_conll
    
    # We need to create a new CoNLL format for the injected paragraph
    # This requires tokenization and alignment
    modified_conll = []
    
    # Copy all paragraphs before the injected one
    for i in range(injected_paragraph_idx):
        modified_conll.append(original_conll[i])
    
    # Process the injected paragraph
    injected_text = modified_paragraphs[injected_paragraph_idx]
    injected_tokens = injected_text.split()
    
    # Create CoNLL lines for the injected paragraph
    injected_conll = []
    
    # Extract all entities with their token ranges
    entity_ranges = []
    
    # Extract all medical entity texts for additional matching
    entity_texts = {}
    for entity_type, entities in entity_data.items():
        if entity_type not in entity_texts:
            entity_texts[entity_type] = []
        for entity in entities:
            if entity["paragraph_idx"] == injected_paragraph_idx:
                entity_texts[entity_type].append(entity["text"].lower())
    
    # Collect all token positions
    text_so_far = ""
    token_positions = []
    
    for i, token in enumerate(injected_tokens):
        # Add space before tokens except the first one
        if i > 0:
            text_so_far += " "
        
        token_start = len(text_so_far)
        text_so_far += token
        token_end = len(text_so_far)
        token_positions.append((token_start, token_end))
    
    # First, handle entities that were explicitly tagged in the original response
    for entity_type in entity_data:
        for entity in entity_data[entity_type]:
            if entity["paragraph_idx"] != injected_paragraph_idx:
                continue
            
            entity_text = entity["text"]
            entity_start_idx = injected_text.find(entity_text)
            
            if entity_start_idx == -1:
                logger.warning(f"Could not find entity '{entity_text}' in text. This might be a regex extraction issue.")
                continue
                
            entity_end_idx = entity_start_idx + len(entity_text)
            
            # Find all tokens that overlap with this entity
            entity_tokens = []
            for i, (token_start, token_end) in enumerate(token_positions):
                # Check if token overlaps with entity
                if token_end > entity_start_idx and token_start < entity_end_idx:
                    entity_tokens.append(i)
            
            # Tag the first token as B- and the rest as I-
            if entity_tokens:
                entity_ranges.append((entity_tokens[0], f"B-{entity_type}"))
                for token_idx in entity_tokens[1:]:
                    entity_ranges.append((token_idx, f"I-{entity_type}"))
    
    # Now detect additional occurrences of the same entities
    for i, token in enumerate(injected_tokens):
        # Skip tokens that already have a tag
        if any(i == token_idx for token_idx, _ in entity_ranges):
            continue
        
        # Check if this token matches any entity
        token_lower = token.lower().strip('.,;:?!"\'()')
        
        for entity_type, texts in entity_texts.items():
            for text in texts:
                # Handle multi-token entities
                if token_lower == text or token_lower in text.split():
                    # If it's a complete match or the first token
                    if token_lower == text or token_lower == text.split()[0]:
                        entity_ranges.append((i, f"B-{entity_type}"))
                    # If it's a partial match and looks like continuation
                    else:
                        # Check if previous token is already tagged as part of this entity
                        if i > 0 and any(i-1 == token_idx and tag.endswith(entity_type) for token_idx, tag in entity_ranges):
                            entity_ranges.append((i, f"I-{entity_type}"))
    
    # Now create CoNLL lines with the proper tagging
    for i, token in enumerate(injected_tokens):
        # Default tag is O (outside)
        tag = "O"
        
        # Check if this token has a tag
        for token_idx, entity_tag in entity_ranges:
            if i == token_idx:
                tag = entity_tag
                break
        
        # Create CoNLL line (simple version for now)
        injected_conll.append(f"{token} {tag}")
    
    modified_conll.append(injected_conll)
    
    # Copy all paragraphs after the injected one
    for i in range(injected_paragraph_idx + 1, len(original_conll)):
        modified_conll.append(original_conll[i])
    
    return modified_conll

def write_conll_file(conll_data: List[List[str]], output_path: str):
    """
    Write CoNLL data to file.
    
    Args:
        conll_data: CoNLL formatted paragraphs
        output_path: Path to write the file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for i, paragraph in enumerate(conll_data):
            for line in paragraph:
                f.write(f"{line}\n")
            
            # Add blank line between paragraphs (except after the last one)
            if i < len(conll_data) - 1:
                f.write("\n")

def get_dataset_files(dataset_dir: str = "dataset_txt/dataset_txt/training") -> List[str]:
    """
    Get list of CoNLL formatted files from the dataset directory.
    
    Args:
        dataset_dir: Directory containing CoNLL formatted files
        
    Returns:
        List of file paths
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory {dataset_dir} not found")
        
    files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) 
             if f.endswith('.txt')]
    
    logger.info(f"Found {len(files)} CoNLL formatted files in {dataset_dir}")
    return files

def main():
    """Main function to process CoNLL formatted files."""
    parser = argparse.ArgumentParser(description="Process CoNLL formatted files")
    parser.add_argument("--dataset_dir", type=str, 
                       default="dataset_txt/dataset_txt/training",
                       help="Directory containing CoNLL formatted files")
    parser.add_argument("--num_samples", type=int, default=1,
                       help="Number of samples to process")
    parser.add_argument("--model", type=str, default="mistral",
                       help="Ollama model to use")
    parser.add_argument("--output_dir", type=str, default="examples",
                       help="Output directory for processed files")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize LLM client
    llm_client = OllamaClient(model=args.model)
    
    # Get list of dataset files
    try:
        dataset_files = get_dataset_files(args.dataset_dir)
    except FileNotFoundError as e:
        logger.error(e)
        return
    
    # Process specified number of files
    for i, file_path in enumerate(dataset_files[:args.num_samples]):
        logger.info(f"Processing file {i+1}/{args.num_samples}: {file_path}")
        try:
            # Read CoNLL file
            conll_paragraphs = read_conll_file(file_path)
            
            # Reconstruct text paragraphs
            text_paragraphs = reconstruct_paragraphs(conll_paragraphs)
            
            # Inject medical entities
            modified_paragraphs, entity_data = inject_medical_entities(text_paragraphs, llm_client)
            
            # Create CoNLL output with entity tags
            modified_conll = create_conll_output(conll_paragraphs, modified_paragraphs, entity_data)
            
            # Save original and modified text
            basename = os.path.basename(file_path)
            with open(os.path.join(args.output_dir, f"{basename}_original.txt"), "w", encoding="utf-8") as f:
                f.write("\n\n".join(text_paragraphs))
            
            with open(os.path.join(args.output_dir, f"{basename}_modified.txt"), "w", encoding="utf-8") as f:
                f.write("\n\n".join(modified_paragraphs))
            
            # Save entity data
            with open(os.path.join(args.output_dir, f"{basename}_entities.json"), "w", encoding="utf-8") as f:
                json.dump(entity_data, f, indent=2)
            
            # Save CoNLL output
            write_conll_file(modified_conll, os.path.join(args.output_dir, f"{basename}_conll.txt"))
            
            # Preview modifications
            print("\nOriginal paragraph:")
            print("-" * 50)
            # Find modified paragraph
            for idx, (orig, mod) in enumerate(zip(text_paragraphs, modified_paragraphs)):
                if orig != mod:
                    print(orig)
                    print("\nModified paragraph:")
                    print("-" * 50)
                    print(mod)
                    print("-" * 50)
                    
                    # Also print entity data
                    print("\nExtracted entities:")
                    for entity_type in entity_data:
                        for entity in entity_data[entity_type]:
                            if entity["paragraph_idx"] == idx:
                                print(f"{entity_type}: {entity['text']}")
                    print("-" * 50)
                    break
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue

if __name__ == "__main__":
    main()