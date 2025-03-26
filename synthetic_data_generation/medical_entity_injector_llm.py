#!/usr/bin/env python3
"""
LLM-based Medical Entity Injector

This script uses a Large Language Model (LLM) to inject medical entities
into text in a more natural way for PII detection training.
It works with Ollama/Mistral locally but can be adapted for other LLMs.
"""

import os
import re
import random
import logging
import json
import argparse
import spacy
import requests
from typing import Dict, List, Tuple, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to load spaCy model
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

class LLMClient:
    """Base class for LLM clients that can be extended for different providers."""
    
    def generate(self, prompt: str) -> str:
        """Generate text based on prompt."""
        raise NotImplementedError("Subclasses must implement this method")

class OllamaClient(LLMClient):
    """Client for local Ollama API."""
    
    def __init__(self, model_name: str = "mistral", api_url: str = "http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.api_url = api_url
    
    def generate(self, prompt: str) -> str:
        """Generate text using local Ollama instance."""
        try:
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(self.api_url, json=data)
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"Error from Ollama API: {response.text}")
                return ""
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return ""

class DummyClient(LLMClient):
    """Dummy client for testing that returns template-based responses."""
    
    def generate(self, prompt: str) -> str:
        """Generate a simple response based on templates."""
        if "medical condition" in prompt:
            condition = random.choice(MEDICAL_CONDITIONS)
            return f"At age 34, the person was diagnosed with {condition} at {random.choice(HOSPITALS)}."
        elif "additional medical details" in prompt:
            medication = random.choice(MEDICATIONS)
            return f"They have been taking {medication} to manage their condition since 2018."
        else:
            return "No relevant medical information to add."

def create_injection_prompt(text: str, condition: Optional[str] = None) -> str:
    """
    Create a prompt for the LLM to inject medical information.
    
    Args:
        text: The original text to inject information into
        condition: Optional specific condition to include
        
    Returns:
        str: Formatted prompt for the LLM
    """
    # Extract biographical info for context
    doc = nlp(text[:500])  # Process just the beginning for efficiency
    people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    person = people[0] if people else "the person"
    
    # Create a condition-specific prompt if provided
    condition_text = ""
    if condition:
        condition_text = f" specifically mentioning {condition}"
    
    prompt = f"""
I have a biographical text about {person}. I need you to create ONE SENTENCE that contains medical information{condition_text} that could be inserted into this text naturally. The sentence should feel like it belongs in a biography.

The medical information should include AT LEAST one of:
1. A medical condition
2. A medication 
3. A hospital or medical facility

Original text beginning:
"{text[:300]}..."

Please provide ONLY the new sentence to insert, with no additional explanation or commentary. Make it sound natural and factual.
"""
    return prompt

def create_medical_info_json_prompt(text: str, entity_type: Optional[str] = None) -> str:
    """
    Create a prompt asking for a JSON response with medical entity information.
    
    Args:
        text: Text to analyze for context
        entity_type: Optional specific entity type to focus on
        
    Returns:
        str: JSON prompt for the LLM
    """
    # Process beginning of text for context
    doc = nlp(text[:500])
    people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    person = people[0] if people else "the subject"
    
    focus = ""
    if entity_type:
        if entity_type == "MED":
            focus = " Focus on medical conditions."
        elif entity_type == "DRUG":
            focus = " Focus on medications/treatments."
        elif entity_type == "HSP":
            focus = " Focus on hospitals/medical facilities."
    
    prompt = f"""
Based on the biographical text about {person}, create a JSON object containing synthetic medical information that could be inserted into the biography.{focus}

Original text beginning:
"{text[:300]}..."

Return ONLY a valid JSON object with this structure:
{{
  "sentence": "The sentence to insert with medical information",
  "entities": [
    {{
      "text": "entity text",
      "type": "MED or DRUG or HSP", 
      "start_idx": 10, 
      "end_idx": 20
    }}
  ]
}}

The types should be:
- "MED" for medical conditions
- "DRUG" for medications
- "HSP" for hospitals/medical facilities

Ensure the sentence is natural, factual-sounding, and relevant to the biography. Position indices should be correct within your sentence.
"""
    return prompt

def parse_llm_json_response(response: str) -> Dict:
    """
    Parse JSON from LLM response, handling common formatting issues.
    
    Args:
        response: Raw response from LLM
        
    Returns:
        Dict: Parsed JSON or empty dict if parsing fails
    """
    try:
        # Extract JSON if wrapped in other text
        json_pattern = r'(\{[\s\S]*\})'
        json_match = re.search(json_pattern, response)
        
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
        else:
            return json.loads(response)
    
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.debug(f"Response was: {response}")
        return {}

def inject_medical_entity_with_llm(text: str, llm_client: LLMClient) -> Tuple[str, Dict]:
    """
    Use LLM to inject medical entity information into text.
    
    Args:
        text: Original text
        llm_client: LLM client instance
        
    Returns:
        Tuple of (modified_text, entities)
    """
    # Choose a random entity type to focus on
    entity_type = random.choice(["MED", "DRUG", "HSP"])
    
    # Create prompt for JSON response
    prompt = create_medical_info_json_prompt(text, entity_type)
    
    # Get response from LLM
    response = llm_client.generate(prompt)
    
    # Parse the response
    result = parse_llm_json_response(response)
    
    if not result or "sentence" not in result or "entities" not in result:
        logger.warning("LLM did not return properly formatted data, using fallback approach")
        # Fallback to simpler approach
        simple_prompt = create_injection_prompt(text)
        injected_sentence = llm_client.generate(simple_prompt)
        
        # Find injection point (after a sentence)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) < 2:
            return text, {}
            
        injection_point = random.randint(1, min(len(sentences) - 1, 5))
        sentences.insert(injection_point, injected_sentence)
        modified_text = " ".join(sentences)
        
        # Simple entity extraction for fallback
        entities = {}
        for condition in MEDICAL_CONDITIONS:
            if condition.lower() in injected_sentence.lower():
                if "MED" not in entities:
                    entities["MED"] = []
                entities["MED"].append(condition)
                
        for medication in MEDICATIONS:
            if medication.lower() in injected_sentence.lower():
                if "DRUG" not in entities:
                    entities["DRUG"] = []
                entities["DRUG"].append(medication)
                
        for hospital in HOSPITALS:
            if hospital.lower() in injected_sentence.lower():
                if "HSP" not in entities:
                    entities["HSP"] = []
                entities["HSP"].append(hospital)
        
        return modified_text, {
            "MED": entities.get("MED", []),
            "DRUG": entities.get("DRUG", []),
            "HSP": entities.get("HSP", []),
            "injected_sentence": injected_sentence,
            "injection_point": injection_point
        }
    
    # Extract information from result
    injected_sentence = result["sentence"]
    entity_info = result["entities"]
    
    # Find a good injection point
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) < 2:
        return text, {}
        
    injection_point = random.randint(1, min(len(sentences) - 1, 5))
    sentences.insert(injection_point, injected_sentence)
    modified_text = " ".join(sentences)
    
    # Format entities for return
    entities = {"MED": [], "DRUG": [], "HSP": []}
    
    for entity in entity_info:
        entity_type = entity.get("type")
        entity_text = entity.get("text")
        
        if entity_type in entities and entity_text:
            entities[entity_type].append(entity_text)
    
    entities["injected_sentence"] = injected_sentence
    entities["injection_point"] = injection_point
    
    return modified_text, entities

def create_conll_format(text: str, entities: Dict) -> List[str]:
    """
    Convert text with injected entities to CoNLL format with BIO tagging.
    
    Args:
        text: The modified text
        entities: Dictionary of entities by type
        
    Returns:
        List of strings in CoNLL format
    """
    # Process with spaCy
    doc = nlp(text)
    
    # Prepare CoNLL format data
    conll_data = []
    
    # List to track entities to tag
    entities_to_tag = []
    
    # Add each entity with its position in the text
    for entity_type, entity_list in entities.items():
        if entity_type in ["MED", "DRUG", "HSP"]:
            for entity in entity_list:
                # Find all occurrences
                for match in re.finditer(re.escape(entity.lower()), text.lower()):
                    start, end = match.span()
                    entities_to_tag.append((entity_type, start, end))
                    logger.debug(f"Found {entity_type} at positions {start}-{end}: {text[start:end]}")
    
    # Sort entities by start position
    entities_to_tag.sort(key=lambda x: x[1])
    
    # Process each sentence
    for sent in doc.sents:
        sent_tokens = [token.text for token in sent]
        sent_tags = ["O"] * len(sent_tokens)
        
        # Tag entities in this sentence
        sent_start = sent[0].idx
        sent_end = sent[-1].idx + len(sent[-1].text)
        
        # Check which entities are in this sentence
        for ent_type, ent_start, ent_end in entities_to_tag:
            if ent_start >= sent_start and ent_end <= sent_end:
                # Entity is in this sentence, find which tokens it corresponds to
                for i, token in enumerate(sent):
                    token_start = token.idx
                    token_end = token_start + len(token.text)
                    
                    # Check if token overlaps with entity
                    if token_start < ent_end and token_end > ent_start:
                        # First token of entity gets B- tag, rest get I- tag
                        if token_start <= ent_start < token_end:
                            sent_tags[i] = f"B-{ent_type}"
                        else:
                            sent_tags[i] = f"I-{ent_type}"
        
        # Add tokens and tags to CoNLL data
        for token, tag in zip(sent_tokens, sent_tags):
            conll_data.append(f"{token} {tag}")
        
        # Add empty line after sentence
        conll_data.append("")
    
    return conll_data

def save_to_conll_file(conll_data: List[str], output_file: str) -> None:
    """Save tokens and tags to a file in CoNLL format."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(conll_data))

def sample_text_for_processing() -> List[str]:
    """
    Sample some example texts for processing.
    In a real scenario, you would load from Wikipedia or another source.
    """
    # Kendrick Lamar example
    texts = [
        """Kendrick Lamar Duckworth was born on June 17, 1987, in Compton, California. He is the first child of former gang hustler Kenneth "Kenny" Duckworth and hairdresser Paula Oliver. Both of his parents are African Americans from the South Side of Chicago. When they were teenagers, they relocated to Compton in 1984, due to his father's affiliation with the Gangster Disciples. Lamar was named after singer-songwriter Eddie Kendricks of the Temptations. He was an only child until the age of seven and was described as a loner by his mother.""",
        
        """Albert Einstein was born in Ulm, in the Kingdom of Württemberg in the German Empire, on 14 March 1879. His parents were Hermann Einstein, a salesman and engineer, and Pauline Koch. In 1880, the family moved to Munich, where Einstein's father and his uncle Jakob founded Elektrotechnische Fabrik J. Einstein & Cie, a company that manufactured electrical equipment based on direct current.""",
        
        """Marie Skłodowska Curie was born in Warsaw, in what was then the Kingdom of Poland, part of the Russian Empire. She was the youngest of five children of well-known teachers Bronisława and Władysław Skłodowski. Marie's early years were marked by the death of her sister from typhus and, two years later, the death of her mother from tuberculosis when Marie was only 10. Marie was a top student in her secondary school.""",
        
        """Elon Reeve Musk was born on June 28, 1971, in Pretoria, one of South Africa's capital cities. His father is Errol Musk, a South African electromechanical engineer, pilot, sailor, consultant, and property developer who was once a half-owner of a Zambian emerald mine near Lake Tanganyika. His mother is Maye Musk, a model and dietitian born in Saskatchewan, Canada, and raised in South Africa."""
    ]
    
    return texts

def main():
    """Main function to demonstrate LLM-based synthetic medical entity injection."""
    parser = argparse.ArgumentParser(description="LLM-based Medical Entity Injector")
    parser.add_argument("--model", type=str, default="mistral", help="Ollama model to use")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples to generate")
    parser.add_argument("--use_dummy", action="store_true", help="Use dummy LLM client for testing")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs("examples", exist_ok=True)
    
    # Initialize LLM client
    if args.use_dummy:
        logger.info("Using dummy LLM client for testing")
        llm_client = DummyClient()
    else:
        logger.info(f"Using Ollama with model {args.model}")
        llm_client = OllamaClient(model_name=args.model)
    
    # Load sample texts
    sample_texts = sample_text_for_processing()
    if args.num_samples < len(sample_texts):
        sample_texts = sample_texts[:args.num_samples]
    
    logger.info(f"Processing {len(sample_texts)} text samples")
    
    # Process each text
    all_conll_data = []
    tagged_entities_count = 0
    
    for i, text in enumerate(sample_texts):
        logger.info(f"Processing text {i+1}/{len(sample_texts)}")
        
        # Inject medical entity using LLM
        try:
            modified_text, entities = inject_medical_entity_with_llm(text, llm_client)
            
            if "injected_sentence" in entities:
                # Convert to CoNLL format
                conll_data = create_conll_format(modified_text, entities)
                
                # Save individual samples
                output_file = f"examples/llm_medical_sample_{i+1}.txt"
                save_to_conll_file(conll_data, output_file)
                
                # Save the original and modified text for reference
                with open(f"examples/llm_medical_sample_{i+1}_original.txt", 'w', encoding='utf-8') as f:
                    f.write(text)
                
                with open(f"examples/llm_medical_sample_{i+1}_modified.txt", 'w', encoding='utf-8') as f:
                    f.write(modified_text)
                
                # Add to combined dataset
                all_conll_data.extend(conll_data)
                
                # Count tagged entities
                for line in conll_data:
                    if "B-" in line:
                        tagged_entities_count += 1
                
                # Print example of the first sample
                if i == 0:
                    injected = entities.get("injected_sentence", "No injection")
                    logger.info(f"Example injection: {injected}")
                    
                    # Show entity summary
                    entity_summary = ", ".join([f"{k}: {v}" for k, v in entities.items() 
                                              if k in ["MED", "DRUG", "HSP"] and v])
                    logger.info(f"Entities injected: {entity_summary}")
                    
                    # Show a sample of the CoNLL format
                    entity_lines = [line for line in conll_data if "B-" in line or "I-" in line]
                    if entity_lines:
                        sample_str = "\n".join(entity_lines[:10])
                        logger.info(f"CoNLL format sample (tagged entities):\n{sample_str}")
            else:
                logger.warning(f"No medical entities were injected in sample {i+1}")
        
        except Exception as e:
            logger.error(f"Error processing sample {i+1}: {e}")
    
    # Save combined dataset
    if all_conll_data:
        combined_file = "examples/llm_medical_combined_dataset.txt"
        save_to_conll_file(all_conll_data, combined_file)
        logger.info(f"Created {len(sample_texts)} samples in the examples/ directory")
        logger.info(f"Tagged {tagged_entities_count} entities in total")
        logger.info(f"Combined dataset saved to {combined_file}")
    else:
        logger.warning("No data was generated")
    
    logger.info("Done! The generated dataset can now be used for training a NER model for PII detection.")

if __name__ == "__main__":
    main() 