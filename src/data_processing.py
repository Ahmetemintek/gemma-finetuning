"""
Simple preprocessing script that converts raw dataset files into formatted JSONL.
Loads train/validation/test splits, applies formatting, and saves to data/processed/.
"""

import os
import json
import pandas as pd
from tqdm import tqdm

from formatting import format_example


# Hardcoded paths
RAW_DATA_DIR = "data/raw/MedTurkQuAD"
PROCESSED_DATA_DIR = "data/processed"
SPLITS = ["train", "validation", "test"]


def process_split(split_name):
    """Process a single split and save as JSONL."""
    # Load raw JSON file
    raw_file = os.path.join(RAW_DATA_DIR, f"{split_name}.json")
    df = pd.read_json(raw_file)
    
    # Prepare output
    output_file = os.path.join(PROCESSED_DATA_DIR, f"{split_name}.jsonl")
    processed_rows = []
    
    # Process each row
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
        context = row["context"]
        question = row["question"]
        answer = row["answers"]["text"][0]
        
        # Format using formatting.py
        formatted_text = format_example(context, question, answer)
        
        # Create output row
        processed_row = {
            "context": context,
            "question": question,
            "answer": answer,
            "formatted_text": formatted_text
        }
        processed_rows.append(processed_row)
    
    # Save as JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for row in processed_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    # Print summary
    print(f"✓ {split_name}: {len(processed_rows)} examples saved to {output_file}")


if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Process all splits
    for split in SPLITS:
        process_split(split)
    
    print("\nProcessing complete.")
