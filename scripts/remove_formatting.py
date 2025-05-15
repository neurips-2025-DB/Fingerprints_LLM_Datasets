import json
from tqdm import tqdm
import re
import argparse

def flatten_text(text):
    # Remove all newlines and merge paragraphs
    text = re.sub(r'\s*\n\s*', ' ', text)
    # Remove itemization and enumeration patterns (numbers, bullets, letters, etc.)
    text = re.sub(r'(^|\s)([\d•\-a-zA-Z\(\)]+\.\s)', ' ', text)
    # Normalize excessive spaces
    text = re.sub(r'\s{2,}', ' ', text)
    # Remove other special characters typically used for structuring (if needed)
    text = re.sub(r'[\t\r]', '', text)
    return text.strip()

def process_jsonl(input_file, output_file):
    # Count total lines for tqdm
    with open(input_file, 'r', encoding='utf-8') as infile:
        total_lines = sum(1 for _ in infile)

    # Process file with tqdm
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in tqdm(infile, total=total_lines, desc="Processing JSONL"):
            # Parse JSON object from line
            data = json.loads(line)
            # Apply flatten_text to the "text" field if it exists
            if "text" in data:
                flattened_text = flatten_text(data["text"])
                # Write the modified "text" field to the new file
                outfile.write(json.dumps({"text": flattened_text}) + '\n')

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process and flatten text fields in a JSONL file.")
    parser.add_argument("input_file", help="Path to the input JSONL file")
    parser.add_argument("output_file", help="Path to the output JSONL file")

    # Parse arguments
    args = parser.parse_args()

    # Run the processing function
    process_jsonl(args.input_file, args.output_file)
