##Takes as input one jsonl file, tokenizes it, and saves it as a list of pytorch tensors in one output file

import json
import torch
import argparse
from transformers import GPTNeoXTokenizerFast

def main(input_file, output_file):
    # Initialize the tokenizer
    tokenizer = GPTNeoXTokenizerFast.from_pretrained('EleutherAI/gpt-neox-20b')

    sequences = []

    # Load data from the input JSONL file
    with open(input_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            data = json.loads(line)
            sequences.append(data['text'])  # Assuming the sequences are stored under the key 'text'

    # Tokenize each sequence and convert to tensors
    tensors = []

    for sequence in sequences:
        tokenized = tokenizer(sequence, return_tensors='pt', truncation=False, padding=False)
        tensor = tokenized['input_ids']
        tensors.append(tensor)

    # Save the tensors to the specified output file
    torch.save(tensors, output_file)
    print(f"Tensors saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize a JSONL file and save it as a tensor.")
    parser.add_argument('--input-file', type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument('--output-file', type=str, required=True, help="Path to save the output tensor file.")
    args = parser.parse_args()

    main(args.input_file, args.output_file)
