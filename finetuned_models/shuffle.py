import json
import argparse
import random

def shuffle_jsonl(input_path, output_path, seed=42):
    # Read all lines from the input file
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]

    # Shuffle with the specified seed
    random.seed(seed)
    random.shuffle(data)

    # Write the shuffled data to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shuffle a JSONL file with a fixed seed.")
    parser.add_argument("input", help="Path to the input JSONL file")
    parser.add_argument("output", help="Path to the output (shuffled) JSONL file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    shuffle_jsonl(args.input, args.output, args.seed)
