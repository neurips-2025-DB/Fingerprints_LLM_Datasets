from datasets import get_dataset_config_names, load_dataset
import json
from datetime import datetime
import os

# Base output directory
output_dir = "paloma"
os.makedirs(output_dir, exist_ok=True)

# Get all configs
configs = get_dataset_config_names("allenai/paloma")
print("Configs found:", configs)

# Helper function to convert datetime recursively
def serialize(obj):
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize(v) for v in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj

# Write each config to a separate JSONL file
for config in configs:
    print(f"Processing config: {config}")
    dataset = load_dataset("allenai/paloma", config, split="test")
    
    output_file = os.path.join(output_dir, f"{config}.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for example in dataset:
            example["config"] = config
            serialized = serialize(example)
            json.dump(serialized, f)
            f.write("\n")

    print(f"✅ Saved: {output_file}")

print("\n🎉 All test sets saved individually in:", output_dir)


import os
import random

def trim_jsonl_files(directory, num_lines=1000, seed=7):
    # Set the seed for reproducibility
    random.seed(seed)

    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(directory, filename)

            # Read all lines from the file
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # If file has fewer than num_lines lines, keep all
            if len(lines) <= num_lines:
                print(f"{filename}: Only {len(lines)} lines, skipping trimming.")
                continue

            # Randomly select num_lines lines
            selected_lines = random.sample(lines, num_lines)

            # Write back the selected lines
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(selected_lines)

            print(f"{filename}: Trimmed to {num_lines} lines.")

# Example usage with a seed
trim_jsonl_files('paloma')
