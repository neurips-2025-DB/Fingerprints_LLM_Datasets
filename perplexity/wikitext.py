import os
import json
import random
from datasets import load_dataset

# Set random seed for reproducibility
random.seed(7)

# Create output directory
output_dir = "wikitext_103"
os.makedirs(output_dir, exist_ok=True)

# Output file path
output_file = os.path.join(output_dir, "wikitext-103.jsonl")

# Load dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

articles = []
current_article = []
prev_line_was_empty = True  # Flag to track empty lines

# Function to detect the start of a new article
def is_article_start(line, prev_line_empty):
    line = line.strip()
    return (
        prev_line_empty and
        line.startswith("= ") and line.endswith(" =") and 
        line.count("=") == 2
    )

# Process dataset to extract articles
for entry in dataset:
    line = entry["text"]
    stripped_line = line.strip()
    
    if is_article_start(stripped_line, prev_line_was_empty):
        if current_article:
            articles.append("\n".join(current_article))
            current_article = []

    if stripped_line:
        current_article.append(stripped_line)
        prev_line_was_empty = False
    else:
        prev_line_was_empty = True

# Add last article if any
if current_article:
    articles.append("\n".join(current_article))

print(f"Processed {len(articles)} articles.")

# Ensure we have at least 1000 articles
if len(articles) < 1000:
    raise ValueError("Fewer than 1000 articles found.")

# Randomly sample 1000 articles
sampled_articles = random.sample(articles, 1000)

# Save to JSONL
with open(output_file, "w", encoding="utf-8") as f:
    for article in sampled_articles:
        json.dump({"text": article}, f, ensure_ascii=False)
        f.write("\n")

print(f"Saved 1000 sampled articles to {output_file}")
