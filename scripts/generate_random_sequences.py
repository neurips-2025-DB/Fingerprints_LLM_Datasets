# Section 5 in the paper
# --hf-model "tiiuae/falcon-7b" # RefinedWeb (pretrained)
# --hf-model "apple/DCLM-Baseline-7B"  # DCLM-Baseline (pretrained)
# --hf-model "HuggingFaceFW/ablation-model-fineweb-edu" #FineWeb-Edu (pretrained)


# Section 5.1 in the paper
# --hf-model "MBZUAI-LLM/SlimPajama-DC" # SlimPajama (pretrained)
# When using this model, modify the code below to inlcude "revision" when loading the model and the tokenizer as such:
# tokenizer = AutoTokenizer.from_pretrained(hf_model, revision="SlimPajama-DC-6")
# model = AutoModelForCausalLM.from_pretrained(hf_model, revision="SlimPajama-DC-6")
# revision= "SlimPajama-DC-6" is for the LLM pretrained on all 7 domains
# revision= "SlimPajama-DC-5" is for the LLM pretrained on only 4 domains
# More info can be found here: https://huggingface.co/MBZUAI-LLM/SlimPajama-DC


# Appendix I in the paper
# --hf-model "tiiuae/falcon-7b-instruct" # RefinedWeb (finetuned)
# --hf-model "mlfoundations/dclm-7b-it" # DCLM (finetuned)


import json
import random
import torch
import time
from open_lm.hf import *
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Generates N random start tokens from a given file
def generate_start_tokens(tokenizer, N, file_path='input_file.jsonl'):
    # Read texts from file
    texts = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            texts.append(data['text'][:20])

    logging.info(f"Loaded {len(texts)} sequences from {file_path}")

    # Tokenize texts
    start_tokens = []
    for text in texts:
        input = tokenizer([text], return_tensors="pt")
        input = {k: v[:, :1] for k, v in input.items()}
        start_tokens.append(input)
    
    start_tokens = random.sample(start_tokens, N)
    return start_tokens

# Generates N random tokens sampled from the tokenizer's vocabulary
def generate_random_tokens(tokenizer, N):
    vocab_size = tokenizer.vocab_size
    random_tokens = []

    for _ in range(N):
        # Randomly sample token IDs from the vocabulary range
        token_id = random.randint(0, vocab_size - 1)
        # Convert token ID to tensor and structure like input
        input = {
            "input_ids": torch.tensor([[token_id]]),
            "attention_mask": torch.tensor([[1]]),
        }
        random_tokens.append(input)

    return random_tokens

def generate_sequences(start_tokens, tokenizer, hf_model, batch_size=16, max_new_tokens=800, output_file='output.jsonl'):
    model = AutoModelForCausalLM.from_pretrained(hf_model)
    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Process inputs in batches
    for i in range(0, len(start_tokens), batch_size):
        batch = start_tokens[i:i + batch_size]
    
        # Concatenate tensors to create batch
        input_ids = torch.cat([item['input_ids'] for item in batch], dim=0).to(device)
        attention_mask = torch.cat([item['attention_mask'] for item in batch], dim=0).to(device)
    
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
    
        # Measure the time
        start = time.time()
        # Generate text for the batch
        eos_token_id = tokenizer.eos_token_id  # Use the EOS token from your tokenizer
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "top_p": 0.8,
            "temperature": 0.99,
            "do_sample": True,
            "repetition_penalty": 1.1,
            "eos_token_id": eos_token_id  # Explicitly set the EOS token ID
        }        
        output = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], **gen_kwargs)
        output = output.cpu()
        end = time.time()
    
        # Append output to jsonl file
        with open(output_file, 'a') as file:
            for generated_seq in output:
                decoded_output = tokenizer.decode(generated_seq.tolist(), skip_special_tokens=True)
                file.write(json.dumps({'text': decoded_output}) + '\n')

        # Time taken in seconds
        time_taken = end - start
        tokens_generated = sum([len(seq) for seq in output])
        # Log the time taken per batch
        logging.info(f"Tokens per second: {tokens_generated/time_taken}")

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Generation script.')
    
    # Add arguments
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for generation')
    parser.add_argument('--num-seqs', type=int, default=128, help='Number of sequences to generate')
    parser.add_argument('--max-new-tokens', type=int, default=800, help='Maximal number of tokens to generate')
    parser.add_argument('--input-file', type=str, help='Path to the jsonl input file, for determining the distribution of the starting token')
    parser.add_argument('--output-file', type=str, required=True, help='Path to the jsonl output file, for writing the seqs to')
    parser.add_argument('--hf-model', type=str, default="apple/DCLM-Baseline-7B", help='HuggingFace model to use for generation')

    # Parse the arguments
    args = parser.parse_args()
    logging.info(f"Arguments received: {args}")
        
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    
    # Generate start tokens based on input file availability
    if args.input_file:
        logging.info("Generating start tokens from input file...")
        start_tokens = generate_start_tokens(tokenizer, args.num_seqs, file_path=args.input_file)
    else:
        logging.info("Generating random start tokens from tokenizer's vocabulary...")
        start_tokens = generate_random_tokens(tokenizer, args.num_seqs)

    # Generate sequences
    generate_sequences(start_tokens, tokenizer, args.hf_model, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens, output_file=args.output_file)

# Entry point of the script
if __name__ == "__main__":
    main()



