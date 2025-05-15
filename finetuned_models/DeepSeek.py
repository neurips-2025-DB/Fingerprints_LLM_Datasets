#python DeepSeek.py --input-file openhermes2_5.jsonl --output-file DeepSeek_responses.jsonl --end-index 15000

#Input your API key here
API_KEY = "API_KEY_COMES_HERE"

from openai import OpenAI
import json
import argparse
import logging
import time
import backoff
from openai import RateLimitError
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to retry the API call with exponential backoff
@backoff.on_exception(backoff.expo, RateLimitError, max_tries=5)
def completions_with_backoff(client, **kwargs):
    response = client.chat.completions.create(**kwargs)
    return response    

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Rewriting script.')
    
    # Add arguments
    parser.add_argument('--input-file', type=str, required=True, help='Path to the jsonl input file, containt the texts to rewrite')
    parser.add_argument('--output-file', type=str, required=True, help='Path to the jsonl output file, for writing the rewritten texts to')
    parser.add_argument('--end-index', type=int, required=True, help='End index of the data to process')
    
    # Parse the arguments
    args = parser.parse_args()
    logging.info(f"Arguments received: {args}")
        
    
    client = OpenAI(
    base_url= "https://api.together.xyz/v1/",
    api_key= API_KEY
    )
    
    # Determine the start index based on the number of lines in the output file
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            start_index = len(f.readlines())  # Number of lines in the output file
    else:
        start_index = 0  # If the file doesn't exist, start from 0

    # Load the input data
    with open(args.input_file, "r") as f:
        data = [json.loads(line) for line in f]


    for i, entry in enumerate(data[start_index:args.end_index]):
    
        # measure the time
        start = time.time()

        prompt = next((conv["value"] for conv in entry["conversations"] if conv["from"] == "human"), None)
        
        messages = [{"role": "system", "content": "You are a helpful assistant."}, 
                    {"role": "user", "content": prompt}
        ]
        
        try:
            completion = completions_with_backoff(client, model="deepseek-ai/DeepSeek-V3", messages=messages, max_tokens=2048)
            

            # append output to jsonl file
            with open(args.output_file, 'a') as file:
                file.write(json.dumps({'text': completion.choices[0].message.content}) + '\n')
        
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            continue

        end = time.time()
        # log the time and length of the generated text
        time_taken = end - start
        characters_generated = len(completion.choices[0].message.content)
        logging.info(f"Characters per second: {characters_generated/time_taken}")



# Entry point of the script
if __name__ == "__main__":
    main()
