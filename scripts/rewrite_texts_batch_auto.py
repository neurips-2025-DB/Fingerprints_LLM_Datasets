# An OpenAI account is required, and the API key should be set as an environment variable. 
# export OPENAI_API_KEY=


import time
import json
import argparse
import logging
from openai import OpenAI

# Predefined prompts
PROMPTS = {
    "prompt1": "Rewrite the following text sentence by sentence while preserving its length and the accuracy of its content. Maintain the overall format, structure, and flow of the text: \n ",
    "prompt2": "Rewrite the following text while preserving its length and the accuracy of its content: \n ",
    "prompt3": "Rewrite the following text while preserving its length and the accuracy of its content. Do not use newlines, new paragraphs, itemization, enumeration, and other formatting, unless it is important or appropriate for better readability: \n "
}


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to submit a new batch
def submit_batch(client, input_data, prompt):
    # Generate the batch file name
    batch_file_name = "batch_input.jsonl"

    # Write the batch JSONL file
    with open(batch_file_name, 'w') as file:
        for i, entry in enumerate(input_data):
           
            json_data = json.dumps({'custom_id': f"request-{i}", 
                                    'method': 'POST', 
                                    'url': '/v1/chat/completions', 
                                    'body': {'model': 'gpt-4o-mini', 
                                             'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'}, 
                                                          {'role': 'user', 'content': prompt + entry["text"]}],
                                             'max_tokens': 2048}})
            file.write(json_data + '\n')

    # Upload batch input file
    batch_input_file = client.files.create(
        file=open(batch_file_name, "rb"),
        purpose="batch"
    )
    
    # Create the batch
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "Rewrite texts"
        }
    )

    logging.info(f"Submitted batch request with id {batch.id}")
    return batch.id

# Function to retrieve a completed batch
def retrieve_batch(client, batch_id, output_file):
    # Check batch status and wait for completion
    while True:
        batch = client.batches.retrieve(batch_id)
        logging.info(f"Batch {batch_id} status: {batch.status}")
        
        if batch.status == "completed":
            logging.info(f"Batch {batch_id} completed. Retrieving results...")
            break
        else:
            time.sleep(60)  # Wait for 60 seconds before checking again

    # Retrieve and save results
    file_response = client.files.content(batch.output_file_id)
    with open(output_file, 'a') as file:  # Append to the output file
        for line in file_response.text.split("\n"):
            if line.strip():  # Skip empty lines
                data = json.loads(line)
                file.write(json.dumps({'text': data['response']['body']['choices'][0]['message']['content']}) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Automated rewriting script.')
    
    # Add arguments
    parser.add_argument('--input-file', type=str, required=True, help='Path to the jsonl input file, containing the texts to rewrite')
    parser.add_argument('--output-file', type=str, required=True, help='Path to the jsonl output file for writing the rewritten texts to')
    parser.add_argument('--batch-size', type=int, default=2000, help='Number of lines to process in each batch')
    parser.add_argument('--prompt', type=str, choices=['prompt1', 'prompt2', 'prompt3'], required=True, help='Select the rewriting prompt: prompt1, prompt2, or prompt3')

    args = parser.parse_args()
    logging.info(f"Arguments received: {args}")

    # Select the appropriate prompt
    prompt = PROMPTS[args.prompt]

    client = OpenAI()

    # Load the input data
    with open(args.input_file, "r") as f:
        data = [json.loads(line) for line in f]

    # Automate batch processing
    for i in range(0, len(data), args.batch_size):
        batch_data = data[i:i + args.batch_size]
        logging.info(f"Submitting new batch with {len(batch_data)} entries...")
        batch_id = submit_batch(client, batch_data)

        logging.info(f"Waiting for batch {batch_id} to complete...")
        retrieve_batch(client, batch_id, args.output_file)

        logging.info(f"Batch {batch_id} completed and results saved.")

if __name__ == "__main__":
    main()
