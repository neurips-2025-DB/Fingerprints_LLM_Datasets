#python GPT.py --input-file openhermes2_5.jsonl --output-file GPT_responses.jsonl --end-index 15000

#Export the OpenAI API key as an environment variable before running this script
#export OPENAI_API_KEY= "API_KEY_COMES_HERE"

import time
import json
import argparse
import logging
import os
from openai import OpenAI


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to submit a new batch
def submit_batch(client, input_data):
    # Generate the batch file name
    batch_file_name = "gpt4o-batch_input.jsonl"

    # Write the batch JSONL file
    with open(batch_file_name, 'w') as file:
        for i, entry in enumerate(input_data):
            
            prompt = next((conv["value"] for conv in entry["conversations"] if conv["from"] == "human"), None)
            #print(prompt)
            json_data = json.dumps({'custom_id': f"request-{i}", 
                                    'method': 'POST', 
                                    'url': '/v1/chat/completions', 
                                    'body': {'model': 'gpt-4o', 
                                             'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'}, 
                                                          {'role': 'user', 'content': prompt}],
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
            "description": "Respond to prompts from OpenHermes 2.5"
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
    parser = argparse.ArgumentParser(description='Automated prompt following script.')
    
    # Add arguments
    parser.add_argument('--input-file', type=str, required=True, help='Path to the jsonl input file, containing the prompts')
    parser.add_argument('--output-file', type=str, required=True, help='Path to the jsonl output file for saving the response')
    parser.add_argument('--batch-size', type=int, default=2000, help='Number of lines to process in each batch')
    parser.add_argument('--end-index', type=int, required=True, help='End index of the data to process')
    
    args = parser.parse_args()
    logging.info(f"Arguments received: {args}")

    client = OpenAI()

    # Determine the start index based on the number of lines in the output file
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            start_index = len(f.readlines())  # Number of lines in output_file
    else:
        start_index = 0  # If the file doesn't exist, start from 0

    # Load the input data
    with open(args.input_file, "r") as f:
        data = [json.loads(line) for line in f]

    # Automate batch processing
    for i in range(start_index, args.end_index, args.batch_size):
        batch_data = data[i: i+args.batch_size]
        logging.info(f"Submitting new batch with {len(batch_data)} entries...")
        batch_id = submit_batch(client, batch_data)

        logging.info(f"Waiting for batch {batch_id} to complete...")
        retrieve_batch(client, batch_id, args.output_file)

        logging.info(f"Batch {batch_id} completed and results saved.")

if __name__ == "__main__":
    main()
