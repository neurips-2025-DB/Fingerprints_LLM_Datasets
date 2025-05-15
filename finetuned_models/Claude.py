#python Claude.py --input-file openhermes2_5.jsonl --output-file Claude_responses.jsonl --end-index 15000


#Input your API key here
API_KEY = "API_KEY_COMES_HERE"

import time
import json
import argparse
import logging
import os
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to submit a new batch
def submit_batch(client, input_data):
    batch_requests = []

    for i, entry in enumerate(input_data):
        prompt = next((conv["value"] for conv in entry["conversations"] if conv["from"] == "human"), None)
        if prompt:
            batch_requests.append(
                Request(
                    custom_id=f"request-{i}",
                    params=MessageCreateParamsNonStreaming(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=2048,
                        messages=[{"role": "user", "content": prompt}]
                    )
                )
            )

    # Submit batch request
    batch = client.messages.batches.create(requests=batch_requests)

    logging.info(f"Submitted batch request with id {batch.id}")
    return batch.id

# Function to retrieve and save batch results
def retrieve_batch(client, batch_id, output_file):
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        logging.info(f"Batch {batch_id} status: {batch.processing_status}")

        if batch.processing_status == "ended":
            logging.info(f"Batch {batch_id} completed. Retrieving results...")
            break
        else:
            time.sleep(60)  # Wait before checking again

    # Retrieve results and save them
    with open(output_file, 'a') as file:
        for result in client.messages.batches.results(batch_id):
            if result.result and result.result.type == 'succeeded':
                message_content = result.result.message.content[0].text
                file.write(json.dumps({'text': message_content}) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Automated prompt processing script.')

    # Add arguments
    parser.add_argument('--input-file', type=str, required=True, help='Path to the jsonl input file containing the prompts')
    parser.add_argument('--output-file', type=str, required=True, help='Path to the jsonl output file for saving the responses')
    parser.add_argument('--batch-size', type=int, default=100, help='Number of lines to process in each batch')
    parser.add_argument('--end-index', type=int, required=True, help='End index of the data to process')


    args = parser.parse_args()
    logging.info(f"Arguments received: {args}")

    client = anthropic.Anthropic(api_key= API_KEY)

    # Determine the start index based on the number of lines in the output file
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            start_index = len(f.readlines())  # Number of lines in the output file
    else:
        start_index = 0  # If the file doesn't exist, start from 0
    print("starting at ", start_index)    

    # Load the input data
    with open(args.input_file, "r") as f:
        data = [json.loads(line) for line in f]

    # Automate batch processing
    for i in range(start_index, args.end_index, args.batch_size):
        batch_data = data[i: i + args.batch_size]
        logging.info(f"Submitting new batch with {len(batch_data)} entries...")
        batch_id = submit_batch(client, batch_data)

        if batch_id:
            logging.info(f"Waiting for batch {batch_id} to complete...")
            retrieve_batch(client, batch_id, args.output_file)
            logging.info(f"Batch {batch_id} completed and results saved.")
        else:
            logging.error("Batch submission failed. Skipping this batch.")

if __name__ == "__main__":
    main()
