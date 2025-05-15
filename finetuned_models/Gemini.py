#python Gemini.py --input-file openhermes2_5.jsonl --output-file Gemini_responses.jsonl --end-index 15000

#Input your API key here
API_KEY = "API_KEY_COMES_HERE"


from google import genai
from google.genai import types
import json
import argparse
import logging
import time
import backoff
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to retry the API call with exponential backoff
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def completions_with_backoff(client, **kwargs):
    response = client.models.generate_content(**kwargs)
    return response    

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Prompts and LLM responses')
    
    # Add arguments
    parser.add_argument('--input-file', type=str, required=True, help='Path to the jsonl input file, containing the texts to rewrite')
    parser.add_argument('--output-file', type=str, required=True, help='Path to the jsonl output file, for writing the rewritten texts to')
    parser.add_argument('--end-index', type=int, required=True, help='End index of the data to process')
    
    # Parse the arguments
    args = parser.parse_args()
    logging.info(f"Arguments received: {args}")
        
    # Initialize the Google Gemini API client
    client = genai.Client(api_key=API_KEY)
    
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
        
        # Use the prompt to make the request to the Gemini API
        response = None
        while response is None:  # Retry until we get a valid response
            try:
                # Make the API call using Gemini with fixed max_tokens=2048
                response = completions_with_backoff(
                    client, model="gemini-2.0-flash", contents=prompt, config=types.GenerateContentConfig(
                    max_output_tokens=2048)
                )
                
                # Check if response.text is None
                if response.text is None:
                    logging.warning(f"Received 'None' response for prompt, retrying...")
                    response = None  # Force a retry if text is None
                
            except Exception as e:
                logging.error(f"An error occurred: {e}")
                time.sleep(2)  # Optional: add a small delay before retrying if an exception is thrown

        # Append output to jsonl file once a valid response is obtained
        with open(args.output_file, 'a') as file:
            file.write(json.dumps({'text': response.text}) + '\n')

        end = time.time()
        # Log the time and length of the generated text
        time_taken = end - start
        characters_generated = len(response.text)
        logging.info(f"Characters per second: {characters_generated/time_taken}")

# Entry point of the script
if __name__ == "__main__":
    main()
