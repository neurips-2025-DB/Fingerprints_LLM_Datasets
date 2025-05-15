import requests
import random
import gzip
import json
import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Base URL for downloading the files
base_url = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.{:05d}-of-01024.json.gz"

# Function to download and process a single file
def download_and_process_file(file_index, output_dir, progress_bar):
    # Construct the URL with the file index
    url = base_url.format(file_index)
    
    # Download the .json.gz file
    response = requests.get(url)
    if response.status_code == 200:
        gz_file = os.path.join(output_dir, f"c4-train.{file_index:05d}.json.gz")
        
        # Save the gzipped file
        with open(gz_file, "wb") as f:
            f.write(response.content)
        
        # Extract and process the .json.gz file
        with gzip.open(gz_file, 'rt', encoding='utf-8') as f:
            # Read JSON data from the gzipped file and write it to a .jsonl file
            json_lines = [json.loads(line) for line in f]
            
            # Keep only the first 7128 lines
            json_lines = json_lines[:7128]
            
            # Write to a new jsonl file
            jsonl_file = os.path.join(output_dir, f"c4-train.{file_index:05d}.jsonl")
            with open(jsonl_file, 'w', encoding='utf-8') as out_f:
                for line in json_lines:
                    json.dump(line, out_f)
                    out_f.write("\n")
        
        # Delete the .gz file after processing
        os.remove(gz_file)  
        
        progress_bar.update(1)  # Update the progress bar
        print(f"File {file_index:05d} processed and saved as {jsonl_file}")
    else:
        print(f"Failed to download {url}")

# Main function to handle the overall task
def main(output_dir, num_files=50, num_cores=50):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Randomly pick `num_files` indices between 0 and 1023
    random_indices = random.sample(range(1024), num_files)
    
    # Set up the progress bar
    with tqdm(total=num_files, desc="Processing files", unit="file") as progress_bar:
        # Using ThreadPoolExecutor to run the download and process function in parallel
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            # Submit tasks to the executor for each randomly picked file index
            futures = [executor.submit(download_and_process_file, index, output_dir, progress_bar) for index in random_indices]
            
            # Wait for all futures to complete
            for future in futures:
                future.result()  # If there are exceptions, they will be raised here

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Download and process C4 dataset files in parallel.")
    parser.add_argument('output_dir', type=str, help="Directory where the resulting files will be saved.")
    parser.add_argument('--cores', type=int, default=50, help="Number of CPU cores (threads) to use for parallel processing.")
    args = parser.parse_args()
    
    # Call the main function with the specified output directory and number of cores
    main(args.output_dir, num_cores=args.cores)
