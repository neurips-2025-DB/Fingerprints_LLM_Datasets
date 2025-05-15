# Description: A script for classifying text accorting to topics with gpt4-mini

# Batches are submitted and retrieved following https://platform.openai.com/docs/guides/batch/getting-started

# usage:
#   Submit a batch request, to classify number-examples from the dataset, copy the batch number printed when submitting:
#       python categorise_text.py --submit --number-examples 2000 --input-file input_file.jsonl
#   Retrieve the batch results, and save them to the output file:  
#       python categorise_text.py --retrieve BATCH_NUMBER --output-file output_results.jsonl
# 
# An openai account is required, and the API key should be set as an environment variable. 
# export OPENAI_API_KEY="your-api-key"

from openai import OpenAI
import json
import argparse
import logging
from datasets import load_dataset


def print_dataset_information(dataset):
    urls = {}
    for entry in dataset["train"]:
        url = entry["url"]
        url = url.split("//")[-1].split("/")[0]
        # remove www if applicable
        if url.startswith("www."):
            url = url[4:]
        if url in urls:
            urls[url] += 1
        else:
            urls[url] = 1

    # sort the urls by frequency
    urls = dict(sorted(urls.items(), key=lambda item: item[1], reverse=True))

    # print the number of unique urls
    print(f"Number of unique urls: {len(urls)}")
    # print the total number of urls
    print(f"Total number of urls: {sum(urls.values())}")

    # print the 25 most frequent urls
    print("25 most frequent urls:")
    for url, count in list(urls.items())[:25]:
        print(f"\t{url}: {count}")


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Rewriting script.")

    # Add arguments
    parser.add_argument('--retrieve', type=str, required=False, help='Batch id to retrieve')
    parser.add_argument('--number-examples', type=int, default=100, required=False, help='Maximum number of examples')
    parser.add_argument('--submit', action='store_true', help="Generate and submit a batch request")
    parser.add_argument('--input-file', type=str, required=True, help='Dataset jsonl file')
    parser.add_argument('--output-file', type=str, default="results.jsonl", required=False, help='File to write results to')

    # Parse the arguments
    args = parser.parse_args()
    logging.info(f"Arguments received: {args}")

    thematic_categories = ['Advertisement', 'Business and Finance', 'News and Media', 'Politics and Policy', 'Science', 'Technology', 'Health, Wellness, and Fitness', 'Food and Nutrition', 'Arts and Entertainment', 'Society and Culture', 'Lifestyle and Recreation', 'Education', 'Community', 'Other']
    purpose_categories = ['Informative', 'Analytical/Explanatory', 'Educational/Instructional', 'Persuasive', 'Promotional', 'Narrative/Storytelling', 'Opinionated', 'Other']

    prompt = """Classify the text according to the thematic categories: 
    `Advertisement', 
    `Business and Finance', 
    `News and Media',
    `Politics and Policy',
    `Science' (physics, biology, chemistry, astronomy, and academic studies), 
    `Technology' (software, hardware, and gadgets),
    'Health, Wellness, and Fitness',
    `Food and Nutrition',
    `Arts and Entertainment', 
    `Society and Culture',
    `Lifestyle and Recreation',
    `Education',
    `Community' (user reviews, forums, social media, etc), 
    `Other'. 
    If the text is an advertisement, always choose `Advertisement', else choose the most appropriate category. 
    If none of the categories are appropriate, choose `Other'.
    Also classify the text according to the purpose of the text:
    `Informative',
    `Analytical/Explanatory',
    `Educational/Instructional',
    `Persuasive',
    `Promotional',
    `Narrative/Storytelling',
    `Opinionated',
    `Other'.
    Justiy your answer with an explanation, and end with the names of the two categories."""

    if args.submit:
        ### generate ..._batch.jsonl file for the batch request

        # 1. Write the batch input file

        batch_file_name = "queries_batch.jsonl"
        with open(batch_file_name, "w") as output_file:
            with open(args.input_file, "r") as input_file:
                for line_number, line in enumerate(input_file):
                    record = json.loads(line.strip())
                    text = record["text"]
                    json_data = json.dumps({'custom_id': f"request-{line_number}", 
                                        'method': 'POST', 
                                        'url': '/v1/chat/completions', 
                                        'body': {'model': 'gpt-4o-mini',
                                        #'messages': [{'role': 'system', 'content': "You are a helpful assistant"}, 
                                        #            {"role": "user", "content": prompt + text}], 
                                        'messages': [{'role': 'system', 'content': prompt}, 
                                                    {"role": "user", "content": text}], 
                                        'max_tokens': 2048}})
                    
                    # Write without adding newline at the end for the last item
                    if line_number < args.number_examples - 1:
                        output_file.write(json_data + "\n")  # Add newline between entries
                    elif line_number == args.number_examples - 1:
                        output_file.write(json_data)  # No newline for the last entry
                    else:
                        break
                    if line_number >= args.number_examples - 1:
                        break

        # 2. upload batch input file
        client = OpenAI()

        batch_input_file = client.files.create(file=open(batch_file_name, "rb"), purpose="batch")

        batch_input_file_id = batch_input_file.id

        batch = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "Classify texts in " + args.input_file},
        )
        logging.info(f"submitted batch request with id {batch.id}")

    elif args.retrieve:

        # 1. check batch status
        client = OpenAI()
        batch = client.batches.retrieve(args.retrieve)
        logging.info("Status: %s", batch.status)
        logging.info("Output file id: %s", batch.output_file_id)
        logging.info("Request counts: %s", batch.request_counts)

        # 2. retrieving the results
        if batch.status == "completed":
            # 4. downloading the batch output file
            client = OpenAI()
            file_response = client.files.content(batch.output_file_id)
            # file_response is the content of the JSONL file, which is iterable


            # dictionary to count the number of examples in each category + invalid
            thematic_category_counts = {category: 0 for category in thematic_categories}
            thematic_category_counts["Invalid"] = 0
            purpose_categories_counts = {category: 0 for category in purpose_categories}
            purpose_categories_counts["Invalid"] = 0

            with open(args.output_file, 'w') as output_file, open(args.input_file, "r") as input_file:
                
                for i, (line, line_input) in enumerate(zip(file_response.text.split("\n"), input_file)):

                    if line.strip():  # Check if the line has content to avoid empty lines
                        data = json.loads(line)
                        # load one line from input_file
                        data_input = json.loads(line_input.strip())
                        text = data_input["text"]
                    
                        response = data['response']['body']['choices'][0]['message']['content']
                        response_end = response[-100:]
                        custom_id = data['custom_id']
                        # extract number from custom_id
                        idx = int(custom_id.split("-")[-1])
                        # check if idx is equal to i
                        assert idx == i, f"idx: {idx}, i: {i}"

                        # check which thematic category is in the response
                        for category in thematic_categories:
                            if category in response_end:
                                thematic_label = category
                                break
                        else:
                            thematic_label = "Invalid"
                        thematic_category_counts[thematic_label] += 1

                        # check which purpose category is in the response
                        for category in purpose_categories:
                            if category in response_end:
                                purpose_label = category
                                break
                        else:
                            purpose_label = "Invalid"
                        purpose_categories_counts[purpose_label] += 1


                        
                        json_data = json.dumps({'thematic_label': thematic_label,'purpose_label': purpose_label, 'text': text, 'response': response})
                        output_file.write(json_data + "\n")
            
            # sum over dictionary to get number of examples
            number_examples = sum(thematic_category_counts.values())
            # print the thematic category counts formated well
            print("Thematic category counts:")
            for category, count in thematic_category_counts.items():
                print(f"\t{category}: {count/number_examples*100:.2f}%")
            print("Purpose category counts:")
            for category, count in purpose_categories_counts.items():
                print(f"\t{category}: {count/number_examples*100:.2f}%")

# Entry point of the script
if __name__ == "__main__":
    main()
