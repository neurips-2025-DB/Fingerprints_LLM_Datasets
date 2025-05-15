# Download Prompt Dataset

Start by downloading the required dataset with the prompts. Note that only the prompts are used, not the responses. 
For example, OpenHermes-2.5 can be downloaded from [Huggingface](https://huggingface.co/datasets/teknium/OpenHermes-2.5).
After downloading the dataset, extract the data into ``.jsonl`` file, which is the format the scripts expect.

## Shuffling

You can shuffle the data using the [shuffle](https://github.com/neurips-2025-DB/Fingerprints_LLM_Datasets/blob/main/finetuned_models/shuffle.py) script. To do so run:

```
python shuffle.py input_file.jsonl output_file.jsonl
```

# API Requests
To generate responses from the six finetuned LLMs run the respective file. The command for running the script is found on the top of each script, and follows the format:

```
python LLM.py --input-file prompt_dataset.jsonl --output-file LLM_responses.jsonl --end-index 15000
```
where ``--end-index`` is the number of prompts to generate responses for. 

An API key is required for generating the responses. For GPT, Claude, Gemini obtain the API key from OpenAI, Anthropic, Google respectively. For DeepSeek, Qwen, Llama obtain the API key from Together AI.

# Classification
After running all the scripts, you will get jsonl files containing the data to be classified. Then follow the steps in Data Preparation, Classification, and Evaluation as before. 
