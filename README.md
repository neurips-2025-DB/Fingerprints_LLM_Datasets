This repository is based on the [Open LM](https://github.com/mlfoundations/open_lm) repository, which we modified to allow for text classification. 

## Installation
We require python >=3.9 as well as several other packages. Start by cloning our project, and then installing the neccessary requirements as follows:

```
git clone https://github.com/neurips-2025-DB/Fingerprints_LLM_Datasets
cd Fingerprints_LLM_Datasets
pip install -r requirements.txt
pip install --editable .
```

## Data Preparation

Check the [data preparation](https://github.com/neurips-2025-DB/Fingerprints_LLM_Datasets/tree/main/data_preparation) section for instructions on how to download and process the datasets.

## Pretraining

The classification model is first pretrained to predict the next token. Run the following command to run pretraining: 

```
torchrun --nproc-per-node 8 -m open_lm.main   \
 --model open_lm_160m \
 --dataset-manifest /preproc_data/manifest.jsonl \
 --train-num-samples 3200000000 \
 --epochs 1 \
 --workers 8 \
 --precision amp_bfloat16 \
 --global-batch-size 16 \
 --grad-checkpointing \
 --log-every-n-steps 100 \
 --grad-clip-norm 1 \
 --data-key txt \
 --lr 3e-4 \
 --fsdp --fsdp-amp \
 --warmup 2000 \
 --wd 0.1 \
 --beta2 0.95 \
 --resume latest \
 --report-to wandb \
 --wandb-project-name name_of_the_run \
 --logs path_to_logging_directory
 --name name_of_the_run \
```

Some of the important arguments are:

- `nproc-per-node`: Number of GPUs
- `model`: Model size, our default model size is 160M. The available model sizes can be found in [model_configs](https://github.com/neurips-2025-DB/Fingerprints_LLM_Datasets/tree/main/open_lm/model_configs)
- `dataset-manifest`: Path to the manifest file
- `train-num-samples`: Number of tokens per epoch. For the 160M model, 3.2B tokens are used (Chinchilla optimal)
- `epochs`: Model weights and optimizer are saved every epoch. To save intermediate checkpoints, set it to a higher value. For example setting `epochs` to 10, and `train-num-samples` to 320M will overall use 3.2B tokens
- `report-to wandb` and `wandb-project-name`: Omit if logging to wandb is not desired
- `logs`: Path where logging files and checkpoints are saved
- `name`: Project name. This creates a directory in `logs` with the project name


## Classification

The command for classification is similar to pretraining, but the following three arguments are added:

```
--classification True \
--num-classes 3 \
--classif-model-path path_to_pretrained_model
```

- `classification`: Indicates that we are doing classification not pretraining. Default value is False
- `num-classes`: Number of classification classes
- `classif-model-path`: Path to pretrained model. Can be omitted if you want to run classification from scratch, instead of finetuning from a pretrained model

## Evaluation

To evaluate the classification model, run the following command:

```
python open_lm/eval.py \
  --model open_lm_160m \
  --classif-model-path path_to_classification_model \
  --num-classes 3 \
  --test-sets C4 FW RW \
  --base-path path_to_test_sets
```

This example evaluates a 3-way classifier. The test sets (`C4.pt`, `FW.pt`, `RW.pt`) are specified with the same order as during training: C4 (class 0), FW (class 1), RW (class 2), and should be placed in `base-path`. Ensure that the number of strings in `test-sets` matches `num-classes`. The script adds the `.pt` extension automatically to the strings in `test-sets`. The script runs on one GPU by default.  

## Generalisation

For calculating the perplexity for the generalisation experiments, follow the previous instructions on preparing and tokenizing 3.2B training tokens from each of the seven main datasets (C4, FineWeb, RefinedWeb, DolmaCC, RedPajama-V2, DCLM, FineWeb-Edu). Pretrain a 160M transformer on each of the datasets. Then pretrain another transformer on a mixture of all datasets (3.2B tokens equally sampled from each dataset). 

Also prepare 1000 unseen evaluation sequences from each dataset. The evaluation sequences should be in `.jsonl` format, and in their raw text form (untokenized). Make sure they have the text sequences in the field `'text'`. 

Run `cd perplexity` and create a directory `pretrained_models` and place all the pretrained models in it. Also create a directory `cross_dataset` and place the evaluation sequences in it (1 jsonl file for each dataset, each jsonl file has 1000 lines). 

To download and prepare the benchmark data: WikiText-103 and Paloma, simply run:
```
python wikitext.py
python paloma.py
```
which will store the data in the required format.

To calculate the perplexity run the notebook [Evaluate_perplexity](https://github.com/neurips-2025-DB/Fingerprints_LLM_Datasets/blob/main/perplexity/Evaluate_perplexity.ipynb), and set the model path and the directory to the evaluation data. 

## Rewriting

We rewrite text with OpenAI's batch API. After obtaining an API key, set it as an environment variable with export `OPENAI_API_KEY= YOUR_API_KEY`, then run the following command to rephrase text:  

```
python scripts/rewrite_texts_batch_auto.py \
  --input-file path_to_input_file \
  --output-file path_to_output_file \
  --batch-size 2000 \
  --prompt prompt1
```

- `input-file` and `output-file`: jsonl files containing the original and rephrased texts respectively. The text is assumed to have the key "text"
- `batch-size`: number of sequences being rephrased. Set to 2000 for a tier 1 OpenAI account
- `prompt`: rephrasing prompt. Set to prompt1 or prompt2 or prompt3

## Formatting Removal

To remove formatting, run the following command:

```
python scripts/remove_formatting.py input_file.jsonl output_file.jsonl
```

## Bag of Words

To train a bag of words model for a 2-way classification task, run the following command:

```
python scripts/train_BoW.py class0_train.jsonl class1_train.jsonl
```

For evaluation:

```
python scripts/test_BoW.py class0_test.jsonl class1_test.jsonl
```

## Dataset Categorization

Similar to rewriting, classifying text sequences of a dataset into one of the 13 thematic categories requires an OpenAI API key. After setting the API key as an environment variable run:

```
python scripts/categorise_text.py --submit --number-examples 2000 --input-file input_file.jsonl
```

where input file is a jsonl file with keys "text" and "url". This will print a batch number that should be copied and used to retrieve the results with: 

```
python scripts/categorise_text.py --retrieve BATCH_NUMBER --output-file output_results.jsonl
```

## Data Generation

We generate data from an LLM by prompting it with one single token with the following command: 

```
python scripts/generate_random_sequences.py \
  --hf-model apple/DCLM-Baseline-7B \
  --batch-size 16 \
  --num-seqs 8192 \
  --max-new-tokens 800 \
  --output-file path_to_output_file.jsonl \
  --input-file path_to_input_file.jsonl
```

- `hf-model`: HuggingFace model. A list of all models used in the paper is found at the top of the [script](https://github.com/neurips-2025-DB/Fingerprints_LLM_Datasets/blob/main/scripts/generate_random_sequences.py)
- `batch-size`: Batch size. Scale it to fill the GPU
- `num-seqs`: Number of sequences to generate
- `max-new-tokens`: Maximal number of tokens per sequence to generate
- `output-file`: Jsonl file where the generated sequences are saved
- `input-file`: Jsonl file from which the first token of each sequence is used to prompt the LLM. Must have equal or more sequences than `num-seqs`. If `input-file` is not specified, a token will be drawn uniformly at random

## Mixture Proportions Estimation

To estimate the mixture proportions of the domains an LLM was trained on, first train a classifier to distinguish between the potential domains. Second generate random sequences from the LLM (do not specify `--input-file`) and tokenize them into tensors as described [here](https://github.com/neurips-2025-DB/Fingerprints_LLM_Datasets/tree/main/data_preparation) under test data. Finally run the following command to classify the generated sequences:

```
python open_lm/classify.py \
  --model open_lm_160m \
  --classif-model-path path_to_classification_model \
  --num-classes 7 \
  --generated-data-path path_to_data_generated_from_LLM.pt
```

## Finetuned Models

Check the [finetuned models](https://github.com/neurips-2025-DB/Fingerprints_LLM_Datasets/tree/main/finetuned_models) section.
