# Data Preparation

All the datasets used in this work are publicly available on Hugging Face, the download links can be found in the [download_urls](https://github.com/neurips-2025-DB/Fingerprints_LLM_Datasets/blob/main/data_preparation/download_urls.md) file.

After downloading the dataset, extract the data into ``.jsonl`` files, which is the format the tokenizer expects. Place all ``.jsonl`` files of the same dataset in one folder. 

## Tokenization

 We tokenize the data using a BPE tokenizer and chunk it into shards. We use the [GPT-NeoX-20B](https://github.com/EleutherAI/gpt-neox) tokenizer. To tokenize and chunk the data, run the following from the root directory (Fingerprints_LLM_Datasets):

```
    python open_lm/datapreprocess/make_2048.py \
    --input-files path_to_input_jsonl_data/*.jsonl
    --output-dir path_to_output_tokenized_data
    --num-workers 32
    --num-consumers 1
```  

This script tokenizes the data, and concatenates tokenized texts to form sequences of length 2048, which is the context length of the transformer. An <|endoftext|> token is added between the different texts. The sequences are divided into shards, where each shard has 8192 sequences. This means each shard has 8192 x 2048 = 16.78M tokens.


## Training Data

After tokenizing the data, all training shards should be placed be in the same directory. The tokenization script names the shards shard-0000000.tar, 	shard-0000001.tar, etc. When doing multi-class classification, the class label should be added to the beginning of the shard's name, as the classification script extracts the label from the shard name. For example, when doing binary classification, the training directory should look like this:

`train_dir` <br/>
       `├──0-shard-0000000.tar`   <br/>
              `├──0-shard-0000001.tar`   <br/>
                  │<br/>
                  │...<br/>
                  │<br/>
                            `├──1-shard-0000000.tar`   <br/>
       `└──1-shard-0000001.tar`   <br/>



## Manifest File

By default, the open_lm repoistory picks shards to train on by sampling with replacement. To sample without replacement, which is recommended, we create a manifest file which specifies the exact shards to train on. The training script then samples from the specified shards without replacement. To create the manifest file, run the following from the root directory (Fingerprints_LLM_Datasets):

```
python -m open_lm.utils.make_wds_manifest --data-dir path_to_train_dir
```

This will create a ``manifest.jsonl`` in the same directory as the training shards. This file is used in the training script to specify the path of the training data. 

## Test data

Each test sequence is classified individually as a whole. There is no concatenation or splitting like in training. The test sequences from one dataset are tokenized and saved as a list of tensors. To prepare the test data, run the following from the root directory (Fingerprints_LLM_Datasets): 

```
    python open_lm/datapreprocess/prepare_test_data.py \
    --input-file path_to_test_set/test.jsonl
    --output-file path_to_tokenized_test_data/test.pt
```  
