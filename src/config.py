import os

import transformers

# this is the maximum number of tokens in a sentence
MAX_LEN = 512

# batch size is small because the model is huge
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4

# number of epochs
EPOCHS = 10

# data
DATA_DIR = "../input/"
ORIGINAL_TRAIN_DATA = DATA_DIR + "train.csv"
ORIGINAL_TEST_DATA = DATA_DIR + "test.csv"
SUBMISSION = DATA_DIR + "sample_submission.csv"
BERT_MODEL = "bert-base-uncased"
CACHE_DIR = os.path.join(DATA_DIR, "transformers-cache")

# where to save the model
MODEL_PATH = DATA_DIR + "model/model.bin"

# define the tokenizer
# we use tokenizer and model
# from huggingface's transformers
tokenizer = transformers.BertTokenizer.from_pretrained(BERT_MODEL, cache_dir=CACHE_DIR, do_lower_case=True)
