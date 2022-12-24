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
ORIGINAL_TRAIN_DATA = os.path.join(DATA_DIR, "train.csv")
ORIGINAL_TEST_DATA = os.path.join(DATA_DIR, "test.csv")
SUBMISSION = os.path.join(DATA_DIR, "sample_submission.csv")
BERT_MODEL = "bert-base-uncased"
CACHE_DIR = os.path.join(DATA_DIR, "transformers-cache")
BERT_PATH = os.path.join(CACHE_DIR, "models--bert-base-uncased")

# where to save the model
MODEL_PATH = DATA_DIR + "model/model.bin"

# Next we specify the pre-trained BERT model we are going to use. The
# model `"bert-base-uncased"` is the lowercased "base" model
# (12-layer, 768-hidden, 12-heads, 110M parameters).
#
# We load the used vocabulary from the BERT model, and use the BERT
# tokenizer to convert the sentences into tokens that match the data
# the BERT model was trained on.
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_MODEL, cache_dir=CACHE_DIR, do_lower_case=True)