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
BERT_PATH = DATA_DIR + "bert_based_uncased/"
ORIGINAL_TRAIN_DATA = DATA_DIR + "train.csv"
ORIGINAL_TEST_DATA = DATA_DIR + "test.csv"
SUBMISSION = DATA_DIR + "sample_submission.csv"

# where to save the model
MODEL_PATH = DATA_DIR + "model/model.bin"

# define the tokenizer
# we use tokenizer and model
# from huggingface's transformers
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
