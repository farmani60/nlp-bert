import torch

from src import config


class BERTDataset:
    def __init__(self, tweet, target):
        self.tweet = tweet
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        # return the length of dataset
        return len(self.tweet)

    def __getitem__(self, item):
        # for a given item index return a dictionary of inputs
        tweet = str(self.tweet[item])
        tweet = " ".join(tweet.split())

        # encode_plus comes from huggingface's transformers
        # and exists for all tokenizers they offer
        # it can be used to convert any offer
        # to ids, mask, and token type ids which are
        # needed for models like BERT
        # here, tweet is a string
        inputs = self.tokenizer.encode_plus(
            tweet,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
        )

        # ids are ids of tokens generated
        # after tokenizing tweets
        ids = inputs["input_ids"]

        # mask is 1 where we have input
        # and 0 where we have padding
        mask = inputs["attention_mask"]

        # token type ids behave the same way as
        # mask in this specific case
        # in case of two sentences, this is 0
        # for first sentence and 1 for second sentence
        token_type_ids = inputs["token_type_ids"]

        # now we return everything
        # note that ids, mask and token_tye_ids
        # are all long datatypes and targets is float
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.target[item], dtype=torch.float),
        }


class TESTDataset:
    def __init__(
        self,
        tweet,
    ):
        self.tweet = tweet
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        tweet = str(self.tweet[item])
        tweet = " ".join(tweet.split())

        inputs = self.tokenizer.encode_plus(
            tweet,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }
