import torch.nn as nn
import transformers

from src import config


class BERTBaseUncased(nn.Module):
    _dropout_rate = 0.3

    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        # we fetch the model from CACHE_DIR defined in
        # config.py
        self.bert = transformers.BertModel.from_pretrained(config.BERT_MODEL, cache_dir=config.CACHE_DIR)
        # add a dropout for regularization
        self.bert_drop = nn.Dropout(self._dropout_rate)
        # as there is only one output, we add a
        # single layer to the model
        self.linear = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        o2 = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
        )
        # pass through dropout layer
        bo = self.bert_drop(o2[1])
        # pass through linear layer
        output = self.linear(bo)
        # return output
        return output
