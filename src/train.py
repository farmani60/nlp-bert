import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics, model_selection
from transformers import AdamW, get_linear_schedule_with_warmup

from src import config, dataset, engine
from src.data_cleaning import (
    clean,
    convert_abbrev_in_text,
    relabel_targets,
    remove_repetition_subtext_in_df,
)
from src.model import BERTBaseUncased


def train(df, device="gpu"):
    # first split data into single training and validation sets
    df_train, df_valid = model_selection.train_test_split(
        df, test_size=config.TEST_SIZE, random_state=42, stratify=df.keyword.values
    )

    # reset index
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    # initialize BERTDataset from dataset.py
    # for training dataset
    train_dataset = dataset.BERTDataset(tweet=df_train[config.TEXT].values, target=df_train[config.TARGET].values)

    # create training data loader
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4)

    # initialize BERTDataset from dataset.py
    # for valid dataset
    valid_dataset = dataset.BERTDataset(tweet=df_valid[config.TEXT].values, target=df_valid[config.TARGET].values)

    # create valid data loader
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=1)

    # initialize the cuda device cpu/gpu
    device = torch.device(device)

    # load model and send it to the device
    model = BERTBaseUncased()
    model.to(device)

    # create parameters we want to optimize
    # we generally don't use any decay for bias
    # and weight layers
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.001},
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    # calculate the number of training steps
    # this is used by scheduler
    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)

    # AdamW optimizer
    # AdamW optimizer is the most widely used optimizer
    # for transformer based networks
    optimizer = AdamW(optimizer_parameters, lr=config.RELABELED_TARGET)

    # fetch a schedular
    # you can also try using reduce lr on plateau
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_train_steps=num_train_steps)

    # if you have multi gpus
    # model model to DataParallel
    # to use multiple gpus
    model = nn.DataParallel(model)

    # start training the epochs
    best_accuarcy = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = engine.eval_fn(valid_data_loader, model, device)
        outputs = np.array(outputs) >= config.THRESHOLD
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy Score = {accuracy}")
        if accuracy > best_accuarcy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuarcy = accuracy


if __name__ == "__main__":

    print("Load data...")
    df_train = pd.read_csv(config.ORIGINAL_TRAIN_DATA)
    df_test = pd.read_csv(config.ORIGINAL_TEST_DATA)

    # just to test
    df_train = df_train.sample(100, ignore_index=True)
    df_test = df_test.sample(100, ignore_index=True)

    # fill nan values
    df_train["keyword"] = df_train["keyword"].fillna("no_keyword")

    # remove repetition sub texts
    df_train = remove_repetition_subtext_in_df(df_train)
    df_test = remove_repetition_subtext_in_df(df_test)

    # clean tweets
    df_train[config.TEXT] = df_train[config.TEXT].apply(lambda x: clean(x))
    df_test[config.TEXT] = df_test[config.TEXT].apply(lambda x: clean(x))

    # convert abbreviations
    df_train[config.TEXT] = df_train[config.TEXT].apply(lambda x: convert_abbrev_in_text(x))
    df_test[config.TEXT] = df_test[config.TEXT].apply(lambda x: convert_abbrev_in_text(x))

    # fix wrong targets
    df_train = relabel_targets(df_train)

    train(df_train, device="gpu")
