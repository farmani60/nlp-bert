import pandas as pd
from sklearn import model_selection

from src import config
from src.data_cleaning import (
    clean_tweets,
    convert_abbrev_in_text,
    relabel_targets,
    remove_emoji,
    remove_punctuations,
)


def train(df):
    # first split data into single training and validation sets
    df_train, df_validation = model_selection.train_test_split(
        df, test_size=config.TEST_SIZE, random_state=42, stratify=df.keyword.values
    )


if __name__ == "__main__":

    print("Load data...")
    df_train = pd.read_csv(config.ORIGINAL_TRAIN_DATA)
    df_test = pd.read_csv(config.ORIGINAL_TEST_DATA)

    # just to test
    df_train = df_train.sample(100)
    df_test = df_test.sample(100)

    # fill nan values
    df_train["keyword"] = df_train["keyword"].fillna("no_keyword")

    # clean tweets
    df_train[config.TEXT] = df_train[config.TEXT].apply(lambda x: clean_tweets(x))
    df_test[config.TEXT] = df_test[config.TEXT].apply(lambda x: clean_tweets(x))

    # remove emojis
    df_train[config.TEXT] = df_train[config.TEXT].apply(lambda x: remove_emoji(x))
    df_test[config.TEXT] = df_test[config.TEXT].apply(lambda x: remove_emoji(x))

    # remove punctuations
    df_train[config.TEXT] = df_train[config.TEXT].apply(lambda x: remove_punctuations(x))
    df_test[config.TEXT] = df_test[config.TEXT].apply(lambda x: remove_punctuations(x))

    # convert abbreviations
    df_train[config.TEXT] = df_train[config.TEXT].apply(lambda x: convert_abbrev_in_text(x))
    df_test[config.TEXT] = df_test[config.TEXT].apply(lambda x: convert_abbrev_in_text(x))

    # fix wrong targets
    df_train = relabel_targets(df_train)
