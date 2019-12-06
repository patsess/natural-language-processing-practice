
import os
import numpy as np
import pandas as pd


def read_data(path):
    phrase_df = pd.read_table(
        path + 'dictionary.txt', names=['phrase|index'])
    phrase_df = phrase_df['phrase|index'].str.split('|', expand=True)
    phrase_df = phrase_df.rename(columns={0: 'phrase', 1: 'phrase_ids'})

    sentiment_df = pd.read_table(path + 'sentiment_labels.txt')
    sentiment_df = sentiment_df[
        'phrase ids|sentiment values'].str.split('|', expand=True)
    sentiment_df = sentiment_df.rename(
        columns={0: 'phrase_id', 1: 'sentiment_value'})

    combined_df = phrase_df.merge(
        sentiment_df, how='inner', on='phrase_ids')

    return combined_df


def training_data_split(all_data, spitPercent, data_dir):
    msk = (np.random.rand(len(all_data)) < spitPercent)
    train_only = all_data[msk]
    test_and_dev = all_data[~msk]

    msk_test = (np.random.rand(len(test_and_dev)) < 0.5)
    test_only = test_and_dev[msk_test]
    dev_only = test_and_dev[~msk_test]

    dev_only.to_csv(os.path.join(data_dir, 'processed/dev.csv'))
    train_only.to_csv(os.path.join(data_dir, 'processed/train.csv'))
    test_only.to_csv(os.path.join(data_dir, 'processed/test.csv'))

    return train_only, test_only, dev_only
