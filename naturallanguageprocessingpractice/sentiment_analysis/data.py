
import os
import numpy as np
import pandas as pd


# Combine and split the data into train and test
def read_data(path):
    # read dictionary into df
    df_data_sentence = pd.read_table(
        path + 'dictionary.txt', names=['Phrase|Index'])
    df_data_sentence_processed = df_data_sentence['Phrase|Index'].str.split(
        '|', expand=True)
    df_data_sentence_processed = df_data_sentence_processed.rename(
        columns={0: 'Phrase', 1: 'phrase_ids'})

    # read sentiment labels into df
    df_data_sentiment = pd.read_table(path + 'sentiment_labels.txt')
    df_data_sentiment_processed = df_data_sentiment[
        'phrase ids|sentiment values'].str.split('|', expand=True)
    df_data_sentiment_processed = df_data_sentiment_processed.rename(
        columns={0: 'phrase_ids', 1: 'sentiment_values'})

    # combine data frames containing sentence and sentiment
    df_processed_all = df_data_sentence_processed.merge(
        df_data_sentiment_processed, how='inner', on='phrase_ids')

    return df_processed_all


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
