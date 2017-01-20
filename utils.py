import pandas as pd
from html2text import html2text
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def clean_dataframe(df):
    """
    Removes html and cleans the text.
    TODO : remove links
    """
    # remove the html stuff
    df.content = df.content.apply(html2text);

    # add a column title_content and clean that column
    # - remove \n, /, .
    # - only keep letters of the alphabet (and space)
    df['title_content'] = df.title + ' ' + df.content
    alphabet = 'abcdefghijklmnopqrstuvwxyz '
    df.title_content = df.title_content.apply(lambda text: ''.join(
        [e for e in text.replace("\n", ' ').replace("/", ' ').replace('.', ' ').lower() if
         e in alphabet]))
    return df


def load_dataframes():
    df = pd.DataFrame()
    for name in ['biology.csv', 'cooking.csv', 'crypto.csv', 'diy.csv', 'robotics.csv',
                 'travel.csv']:
        df_read = pd.read_csv("data/{}".format(name))
        df_read['site'] = name.split('.')[0]
        df = df.append(df_read)

    return df


def build_freq_words_list(df):
    tf_idf_transformer = TfidfVectorizer(stop_words='english')
    tf_idf = tf_idf_transformer.fit_transform(df.title_content.values)
    inv_vocabulary = {v: k for k, v in tf_idf_transformer.vocabulary_.items()}
    # building the lists of frequencies per word and corresponding actual words
    freq_list, words_list = [], []
    for i in range(tf_idf.shape[0]):
        if not i % 5000:
            print("{}%".format(100 * i / df.shape[0]))
        nzs = tf_idf[i, :].nonzero()[1]
        freq_list.append(np.array(tf_idf[i, nzs].todense())[0])
        words_list.append([inv_vocabulary[word] for word in nzs])
    return freq_list, words_list


def tf_idf_prediction(freq_list, words_list, threshold):
    res = []

    for i, [freqs, words] in enumerate(zip(freq_list, words_list)):
        # words whose frequency is above the threshold
        words_above_thresh = np.array(words)[freqs > threshold]
        res.append(words_above_thresh)

    return res
