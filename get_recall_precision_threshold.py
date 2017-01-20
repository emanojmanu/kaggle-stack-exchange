from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt

from utils import clean_dataframe
from utils import load_dataframes

df = load_dataframes()
df = clean_dataframe(df)

tf_idf_transformer = TfidfVectorizer(stop_words='english')
tf_idf = tf_idf_transformer.fit_transform(df.title_content.values)
inv_vocabulary = {v:k for k, v in tf_idf_transformer.vocabulary_.items()}

highest_tf_idf_s = []

for i in range(tf_idf.shape[0]):
    if not i % 10000:
        print(i)
    nzs = tf_idf[i, :].nonzero()[1]
    argmax_ = np.array(tf_idf[i, nzs].todense())[0].argmax()
    highest_tf_idf_s.append(inv_vocabulary[nzs[argmax_]])

# building the lists of frequencies per word and corresponding actual words
freq_list, words_list = [], []
for i in range(tf_idf.shape[0]):
    if not i % 10000:
        print(i)
    nzs = tf_idf[i, :].nonzero()[1]
    freq_list.append(np.array(tf_idf[i, nzs].todense())[0])
    words_list.append([inv_vocabulary[word] for word in nzs])

thresholds, precisions, recalls = [], [], []

for threshold in [0.45, 0.49, 0.53]:
    n_words_found, n_tags, n_correct_prediction = [], [], []
    for i, [freqs, words] in enumerate(zip(freq_list, words_list)):
        # words whose frequency is above the threshold
        words_above_thresh = set(np.array(words)[freqs > threshold])
        # all the tags
        tags = set(df.iloc[i].tags.split())
        n_words_found.append(len(words_above_thresh))
        n_tags.append(len(tags))
        # the correct prediction is the intersection of tags and words of high frequency
        n_correct_prediction.append(len(words_above_thresh.intersection(tags)))

    # save the threshold and the compute recall and precision
    thresholds.append(threshold)
    precisions.append(sum(n_correct_prediction) / sum(n_words_found))
    recalls.append(sum(n_correct_prediction) / sum(n_tags))

f1_score = [2 * pre * rec / (pre + rec) for pre, rec in zip(precisions, recalls)]

plt.plot(thresholds, f1_score, 'r', thresholds, recalls, 'b', thresholds, precisions, 'g')

# for which site does tf-idf work better?
df.groupby('site')['res'].mean()
# site
# biology     0.079645
# cooking     0.352051
# crypto      0.104965
# diy         0.196813
# robotics    0.133887
# travel      0.205405
# Name: res, dtype: float64
