import csv
import re
from pprint import pprint
import matplotlib.pyplot as plt
import gensim
from gensim.models import CoherenceModel, LdaMulticore
import gensim.corpora as corpora
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


def tokenize_text(t):
    ps = PorterStemmer()
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    return [ps.stem(w) for w in tokenizer.tokenize(t) if w.lower() not in stop_words]


def tokenize_document(f):
    data = []
    for line in f:
        title = tokenize_text(line)
        data.append(title)
    return data


with open('../data/train_fake.txt') as f:
    fake = tokenize_document(f)

with open('../data/train_true.txt') as f:
    true = tokenize_document(f)

with open('../data/test.txt') as f:
    test = tokenize_document(f)


id2word = corpora.Dictionary(true + fake)

true_corpus = [id2word.doc2bow(text) for text in true]
fake_corpus = [id2word.doc2bow(text) for text in fake]
test_corpus = [id2word.doc2bow(text) for text in test]

mallet_path = '../mallet-2.0.8/bin/mallet'

num_topics = 210

lda_model = gensim.models.wrappers.LdaMallet(
    mallet_path,
    alpha=1200,
    corpus=true_corpus + fake_corpus,
    id2word=id2word,
    num_topics=num_topics,
    random_seed=42)


with open('../data/lda/train_fake_lda.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')

    for row in lda_model[fake_corpus]:
        csv_row = [0.0] * num_topics

        for topic in row:
            csv_row[topic[0]] = topic[1]

        writer.writerow(csv_row)

with open('../data/lda/train_true_lda.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')

    for row in lda_model[true_corpus]:
        csv_row = [0.0] * num_topics

        for topic in row:
            csv_row[topic[0]] = topic[1]

        writer.writerow(csv_row)


with open('../data/lda/test_lda.csv', mode='w') as f:
    writer = csv.writer(f, delimiter=',')

    for row in lda_model[test_corpus]:
        csv_row = [0.0] * num_topics

        for topic in row:
            csv_row[topic[0]] = topic[1]

        writer.writerow(csv_row)
        print(csv_row)
