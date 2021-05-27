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


counts = {}

for title in fake:
    for w in title:
        counts[w] = counts.get(w, 0) + 1

for title in true:
    for w in title:
        counts[w] = counts.get(w, 0) + 1

# Create Dictionary
id2word = corpora.Dictionary(true + fake)
# Create Corpus
# Term Document Frequency
true_corpus = [id2word.doc2bow(text) for text in true]
fake_corpus = [id2word.doc2bow(text) for text in fake]
test_corpus = [id2word.doc2bow(text) for text in test]

mallet_path = '../mallet-2.0.8/bin/mallet' 

num_topics = 32
# Build LDA model
lda_model = gensim.models.wrappers.LdaMallet(
    mallet_path, 
    corpus=true_corpus + fake_corpus,
    id2word=id2word,
    num_topics=num_topics,
    random_seed=42)

# def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
#     coherence_values = []
#     model_list = []

#     for num_topics in range(start, limit, step):
#         model = gensim.models.wrappers.LdaMallet(
#             mallet_path, 
#             corpus=corpus, 
#             num_topics=num_topics, 
#             id2word=id2word,
#             random_seed=42)

#         model_list.append(model)

#         coherencemodel = CoherenceModel(
#             model=model, 
#             texts=texts, 
#             dictionary=dictionary, 
#             coherence='c_v')
#         coherence_values.append(coherencemodel.get_coherence())

#     return model_list, coherence_values

# limit = 40
# start = 8
# step = 4

# model_list, coherence_values = compute_coherence_values(
#     dictionary=id2word, 
#     corpus=true_corpus + fake_corpus, 
#     texts=true + fake, 
#     start=start, 
#     limit=limit, 
#     step=step)


# x = range(start, limit, step)
# plt.plot(x, coherence_values)
# plt.xlabel("Num Topics")
# plt.ylabel("Coherence score")
# plt.legend(("coherence_values"), loc='best')
# plt.show()


cnt = 40

# sort words by their total count
# extract top 300 words
# transform them into a word -> index hash map
features = {x[0]: i for i, x in enumerate(
    sorted(counts.items(), key=lambda kv: kv[1])[-cnt:])}

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
