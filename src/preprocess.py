import csv
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

stop_words = set(stopwords.words('english'))



def tokenize_text(t):
	ps = PorterStemmer()
	tokenizer = nltk.RegexpTokenizer(r"\w+")
	return [ps.stem(w) for w in tokenizer.tokenize(t) if w.lower() not in stop_words]
	#return [x.lower() for x in re.sub('[^a-zA-Z]+', ' ', t).split() if len(x) > 2]
	
def tokenize_document(f):
	data = []
	for line in f:
		title = tokenize_text(line)
		data.append(title)
	return data


f = open('train_fake.txt')
fake = tokenize_document(f)

f = open('train_true.txt')
true = tokenize_document(f)

# let's not do feature selection on test data just to be pedantic
f = open('test.txt')
test = tokenize_document(f)


counts = {}

for title in fake:
	for w in title:
		counts[w] = counts.get(w, 0) + 1

for title in true:
	for w in title:
		counts[w] = counts.get(w, 0) + 1


cnt = 40

# sort words by their total count
# extract top 300 words
# transform them into a word -> index hash map
features = {x[0] : i for i,x in enumerate(sorted(counts.items(), key=lambda kv: kv[1])[-cnt:])}

f = open('train_fake_bow.csv', 'w')
writer = csv.writer(f, delimiter=',')

for title in fake:
	x = [0] * cnt
	for w in title:
		if w in features:
			x[features[w]] += 1
	writer.writerow(x)

f = open('train_true_bow.csv', 'w')
writer = csv.writer(f, delimiter=',')

for title in true:
	x = [0] * cnt
	for w in title:
		if w in features:
			x[features[w]] += 1
	writer.writerow(x)
	
f = open('test_bow.csv', 'w')
writer = csv.writer(f, delimiter=',')

for title in test:
	x = [0] * cnt
	for w in title:
		if w in features:
			x[features[w]] += 1
	writer.writerow(x)
