import csv
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score

test_data = pd.read_csv('../data/lda/test_lda.csv', header=None)

train_true = pd.read_csv('../data/lda/train_true_lda.csv', header=None)
train_fake = pd.read_csv('../data/lda/train_fake_lda.csv', header=None)

train_data = pd.concat([train_true, train_fake]).to_numpy()
train_target = np.array([1] * len(train_true) + [0] * len(train_fake))


def cv(model, data, target):
    scores = cross_val_score(model, data, target, cv=3)

    print(scores.mean(), scores.std())

    return scores.mean(), scores.std()


def make_test_prediction(model, name):
    pred = model.predict(test_data)

    with open('../data/{}_prediction.csv'.format(name), mode='w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)

        writer.writerow(['Id', 'Category'])

        for i, p in enumerate(pred):
            writer.writerow([i + 1, p])
            print(i, p)
