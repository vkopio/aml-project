import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score

train_true = pd.read_csv('../data/train_true_bow.csv', header=None)
train_fake = pd.read_csv('../data/train_fake_bow.csv', header=None)

train_data = pd.concat([train_true, train_fake]).to_numpy()
train_target = np.array([1] * len(train_true) + [0] * len(train_fake))

def cv(model, data, target):
    scores = cross_val_score(model, data, target, cv=10)

    return scores.mean(), scores.std()
