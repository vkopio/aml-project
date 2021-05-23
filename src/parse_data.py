import numpy as np
import pandas as pd

train_true = pd.read_csv('../data/train_true_bow.csv')
train_fake = pd.read_csv('../data/train_fake_bow.csv')

sample = pd.concat(
  [train_true, train_fake]
).fillna(0).astype('int64').to_numpy()

target = np.array([1] * len(train_true) + [0] * len(train_fake))
