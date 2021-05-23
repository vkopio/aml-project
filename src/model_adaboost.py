import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.ensemble import AdaBoostClassifier

from common import train_data, train_target, cv

x = np.linspace(0.6,1.2,1000)

for n in [10, 100, 1000]:
    model = AdaBoostClassifier(n_estimators=n, random_state=42)
    cv_score = cv(model, train_data, train_target)

    y = norm.pdf(x, loc=cv_score[0], scale=cv_score[1])
    plt.plot(x, y, label='n = {}'.format(n))

plt.legend()
plt.show()
