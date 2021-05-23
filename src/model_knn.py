import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.neighbors import KNeighborsClassifier

from common import train_data, train_target, cv

x = np.linspace(0.6,1.2,1000)

for k in range(3, 18, 2):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_cv = cv(knn_model, train_data, train_target)

    y = norm.pdf(x, loc=knn_cv[0], scale=knn_cv[1])
    plt.plot(x, y, label='k = {}'.format(k))

plt.legend()
plt.show()
