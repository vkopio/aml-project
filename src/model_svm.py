import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn import svm

from common import train_data, train_target, cv

svm_model = svm.NuSVC(gamma='auto')

cv_score = cv(svm_model, train_data, train_target)

x = np.linspace(0.5,1.2,1000)
y = norm.pdf(x, loc=cv_score[0], scale=cv_score[1])

plt.plot(x, y)
plt.legend()
plt.show()
