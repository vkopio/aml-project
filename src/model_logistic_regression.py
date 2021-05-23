import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from common import train_data, train_target, cv

logistic_regression_model = LogisticRegression(random_state=42)

cv_score = cv(logistic_regression_model, train_data, train_target)

x = np.linspace(0.5,1.2,1000)
y = norm.pdf(x, loc=cv_score[0], scale=cv_score[1])

plt.plot(x, y)
plt.legend()
plt.show()
