import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier

from common import train_data, train_target, cv, make_test_prediction

# x = np.linspace(0.6,1.2,1000)

# for n in [10, 100, 1000]:
#     model = AdaBoostClassifier(n_estimators=n, random_state=42)
#     cv_score = cv(model, train_data, train_target)

#     y = norm.pdf(x, loc=cv_score[0], scale=cv_score[1])
#     plt.plot(x, y, label='n = {}'.format(n))

# plt.legend()
# plt.show()

gboost = GradientBoostingClassifier(
  n_estimators=2000, 
  learning_rate=1.0,
  max_depth=1, 
  random_state=42)

cv_score = cv(gboost, train_data, train_target)

#gboost.fit(train_data, train_target)

#make_test_prediction(gboost, 'gradientboost')
