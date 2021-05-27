import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from common import train_data, train_target, cv

xgb = XGBRegressor(objective='binary:hinge', n_estimators=10000)

X_train, X_test, y_train, y_test = train_test_split(
     train_data, train_target, test_size=0.33, random_state=42)

#cv_score = cv(xgb, train_data, train_target)

xgb.fit(X_train, y_train)

test_pred = xgb.predict(X_test)
print(accuracy_score(y_test, test_pred))

# for t in range(3, 8):
#   train_pred = [0 if x < (0.1 * t) else 1 for x in xgb.predict(train_data)]

#   print(accuracy_score(train_target, train_pred))

# x = np.linspace(0.5,1.2,1000)
# y = norm.pdf(x, loc=cv_score[0], scale=cv_score[1])

# plt.plot(x, y)
# plt.legend()
# plt.show()
