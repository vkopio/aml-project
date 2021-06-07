import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn import svm
from sklearn.model_selection import GridSearchCV

from common import train_data_bow as train_data, train_target, cv, make_test_prediction

svm_model = svm.SVC(C=1)

#cv_score = cv(svm_model, train_data, train_target)

# param_test1 = {'C': range(1,10,1),}

# gsearch1 = GridSearchCV(
#     estimator=svm_model,
#     param_grid=param_test1,
#     scoring='accuracy',
#     n_jobs=4,
#     cv=5
# )

# gsearch1.fit(train_data, train_target)

# print(gsearch1.cv_results_)
# print(gsearch1.best_params_)
# print(gsearch1.best_score_)

svm_model.fit(train_data, train_target)
make_test_prediction(svm_model, 'bow_svm', bow=True)
