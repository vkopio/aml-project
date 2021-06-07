import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier

from common import train_data_bow as train_data, train_target, cv, make_test_prediction

# x = np.linspace(0.6,1.2,1000)

# for n in [10, 100, 1000]:
#     model = AdaBoostClassifier(n_estimators=n, random_state=42)
#     cv_score = cv(model, train_data, train_target)

#     y = norm.pdf(x, loc=cv_score[0], scale=cv_score[1])
#     plt.plot(x, y, label='n = {}'.format(n))

# plt.legend()
# plt.show()

# for n in range(3, 4):
#     gboost = GradientBoostingClassifier(
#         n_estimators=1000,
#         learning_rate=0.1,
#         max_depth=n,
#         random_state=42)

#     cv_score = cv(gboost, train_data, train_target)

#gboost.fit(train_data, train_target)

#make_test_prediction(gboost, 'gradientboost')

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score, GridSearchCV

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


def modelfit(alg, performCV=True, printFeatureImportance=True, cv_folds=3):
    # Fit the algorithm on the data
    alg.fit(train_data, train_target)

    # Predict training set:
    dtrain_predictions = alg.predict(train_data)
    dtrain_predprob = alg.predict_proba(train_data)[:, 1]

    # Perform cross-validation:
    if performCV:
        cv_score = cross_val_score(
            alg, train_data, train_target, cv=cv_folds)

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(
        train_target, dtrain_predictions))
    print("AUC Score (Train): %f" %
          metrics.roc_auc_score(train_target, dtrain_predprob))

    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" %
              (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))

    # Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(
            alg.feature_importances_).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.show()


gbm = GradientBoostingClassifier(
    loss='exponential',
    learning_rate=0.02,
    n_estimators=700,
    min_samples_split=1400,
    min_samples_leaf=30,
    max_depth=11,
    max_features=3,
    subsample=0.8,
    random_state=42
)

gbm.fit(train_data, train_target)
make_test_prediction(gbm, 'bow_gradientboost', bow=True)

# param_test1 = {'n_estimators': range(700,3000,300), 'learning_rate': [0.01, 0.02]}

# gsearch1 = GridSearchCV(
#     estimator=gbm,
#     param_grid=param_test1,
#     scoring='accuracy',
#     n_jobs=4,
#     cv=5
# )

# gsearch1.fit(train_data, train_target)

# print(gsearch1.cv_results_)
# print(gsearch1.best_params_)
# print(gsearch1.best_score_)

#cv_score = cv(gbm, train_data, train_target)
