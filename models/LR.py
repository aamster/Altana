import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV


class LR:
    def hyperparam_tuning(self, X, y):
        model = LogisticRegression(penalty='elasticnet')
        distributions = dict(C=uniform(loc=0, scale=4), l1_ratio=uniform(), class_weight=['balanced', None])
        clf = RandomizedSearchCV(model, distributions)
        search = clf.fit(X, y)
        return pd.DataFrame(search.cv_results_)