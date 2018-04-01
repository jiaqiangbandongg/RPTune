#-*- coding:utf-8 -*-

from sklearn.model_selection import RandomizedSearchCV,cross_val_score,GridSearchCV
import six
import numpy as np
from time import time
import pandas as pd
from sklearn.utils import shuffle
import itertools
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.datasets import load_digits
from scipy.stats import randint as sp_randint
from sklearn.linear_model import LinearRegression

# Utility function to report best scores
# from : http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

class RPTune():
    def __init__(self,estimator,param_distributions,n_iter='auto',scoring=None,
                 n_jobs=1,cv=None,random_state=None):
        estimator.set_params(random_state=random_state)
        self.estimator = estimator
        self.param_distributions = param_distributions
        max_iter = 1
        for k,v in param_distributions.items():
            if hasattr(v,'rvs'):
                max_iter *= int(v.b-v.a+1)
            else:
                max_iter *= len(v)
        print('max_iter',max_iter)
        if isinstance(n_iter, six.string_types):
            if n_iter == "auto":
                n_iter = max(1, int(np.sqrt(max_iter)))
            elif n_iter == "sqrt":
                n_iter = max(1, int(np.sqrt(max_iter)))
            elif n_iter == "log2":
                n_iter = max(1, int(np.log2(max_iter)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')
        if n_iter > 0 and n_iter < 1:
            n_iter = max(1, int(n_iter*max_iter))
        self.n_iter = n_iter
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.cv = cv
        self.random_state = random_state
        self.random_search = RandomizedSearchCV(estimator,
                                                param_distributions=param_distributions,
                                                n_iter=n_iter,
                                                scoring=scoring,
                                                n_jobs=n_jobs,
                                                cv=cv,
                                                random_state=random_state)

    def _generator_dataset(self,max_distribution_sample=10):

        # generate train
        train = pd.DataFrame(self.random_search.cv_results_['params'])
        train['score'] = self.random_search.cv_results_['mean_test_score']

        # generate test
        param_grids = {}
        for k,v in self.param_distributions.items():
            if hasattr(v,'rvs'):
                param_grids[k] = shuffle(list(range(v.a,v.b+1)))[:max_distribution_sample] # distribution to exact value
            else:
                param_grids[k] = v
        test = []
        for params in itertools.product(*param_grids.values()):
            param_grid = {}
            for param_name, param_value in zip(param_grids.keys(), params):
                param_grid[param_name] = param_value
            test.append(param_grid)
        test = pd.DataFrame(test)

        return train,test

    def fit(self,X,y,n_top=3):

        start = time()
        self.random_search.fit(X,y)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start), self.n_iter))
        report(self.random_search.cv_results_)

        print("Generating Parameter DataSet")
        train, test = self._generator_dataset()

        print("Simple Pre-process")
        train_clean = train.copy()
        test_clean = test.copy()
        for column in test.columns:
            if test.dtypes[column] == 'object':
                t = pd.concat([train_clean[column],test_clean[column]],axis=0)
                train_clean = pd.concat([t[:len(train_clean)],train_clean],axis=1)
                train_clean.drop(column,axis=1,inplace=True)
                test_clean = pd.concat([t[len(train_clean):], test_clean], axis=1)
                test_clean.drop(column, axis=1, inplace=True)
        train_clean[train_clean.isnull()] = 0
        train_clean[train_clean.isnull()] = 0
        test_clean[test_clean.isnull()] = 0
        test_clean[test_clean.isnull()] = 0

        print("Train and Test")
        reg = LinearRegression()
        #reg = RandomForestRegressor()
        label = train_clean.pop('score')
        reg.fit(train_clean,label)
        pred = reg.predict(test_clean)
        idxes = np.argsort(pred)[::-1][:n_top]
        params = test.loc[idxes].to_dict('record')
        for i,(idx,param) in enumerate(zip(idxes,params)):
            # pandas can convert "None" to nan
            for k,v in param.items():
                if param[k] != param[k]: # is nan
                    param[k] = None
            self.estimator.set_params(**param)
            score_mean = cross_val_score(self.estimator,X,y,scoring=self.scoring,cv=self.cv).mean()
            print("Model with rank: {0}".format(i))
            print("Pred score: {0}".format(pred[idx]))
            print("Mean validation score: {0:.3f}".format(score_mean))
            print("Parameters: {0}".format(param))
            print("")

if __name__ == '__main__':
    # get some data
    digits = load_digits()
    X, y = digits.data, digits.target
    clf = RandomForestClassifier(n_estimators=20)
    param_dist = {"max_depth": sp_randint(1,11),
                  "max_features": sp_randint(1, 11),
                  "min_samples_split": sp_randint(2, 11),
                  "min_samples_leaf": sp_randint(1, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}
    rpt = RPTune(clf,param_distributions=param_dist,n_iter='auto',n_jobs=-1,random_state=42,scoring='accuracy')
    rpt.fit(X,y)

