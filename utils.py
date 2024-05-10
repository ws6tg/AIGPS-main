from itertools import combinations


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from collections import defaultdict
from xgboost import XGBClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

from tqdm import tqdm


def dftopair(df_data, idx1, idx2, std=None, a=0):
    g1 = df_data[idx1].values
    g2 = df_data[idx2].values
    # diff=|D(gene1)+D(gene2)|**(1/2)
    if std is not None:
        diff = a * (np.absolute((std[idx1].values ** 2 + std[idx2].values ** 2)) ** (1 / 2))
    else:
        diff=0
    sub1 = g1 - g2 > diff
    sub2 = -1 * (g1 - g2 < -diff)
    sub = sub1 + sub2
    return sub



def get_genepairs(matrix, types, batch_size, a):

    std = matrix.std(axis=0)
    pairs0 = [i[0] for i in combinations(matrix.columns, 2)]
    pairs1 = [i[1] for i in combinations(matrix.columns, 2)]


    num_subarrays = len(pairs0) // batch_size + (1 if len(pairs1) % batch_size > 0 else 0)
    subgene0 = [pairs0[i * batch_size:(i + 1) * batch_size] for i in range(num_subarrays)]
    subgene1 = [pairs1[i * batch_size:(i + 1) * batch_size] for i in range(num_subarrays)]
    df_pairs = pd.DataFrame()

    groups = list(set(types))


    with tqdm(total=len(pairs0)) as pbar:
        for i, j in zip(subgene0, subgene1):

            df_pairs1 = pd.DataFrame()
            df_pairs1["pairs0"] = i
            df_pairs1["pairs1"] = j

            for k in groups:

                x = dftopair(matrix[types == k], i, j, std, a)

                df_pairs1[f'{k}_1'] = [np.sum(i == 1) for i in x.T]
                df_pairs1[f'{k}_0'] = [np.sum(i == 0) for i in x.T]
                df_pairs1[f'{k}_-1'] = [np.sum(i == -1) for i in x.T]

            df_pairs = pd.concat([df_pairs, df_pairs1])
            pbar.update(len(i))

    return df_pairs


def test_by_ml(kf_splits, random, clfs_name, x_train, y_train, test_data):
    CV_result = defaultdict(dict)
    import warnings
    warnings.filterwarnings('ignore')
    kf = StratifiedKFold(shuffle=True, random_state=random, n_splits=kf_splits)

    param_test1 = {'n_estimators': [120, 300],
                   'criterion': ['log_loss', 'gini'],
                   'max_depth': [3, 20],
                   'min_samples_split': [2],
                   'min_samples_leaf': [7, 8, 9],
                   'max_features': [7, 9]
                   }
    gsearch1 = GridSearchCV(estimator=RandomForestClassifier(random_state=random),
                            param_grid=param_test1, scoring='roc_auc', cv=kf, n_jobs=50)

    param_test2 = {'n_neighbors': [10, 15, 20, 25, 30, 32],
                   'weights': ['distance', 'uniform'],
                   'p': [1, 2]}
    gsearch2 = GridSearchCV(estimator=KNeighborsClassifier(),
                            param_grid=param_test2, scoring='roc_auc', cv=kf, n_jobs=50)

    param_test3 = {'solver': ['liblinear', 'newton-cg', 'sag', 'saga', 'lbfgs'],
                   'max_iter': [10, 20, 50, 100, 300],
                   'C': [0.5, 1, 2]}
    gsearch3 = GridSearchCV(estimator=LogisticRegression(random_state=random),
                            param_grid=param_test3, scoring='roc_auc', cv=3, n_jobs=50)

    param_test4 = {'kernel': ['sigmoid', 'poly', 'linear', 'rbf'],
                   'max_iter': [10, 20, 50, 100, 300],
                   'C': [0.5, 1, 2]}
    gsearch4 = GridSearchCV(estimator=svm.SVC(probability=True, random_state=random),
                            param_grid=param_test4, scoring='roc_auc', cv=kf, n_jobs=50)

    param_test5 = {
        "activation": ['identity', 'logistic', 'tanh', 'relu'],
        "hidden_layer_sizes": [30, 60, 100, 200],
        "solver": ["adam", "sgd", "lgbfs"],
        "max_iter": [500, 1000],
        "learning_rate_init": [0.005, 0.001, 0.01, 5e-4, 1e-4, 5e-5]
    }
    gsearch5 = GridSearchCV(estimator=MLPClassifier(random_state=random),
                            param_grid=param_test5, scoring='roc_auc', cv=kf, n_jobs=50)

    param_test6 = {
        "booster": ["gbtree", "gblinear"],
        'n_estimators': [40, 80, 120],
        'max_depth': [3, 5, 7, 9],
        'min_child_weight': [1, 3, 5, 7, 9]
    }
    gsearch6 = GridSearchCV(estimator=XGBClassifier(device="cuda", verbosity=0, random_state=random),
                            param_grid=param_test6, scoring='roc_auc', cv=kf, n_jobs=50)

    param_test7 = {
        "alpha": [1.0, 0.9, 0.8, 0.7],
        "fit_prior": [True, False]
    }
    gsearch7 = GridSearchCV(estimator=BernoulliNB(),
                            param_grid=param_test7, scoring='roc_auc', cv=kf, n_jobs=50)

    method_dict = {
        "RF": gsearch1,
        "KNN": gsearch2,
        "LR": gsearch3,
        "SVC": gsearch4,
        "MLP": gsearch5,
        "XGB": gsearch6,
        "NB": gsearch7,
    }
    for clf_name, clf in [(i, method_dict[i]) for i in clfs_name]:
        clf.fit(x_train, y_train)
        CV_result[clf_name]["auc"] = "|".join(
            [str(round(clf.score(x, y), 3)) for x, y in test_data])

    return [method_dict[i] for i in clfs_name], pd.DataFrame(CV_result)