"""
ridge regression with adding in dummy variables for zipcode
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import tools
from project_tools import *


def preprocess_data(train, test):
    train = add_yr_built_dummies(train)
    train = add_zipcode_dummies(train)
    test = add_yr_built_dummies(test)
    test = add_zipcode_dummies(test)

    return train, test


def show_regression_coeffs(model, features):
    print("Intercept is", model.intercept_)
    zipped = zip(features, model.coef_)
    for feat, coef in zipped:
        print(feat, np.round(coef, 3))


def ridge_CFV(train, features, l2_pen, k):
    my_sets = tools.train_valid_k_fold_sets(train, k)
    total_r_sq = 0
    for t, v in my_sets:
        r_model = Ridge(alpha=l2_pen, normalize=True)
        r_model.fit(t[features], t["price"])
        r_sq = r_model.score(v[features], v["price"])
        total_r_sq += r_sq
    avg_r_sq = total_r_sq / k
    return avg_r_sq


def add_yr_built_dummies(data):
    def f(x):
        if x < 1950:
            return 0
        elif x < 1975:
            return 1
        elif x < 1997:
            return 2
        else:
            return 3

    data['yr_built'] = data['yr_built'].map(f)

    yr_built_dummies = pd.get_dummies(data['yr_built'], prefix='yr_built_cat')
    data = pd.concat([data, yr_built_dummies], axis=1)
    data.drop(columns='yr_built', inplace=True)
    return  data


def add_zipcode_dummies(data):
    zipcode_dummies = pd.get_dummies(data['zipcode'], prefix='zipcode')
    data = pd.concat([data, zipcode_dummies], axis=1)
    data.drop(columns = 'zipcode', inplace = True)
    return data

# load the data
train_data, test_data = get_train_test_data()


print( train_data['zipcode'].value_counts() )
