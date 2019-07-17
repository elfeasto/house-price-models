import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import tools
from project_tools import *

def preprocess_data(train, test):
    min_year_built = min(train["yr_built"].min(), test["yr_built"].min())
    min_year_renovated = min(train["yr_renovated"].min(), test["yr_renovated"].min())

    train["yr_built"] = train["yr_built"] - min_year_built
    test["yr_built"] = test["yr_built"] - min_year_built

    train["yr_renovated"] = train["yr_renovated"] - min_year_renovated
    test["yr_renovated"] = test["yr_renovated"] - min_year_renovated

    return train, test


def show_regression_coeffs(features, model):
    print("Intercept is", model.intercept_)
    zipped = zip(features, model.coef_)
    for feat, coef in zipped:
        print(feat, np.round(coef, 3))


def cross_val_r_sq(train, features, l2_pen, k):
    my_sets = tools.train_valid_k_fold_sets(train, k)
    total_r_sq = 0
    for t, v in my_sets:
        r_model = Ridge(alpha=l2_pen, normalize=True)
        r_model.fit(t[features], t["price"])
        r_sq = r_model.score(v[features], v["price"])
        total_r_sq += r_sq
    avg_r_sq = total_r_sq / k
    return avg_r_sq


#load the data
train_data, test_data = get_train_test_data()
#train_data, test_data = preprocess_data(train_data, test_data)


#get a list of numeric features
numeric_features = []
for feature in train_data.columns:
    if tools.is_numeric(train_data[feature]):
        numeric_features.append(feature)
numeric_features.remove("price")
numeric_features.remove("id")
numeric_features.remove("zipcode")
numeric_features.remove("long")
numeric_features.remove("lat")
#numeric_features.remove("yr_built")

"""
find the best l2 penalty
done by finding r squared values for different l2 penalties
then select the l2 penalty that gives the highest r sqaured value
"""

#choose a value of k for k-fold cross validation
k = 20
# set consisting of (test,valid) pairs for k fold validation
my_sets = tools.train_valid_k_fold_sets(train_data,k)
#find the r squared values for different l2 penalties
#then select the l2 penalty that gives the highest r sqaured value
l2_pens = [0.001,0.01, 0.02,0.03,0.05, 0.1, 0.15]
for l2_pen in l2_pens:
    r_sq = cross_val_r_sq(train_data, numeric_features, l2_pen,k)
    #print("For l2 penalty {} the r sqaured is {}".format(l2_pen, r_sq))

best_model = Ridge(alpha=0.02, normalize=True)
best_model.fit(train_data[numeric_features], train_data["price"])
test_r_sq = best_model.score(test_data[numeric_features], test_data["price"])
show_regression_coeffs(numeric_features, best_model)

