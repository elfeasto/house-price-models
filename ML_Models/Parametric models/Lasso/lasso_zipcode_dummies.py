"""
Lassoo regression with adding in dummy variables for zipcode
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
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


def show_regression_coeffs(model, features):
    print("Intercept is", model.intercept_)
    zipped = zip(features, model.coef_)
    for feat, coef in zipped:
        print(feat, np.round(coef, 3))


def lasso_CFV(train, features, l2_pen, k):
    my_sets = tools.train_valid_k_fold_sets(train, k)
    total_r_sq = 0
    for t, v in my_sets:
        r_model = Lasso(alpha=l2_pen, normalize=True, max_iter=MAX_ITERATIONS)
        r_model.fit(t[features], t["price"])
        r_sq = r_model.score(v[features], v["price"])
        total_r_sq += r_sq
    avg_r_sq = total_r_sq / k
    return avg_r_sq

# max iterations to get lasso to converge
MAX_ITERATIONS = 10**6

#load the data
train_data, test_data = get_train_test_data()
# add in dummy variable columns for the zipcodes
train_data_dummies = pd.get_dummies(train_data['zipcode'])
train_data = pd.concat([train_data, train_data_dummies], axis=1)
test_data_dummies = pd.get_dummies(test_data['zipcode'])
test_data = pd.concat([test_data, test_data_dummies], axis=1)


#get a list of numeric features
numeric_features = []
for feature in train_data.columns:
    if tools.is_numeric(train_data[feature]):
        numeric_features.append(feature)
# remove irrelevant numeric features
numeric_features.remove("price")
numeric_features.remove("id")
numeric_features.remove("zipcode")
numeric_features.remove("long")
numeric_features.remove("lat")
#numeric_features.remove("yr_built")

features = numeric_features

"""
find the best l2 penalty
done by finding r squared values for different l2 penalties
then select the l2 penalty that gives the highest r squared value
"""

# choose a value of k for k-fold cross validation
k = 50
# set consisting of (test,valid) pairs for k fold validation
my_sets = tools.train_valid_k_fold_sets(train_data,k)
# find the r squared values for different l2 penalties
# then select the l2 penalty that gives the highest r squared value
l2_pens = [10, 40, 100]
best_l2_value = None
best_cross_r_sq = -np.inf
for l2_pen in l2_pens:
    cross_r_sq = lasso_CFV(train_data, features, l2_pen,k)
    print("For l2 penalty {}, r squared is {}".format(l2_pen, np.round( cross_r_sq, 5)))
    if cross_r_sq > best_cross_r_sq:
        best_cross_r_sq = cross_r_sq
        best_l2_value = l2_pen
print()
print("The best value for l2 is", best_l2_value)
best_model = Lasso(alpha=0.03, normalize=True, max_iter=MAX_ITERATIONS)
best_model.fit(train_data[features], train_data["price"])
# show_regression_coeffs(numeric_features, best_model)

# find test r squared
test_r_sq = best_model.score(test_data[features], test_data["price"])
print()
print("Test r squared is", np.round(test_r_sq, 5))
print()
print("Features used are", features)

#show_regression_coeffs(best_model, features)


