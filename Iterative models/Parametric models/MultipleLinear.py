"""
Finds the best numeric features for multiple linear regession, with 2 inputs
All possible models with two features are checked
The best model is selected using cross fold validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import tools
import itertools


def findsubsets(set, size):
    return list(itertools.combinations(set, size))


# load train data
train_data = pd.read_csv("kc_house_train_data.csv")
# get all numeric features(excluding price!)
all_features = train_data.columns
numeric_features = [feature for feature in all_features
                    if tools.is_numeric(train_data[feature]) ]
numeric_features.remove("price")

# find all subsets of the numberic features of size 2
possible_features = findsubsets(numeric_features, 2)
# each subset must be a list for future use
possible_features = (list(x) for x in possible_features)


best_RSS = 0
sqft_model = LinearRegression()
for features in possible_features:
    print(features, end = " .")
    X = train_data[features]
    y = train_data["price"]
    sqft_model.fit(X, y)
    cross_RSS = tools.linear_k_fold_cross_validation(10, train_data, features, "price")
    print("RSS is:", np.round(cross_RSS,3))
    if cross_RSS > best_RSS:
        best_RSS = cross_RSS
        best_features = features

print("\n\nThe best feature is:", best_features)


## get test data r sq ##
#fit model used train data
train_X = train_data[best_features]
train_y = train_data["price"]
model = LinearRegression()
model.fit(train_X,train_y)
print(model.intercept_, model.coef_)
#load test data
test_data = pd.read_csv("kc_house_test_data.csv")
#test data using test data
test_X = test_data[best_features]
test_y = test_data["price"]
test_r_sq = model.score(test_X,test_y)
print("Test r squared is:", np.round(test_r_sq,4))





