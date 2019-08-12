"""
Finds the best numeric feature for simple linear regression
checks models with one number feature
finds the best one using cross fold validation

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import tools
from project_tools import *

## find best feature using cross validation ##
# load train data
train_data, test_data = get_train_test_data()

# get all numeric features(excluding price!)
all_features = train_data.columns
numeric_features = [feature for feature in all_features
                    if tools.is_numeric(train_data[feature]) ]
numeric_features.remove("price")
# test which one is best with k fold validtaion(k=10 for now)
best_feature = ""
best_RSS = 0
linear_model = LinearRegression()
for feature in numeric_features:
    X = train_data[[feature]]
    y = train_data["price"]
    linear_model.fit(X, y)
    cross_RSS = tools.linear_k_fold_cross_validation(10, train_data, [feature], "price")
    print(feature, "r squared is:", np.round(cross_RSS,4))
    if cross_RSS > best_RSS:
        best_RSS = cross_RSS
        best_feature = feature

print("\nThe best feature is:\n", best_feature)

# get the linear model with the best feature
model = LinearRegression()
train_X = train_data[[best_feature]]
train_y = train_data["price"]
model.fit(train_X,train_y)


# find test r squared
test_X = test_data[[best_feature]]
test_y = test_data["price"]
test_r_sq = model.score(test_X, test_y)
print("Test r squared with this feature is:", np.round(test_r_sq,4))





