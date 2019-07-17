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

## find best feature using cross validation ##
# load train data
train_data = pd.read_csv("kc_house_train_data.csv")

# get all numeric features(excluding price!)
all_features = train_data.columns
numeric_features = [feature for feature in all_features
                    if tools.is_numeric(train_data[feature]) ]
numeric_features.remove("price")
# test which one is best with k fold validtaion(k=10 for now)
best_feature = ""
best_RSS = 0
sqft_model = LinearRegression()
for feature in numeric_features:
    print(feature, end = " .")
    X = train_data[[feature]]
    y = train_data["price"]
    sqft_model.fit(X, y)
    cross_RSS = tools.linear_k_fold_cross_validation(10, train_data, [feature], "price")
    print("RSS is:", np.round(cross_RSS,3))
    if cross_RSS > best_RSS:
        best_RSS = cross_RSS
        best_feature = feature

print("\n\nThe best feature is:", best_feature)


## get test data r sq ##
#fit model used train data
train_X = train_data[["sqft_living"]]
train_y = train_data["price"]
model = LinearRegression()
model.fit(train_X,train_y)
print(model.intercept_, model.coef_)
#load test data
test_data = pd.read_csv("kc_house_test_data.csv")
#test data using test data
test_X = test_data[["sqft_living"]]
test_y = test_data["price"]
test_r_sq = model.score(test_X,test_y)
print("Test r squared is:", np.round(test_r_sq,4))





