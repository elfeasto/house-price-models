import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import tools
import load_data


def preprocess_data(train, test):
    min_year_built = min(train["yr_built"].min(), test["yr_built"].min())
    min_year_renovated = min(train["yr_renovated"].min(), test["yr_renovated"].min())

    train["yr_built"] = train["yr_built"] - min_year_built
    test["yr_built"] = test["yr_built"] - min_year_built

    train["yr_renovated"] = train["yr_renovated"] - min_year_renovated
    test["yr_renovated"] = test["yr_renovated"] - min_year_renovated

    return train, test



#load the data
train_data, test_data = load_data.get_train_test_data()
#train_data, test_data = preprocess_data(train_data, test_data)
print(len(train_data))
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


#find the best l2 penalty
#choose a value of k
k = 20
# set consisting of (test,valid) pairs for k fold validation
my_sets = tools.train_valid_k_fold_sets(train_data,k)
my_sets = [(train_data[:14000], train_data[14000:])]
l2_pen = 0.1
total_r_sq = 0

r_model = Ridge(alpha=l2_pen, normalize=True)
r_model.fit(train_data[numeric_features], train_data["price"])


temp= zip(numeric_features, r_model.coef_)
for feat, coef in temp:
    print(feat, np.round(coef,3))