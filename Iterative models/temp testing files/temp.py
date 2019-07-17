import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
import tools
from project_tools import *


def data_info(train, test):

    min_year_built = min(train["yr_built"].min(), test["yr_built"].min())

    f  = lambda x: 10000 if x==0 else x
    train_data["yr_renovated"] = train_data["yr_renovated"].map(f)
    print(train_data["yr_renovated"])
    min_year_renovated = min(train["yr_renovated"].min(), test["yr_renovated"].min())

    print("min year built:", min_year_built)
    print("min_year_renovated", min_year_renovated)



    train["yr_built"] = train["yr_built"] - min_year_built
    test["yr_built"] = test["yr_built"] - min_year_built


    train["yr_renovated"] = train["yr_renovated"] - min_year_renovated
    test["yr_renovated"] = test["yr_renovated"] - min_year_renovated





    return train, test

def preprocess_data(train, test):
    min_year_built = min(train["yr_built"].min(), test["yr_built"].min())
    min_year_renovated = min(train["yr_renovated"].min(), test["yr_renovated"].min())

    train["yr_built"] = train["yr_built"] - min_year_built
    test["yr_built"] = test["yr_built"] - min_year_built

    train["yr_renovated"] = train["yr_renovated"] - min_year_renovated
    test["yr_renovated"] = test["yr_renovated"] - min_year_renovated

    return train, test


def show_regression_coeffs(features, model):
    print("Intercept is", int(np.round(model.intercept_)))
    zipped = zip(features, model.coef_)
    for feat, coef in zipped:
        print(feat, int(np.round(coef)))


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


def regression_coeffs_series(features, model):
    coeffs_dict = dict()
    coeffs_dict["Intercept"] = int(np.round(model.intercept_))
    zipped = zip(features, model.coef_)
    for feat, coef in zipped:
        print(feat, int(np.round(coef)))
        coeffs_dict[feat] = int(np.round(coef))
    coeffs_series = pd.Series(coeffs_dict)
    print(coeffs_series)

#load the data
train_data, test_data = get_train_test_data()
#train_data, test_data = preprocess_data(train_data, test_data)

data_info(train_data, test_data)
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
numeric_features.remove("sqft_living")
numeric_features.remove("sqft_living15")
numeric_features.remove("sqft_lot15")
numeric_features.remove("yr_built")
numeric_features.remove("yr_renovated")

"""
find the best l2 penalty
done by finding r squared values for different l2 penalties
then select the l2 penalty that gives the highest r sqaured value
"""


train_x = train_data[numeric_features]
train_y = train_data["price"]

linear_model = LinearRegression()
linear_model.fit(train_x, train_y)
print("Features for Linear Regression:")
show_regression_coeffs(numeric_features, linear_model)

predictions = pd.Series(linear_model.predict(train_x))

print(predictions.head())
