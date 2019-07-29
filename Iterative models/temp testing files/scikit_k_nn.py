"""
performs weighted k nearest neighbour method on the houses using sqft_living as the distance
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
import tools
from project_tools import *
from sklearn.neighbors import KNeighborsRegressor


def k_nn_prediction(training, testing, k):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(training[["sqft_living"]], training["price"])
    return  knn.predict(testing[["sqft_living"]])


def weighted_k_nn_prediction(training, testing, k):
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
    knn.fit(training[["sqft_living"]], training["price"])
    return  knn.predict(testing[["sqft_living"]])


train_data, test_data = get_train_test_data()




for k in range(30,120,10):
    valid_r_sq = tools.general_CFV(train_data, k_nn_prediction, "price", k)
    print("For k = {} we get a CFV r squared of {}".format(k,valid_r_sq))

# k = 40 gives good results
k = 40
test_data["prediction"] = weighted_k_nn_prediction(train_data, test_data, k)
test_r_sq = tools.r_squared(test_data, "price", test_data["prediction"])
print("Test r squared for k = {} is {}".format(k , test_r_sq))