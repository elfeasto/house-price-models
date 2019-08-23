"""
performs k nearest neighbour method on the houses using sqft_living as the distance
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
import tools
from project_tools import *
from sklearn.model_selection import train_test_split


def house_prediction(train_d, house_sqft, k):
    train_d["distance"] = np.abs(train_d["sqft_living"] - house_sqft)
    k_nearest_houses = train_d.nsmallest(k,"distance")
    estimated_price = k_nearest_houses["price"].mean()
    return estimated_price


def data_prediction(train_d, testing_d, k):
    f = lambda x: house_prediction(train_d, x, k)
    predictions = testing_d["sqft_living"].map(f)
    return predictions



# load the data
train_data, test_data = get_train_test_data()

# use 10 fold validation to find the best value for k
num_splits = 10

for num_nn in range(50,101,10):
    cross_r_sq = tools.general_CFV(train_data, data_prediction,"price", num_nn, num_splits)
    print("For {}-nn CFV r squared is {}".format(num_nn, np.round(cross_r_sq,5)))

"""
For 10-nn CFV r squared is 0.48752
For 20-nn CFV r squared is 0.51004
For 30-nn CFV r squared is 0.51277
For 40-nn CFV r squared is 0.51574
For 50-nn CFV r squared is 0.51595
For 60-nn CFV r squared is 0.51425
For 70-nn CFV r squared is 0.51468
For 80-nn CFV r squared is 0.51354
For 90-nn CFV r squared is 0.514
For 100-nn CFV r squared is 0.51297
"""