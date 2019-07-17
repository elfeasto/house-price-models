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

def find_prediction(train_d, house_sqft, k):
    train_d["distance"] = np.abs(train_d["sqft_living"] - house_sqft)

    k_nearest_houses = train_d.nsmallest(k,"distance")
    estimated_price = k_nearest_houses["price"].mean()

    return estimated_price

def find_k_nn_r_sq(train_d, testing_d, k):
    f = lambda x: find_prediction(train_d, x, k)
    predictions = testing_d["sqft_living"].map(f)
    r_sq = tools.r_squared(testing_d, "price", predictions)
    return r_sq


def k_cross_valid_r_sq(train_d,  k):
    """
    use cross fold validation to find the best value of k
    :param train_d:
    :param house_sqft:
    :param k:
    :return:
    """

    my_sets = tools.train_valid_k_fold_sets(train_d, k)
    total_r_sq = 0
    for t,v in my_sets:
        r_sq = find_k_nn_r_sq(t,v,k)
        total_r_sq += r_sq
    avg_r_sq = total_r_sq/k
    return avg_r_sq


def find_best_k(t_data):
    # for k in (25,35):
    #     cross_r_sq = k_cross_valid_r_sq(t_data, k)
    #     print("Cross r squared for k = {} is {}".format(k, cross_r_sq))

    """
    Cross r squared for k = 10 is 0.4875
    Cross r squared for k = 20 is 0.509
    Cross r squared for k = 25 is 0.51336
    Cross r squared for k = 30 is 0.5108661093548654
    Cross r squared for k = 35 is 0.5120461847923536
    Cross r squared for k = 40 is 0.5065457215628785
    """
    #from results above best k is 35
    return 35


train_data, test_data = get_train_test_data()
# use cross fold validation to find the best value for k
k = find_best_k(train_data)
# get test predictions for best k
f = lambda x: find_prediction(train_data, x, k)
test_data["predictions"] = test_data["sqft_living"].map(f)
# get r squared for this value of k
test_r_sq = tools.r_squared(test_data,"price", test_data["predictions"])
print("Test r squared is", test_r_sq)