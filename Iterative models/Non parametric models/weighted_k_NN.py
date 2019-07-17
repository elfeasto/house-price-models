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


def house_prediction(train_d, house_sqft, k):
    # need to find k nearest houses
    # calculate distance between this house and all houses in train_d
    train_d["distance"] = np.abs(train_d["sqft_living"] - house_sqft)
    # now we can get these nearest houses
    k_nearest_houses = train_d.nsmallest(k, "distance")
    # using these houses and their distance, weights are calculated
    k_nearest_houses["weight"] = k_nearest_houses["distance"].map(lambda x: (1 / (x + 1)))
    # now the estimated price is calculated

    # zipped list of price and weights for convenience
    price_and_weights = zip(k_nearest_houses["price"], k_nearest_houses["weight"])
    estimated_price = 0
    for price, weight in price_and_weights:
        estimated_price += price * weight
    estimated_price /= k_nearest_houses["weight"].sum()
    return estimated_price


def data_predictions(training, testing, lam):
    """
    Get weighted k_NN predictions for a testing set based on
    the training set
    :param training:
    :param testing:
    :return:
    """
    f = lambda sqft: house_prediction(training, sqft, lam)
    predictions = testing["sqft_living"].map(f)
    return predictions


def CFV_r_sq(data, lam, k=10):
    """
    Does (k = 10) fold cross validation of the data
    :param training:
    :return:
    """
    my_sets = tools.train_valid_k_fold_sets(data, k)
    total_r_sq = 0
    for t, v in my_sets:
        v_predictions = data_predictions(t, v, lam)
        v_r_sq = tools.r_squared(v, "price", v_predictions)
        total_r_sq += v_r_sq
    avg_r_sq = total_r_sq / k
    return avg_r_sq


def find_best_lambda(training):
    """
    checks the CFV r squared score for various values
    of lambda and returns the best one
    :param training:
    :return:
    """
    possible_ks = [ 200]
    for k in possible_ks:
        r_sq = CFV_r_sq(training, k)
        print("For k = {}, the value of r sqaured is {}".format(l, np.round(r_sq, 4)))
    """
    For k = 1, the value of r sqaured is   0.1043
    For k = 5, the value of r sqaured is   0.4272
    For k = 10, the value of r sqaured is  0.4697
    For k = 30, the value of r sqaured is  0.496
    For k = 50, the value of r sqaured is  0.5008
    For k = 75, the value of r sqaured is  0.5013
    For k = 100, the value of r sqaured is 0.5013
    """


train_data, test_data = get_train_test_data()
train_data  = train_data


find_best_lambda(train_data)
