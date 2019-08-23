"""
Gaussian kernel regression, lambda found using cross validation(10 fold)
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tools
from project_tools import *


def gauss_kernel(d, slambda):
    """
    Kernel gives a positive weight to every point in the input space
    (though very small for far away ones)
    the higher lambda the slower the weights decay
    :param d:
    :param slambda:
    :return:
    """
    return np.exp(-(d**2) / slambda)


def gaussian_house_prediction(data, house_sqft, lam):
    """
    Get the estimate for one house using gaussian regression
    on the input data with the specified lambda
    :param data:
    :param house_sqft:
    :param lam:
    :return:
    """
    house_sqft = float(house_sqft)
    distances = np.abs(data["sqft_living"] - house_sqft)
    weights = distances.map(lambda d:gauss_kernel(d, lam))
    estimate = np.dot(data["price"], weights) / weights.sum()
    return estimate


def gauss_predictions(train, testing, lam):
    """
    Does gaussian kernel regression using a training set (train)
    and returns the predictions for the testing set
    :param train:
    :param testing:
    :param lam:
    :return:
    """
    f = lambda x: gaussian_house_prediction(train, x, lam)
    estimates = (testing["sqft_living"]).map( f )
    return estimates


def gauss_r_sq(train, test, lam):
    """
    Gets the r squared value for gaussian kernel regresion
    using training set(train) and test set (test)
    :param train:
    :param test:
    :param lam:
    :return:
    """
    test_predicted = gauss_predictions(train, test, lam)
    r_sq = tools.r_squared(test, "price", test_predicted)
    return r_sq


def cross_val_lambda(train, lam, k=10):
    """
    Return cross validation r squared
    For gaussian kernel regression
    :param train:
    :param lam:
    :param k:
    :return:
    """
    cross_valid_sets = tools.train_valid_k_fold_sets(train, k)
    total_r_sq = 0
    for t,v in cross_valid_sets:
        v_r_sq = gauss_r_sq(t,v,lam)
        total_r_sq += v_r_sq
    avg_r_sq= total_r_sq / k
    return avg_r_sq


def find_best_lambda(train, k=10):
    """
    Uses cross validation(k = 10) to find the best lambda
    from a given selection of possible values for lambda
    To save time the results are show below as text
    The best lambda is return automatically
    :param train:
    :return:
    """
    """
    lams = [3000, 4000,5000,6000, 8000]
    for lam in lams:
        r_sq = cross_val_lambda(train_data, lam)
        print(lam)
        print(r_sq)


    #cross val r sq for 3000 : 0.5071689782513956
    #cross val r sq for 4000 : 0.5075554640159469
    #cross val r sq for 5000 : 0.5076458571518281
    #cross val r sq for 6000 : 0.5076225153031769
    #cross val r sq for 8000 : 0.5075099473911580
    """

    return 5000

# load the data
train_data,test_data = get_train_test_data()
# get the best value for the parameter lambda
best_lam = find_best_lambda(train_data)
# evaluate r squared on the test data
test_r_sq = gauss_r_sq(train_data, test_data, best_lam)
print("Using Gaussian kernel regression with parameter set to 5000.")
print("This gives a test r squared of", np.round(test_r_sq,5))









