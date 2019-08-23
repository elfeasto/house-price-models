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

def find_prediction(train_d, house_sqft, max_dist):
    train_d["distance"] = np.abs(train_d["sqft_living"] - house_sqft)
    within_distance = train_d[train_data["distance"] <= max_dist]
    estimated_price = within_distance["price"].mean()
    return estimated_price


def get_r_sq(training, testing, dist):
    f = lambda x: find_prediction(training, x, dist)
    testing["predictions"] = testing["sqft_living"].map(f)
    r_sq = tools.r_squared(testing, "price", testing["predictions"])
    return r_sq


def get_valid_r_sq(train, valid, max_dist):
    f = lambda x: find_prediction(train, x, max_dist)
    valid["predictions"] = valid["sqft_living"].map(f)
    r_sq = tools.r_squared(valid, "price", valid["predictions"])
    print( "For {} distance r squared is {}".format( max_dist, np.round( r_sq,3) ) )


def get_best_lambda(train, valid):
    ## TO DO
    ###############################################
    # check max distance between sqft in valid set#
    # and sqft in training set                    #
    ###############################################
    dists = range(45, 4, -5)
    for dist in dists:
        r_sq = get_r_sq(train, valid, dist)
        print("For distance {} r squared is {}".format(dist, r_sq))

    """
    For distance 100 r squared is 0.5217
    For distance 80 r squared is  0.5272
    For distance 60 r squared is  0.5278
    For distance 55 r squared is  0.5287
    For distance 50 r squared is  0.5283
    For distance 45 r squared is  0.53295
    For distance 40 r squared is 0.5330066420898254
    For distance 35 r squared is 0.5379192340146091
    For distance 30 r squared is 0.5374728684322179
    For distance 25 r squared is 0.5405449178740126
    For distance 20 r squared is 0.5388984092932876
    For distance 15 r squared is 0.5416901839196278
    For distance 10 r squared is 0.5423196790471603
    For distance 5 r squared is 0.5724923356625871
    """


def house_distance(house1, house2):
    return np.abs(house1["sqft_living"] - house2["sqft_living"])


def house_data_nearest_house(house, data):
    distances = np.abs(house["sqft_living"] - data["sqft_living"])
    return distances.min()


def data_to_data_max_dist(data1, data2):
    """
    return the max sqft_living between any  houses in data1 and
     all the houses in data2
    :param data1:
    :param data2:
    :return:
    """
    f = lambda house : house_data_nearest_house(house, data2)
    nearest_house = data1.apply(func=f, axis=1)
    max_min_dist = nearest_house.max()
    return max_min_dist


def min_dist_usable_valid_data():
    # want to find the house with the property that
    # there is no house within dist of it
    # each house will have a "data1, house with dist"
    # so we want the largest
    """
    If there is no training house with the distance lambda of a house
    in the testing data then there we cannot calculate an estimate
    :return:
    """
    ## below is the max distance usable if we do the
    ## train,valid and test split
    ## instead using cross validation to calculate lambda
    ## would let the valid data to be merged with the train
    ## data
    ## this could lead to a smaller possible max dist usable
    train_data, valid_data, test_data = get_train_valid_test_data()

    ans = data_to_data_max_dist(test_data, train_data)
    print("If the fixed validation set is used the minimum value we can choose"
          " for distance(lambada) is", ans)

    return ans


def min_dist_usable_no_valid_data():
    # want to find the house with the property that
    # there is no house within dist of it
    # each house will have a "data1, house with dist"
    # so we want the largest
    """
    If there is no training house with the distance lambda of a house
    in the testing data then there we cannot calculate an estimate
    :return:
    """
    ## below is the max distance usable if we do the
    ## train,valid and test split
    ## instead using cross validation to calculate lambda
    ## would let the valid data to be merged with the train
    ## data
    ## this could lead to a smaller possible max dist usable

    train_data, test_data = get_train_test_data()
    ans = data_to_data_max_dist(test_data, train_data)
    print("If no validation set is used the minimum value we can use for"
          " distance(lambda) is", ans)
    return ans



train_data, valid_data, test_data = get_train_valid_test_data()

min_dist_usable_valid_data()
min_dist_usable_no_valid_data()

