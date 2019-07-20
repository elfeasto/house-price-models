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


def k_nn_r_sq(train_d, testing_d, k):
    f = lambda x: house_prediction(train_d, x, k)
    predictions = testing_d["sqft_living"].map(f)
    r_sq = tools.r_squared(testing_d, "price", predictions)
    return r_sq


def CFV_r_sq(train_d, k):
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
        r_sq = k_nn_r_sq(t, v, k)
        total_r_sq += r_sq
    avg_r_sq = total_r_sq/k
    return avg_r_sq


def find_best_k(t_data):
    # for k in [15,45,50]:
    #     cross_r_sq = CFV_r_sq(t_data, k)
    #     rounded_r_sq = np.round(cross_r_sq, 5)
    #     print( "Cross r squared for k = {} is {}".format(k, rounded_r_sq) )

    """
    Cross r squared for k = 10 is 0.4875
    Cross r squared for k = 15 is 0.50381
    Cross r squared for k = 20 is 0.50907
    Cross r squared for k = 21 is 0.50657
    Cross r squared for k = 22 is 0.50866
    Cross r squared for k = 23 is 0.50804
    Cross r squared for k = 24 is 0.50596
    Cross r squared for k = 25 is 0.51336
    Cross r squared for k = 26 is 0.51173
    Cross r squared for k = 27 is 0.5117
    Cross r squared for k = 28 is 0.5076
    Cross r squared for k = 29 is 0.50454
    Cross r squared for k = 30 is 0.51087
    Cross r squared for k = 31 is 0.51126
    Cross r squared for k = 32 is 0.5115
    Cross r squared for k = 33 is 0.51092
    Cross r squared for k = 34 is 0.50867
    Cross r squared for k = 35 is 0.51205
    Cross r squared for k = 36 is 0.50924
    Cross r squared for k = 37 is 0.51012
    Cross r squared for k = 38 is 0.51032
    Cross r squared for k = 39 is 0.50806
    Cross r squared for k = 40 is 0.50655
    Cross r squared for k = 45 is 0.50946
    Cross r squared for k = 50 is 0.51041
    """
    # from results above best k is 35
    return 25


def graph_CFV_results():
    pts = dict()

    pts[15] = 0.50381
    pts[10] = 0.4875
    pts[20] = 0.50907
    pts[21] = 0.50657
    pts[22] = 0.50866
    pts[23] = 0.50804
    pts[24] = 0.50596
    pts[25] = 0.51336
    pts[26] = 0.51173
    pts[27] = 0.5117
    pts[28] = 0.5076
    pts[29] = 0.50454
    pts[30] = 0.51087
    pts[31] = 0.51126
    pts[32] = 0.5115
    pts[33] = 0.51092
    pts[34] = 0.50867
    pts[35] = 0.51205
    pts[36] = 0.50924
    pts[37] = 0.51012
    pts[38] = 0.51032
    pts[39] = 0.50806
    pts[40] = 0.50655
    pts[45] = 0.50946
    pts[50] = 0.51041
    plt.scatter(pts.keys(), pts.values())
    plt.show()



train_data, test_data = get_train_test_data()
# use cross fold validation to find the best value for k
k = find_best_k(train_data)
# get test predictions for best k
f = lambda x: house_prediction(train_data, x, k)
test_data["predictions"] = test_data["sqft_living"].map(f)
# get r squared for this value of k
test_r_sq = tools.r_squared(test_data,"price", test_data["predictions"])
print("Test r squared is", test_r_sq)

#graph of k CFV scores
graph_CFV_results()