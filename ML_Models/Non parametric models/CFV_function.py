import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
import tools
from project_tools import *
from sklearn.model_selection import train_test_split

# do CFV for any given model


def general_CFV(data, prediction_function, target_column, param, k=10):
    """

    :param data: data to perform CFV on
    :param prediction_function: prediction_function: maps (training data, testing data, k)
           to a prediction series of testing data
    :param target_column: column of df that is being predicted
    :param param: parameter of the model being used
    :param k: k to be used in k fold CFV
    :return: CFV r squared
    """
    train_valid_splits = tools.train_valid_k_fold_sets(data, k)

    total_r_sq = 0
    # iterate through each (train,valid) tuple finding r squared for each
    # add this r squared to the running total r squared
    for train_set, valid_set in train_valid_splits:
        # find predictions of valid_set based on train_set
        valid_set_predictions = prediction_function(train_set, valid_set, param)
        valid_set_r_sq = tools.r_squared(valid_set, target_column,
                                         valid_set_predictions)
        total_r_sq += valid_set_r_sq

    # find the average r squared over the different validation sets
    avg_r_sq = total_r_sq / len(train_valid_splits)
    return avg_r_sq
