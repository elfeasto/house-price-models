"""
In this program the sqft area and sqft lot features are looked at

It can be seen that using multiple regression on both that sqft lot has
a negative coefficient

This is perhaps best explained by houses that have a small lot are in
the centre of the city and the premium on the location outweights that
of the lot size

Plotting a histogram of both sqft liv and sqft lot shows they to be left skewed(?)
using x -> log(x) we can normalise both
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import tools
from project_tools import *


def multiple_linear_results(train_data):
    living_lot_model = LinearRegression()
    X = train_data[["sqft_living", "sqft_lot"]]
    y = train_data["price"]
    living_lot_model.fit(X, y)
    living_coeff, lot_coeff = living_lot_model.coef_

    print("Intercept is", living_lot_model.intercept_)
    print("Sqft living coeff is ", living_coeff)
    print("sqft lot coeff is", lot_coeff)

def show_histograms():
    #do later
    pass


#load the data
train, test = get_train_test_data()
multiple_linear_results(train)
