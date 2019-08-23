"""
ridge regression with adding in dummy variables for zipcode
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import tools
from project_tools import *
from sklearn.preprocessing import PolynomialFeatures

# load the data
train_data, test_data = get_train_test_data()


features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view',
             'grade','yr_built','zipcode']


train_data_matrix = PolynomialFeatures()
x_train = train_data_matrix.fit_transform(train_data[features])

polyfeat = PolynomialFeatures(degree=2)
X_trainpoly = polyfeat.fit_transform(train_data[features])

print(type(train_data_matrix))
poly = LinearRegression().fit(x_train, train_data['price'])