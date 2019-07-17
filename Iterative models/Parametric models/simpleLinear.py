"""
Simple linear regession using one feature(sqft_living)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import tools
from project_tools import *

# load train data
train_data, test_data = get_train_test_data()

# pick feature
features = ["sqft_living"]
# fit our model
sqft_model = LinearRegression()
X = train_data[features]
y = train_data["price"]
sqft_model.fit(X, y)
# get test data r sq
X = tools.get_features_matrix(test_data, features)
y = test_data["price"]
test_r_sq = sqft_model.score(X,y)
print("Test r squared is:", np.round(test_r_sq,4))



