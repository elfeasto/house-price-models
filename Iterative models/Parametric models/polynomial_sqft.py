"""
Finds the best degree for polynomial linear regession with sqft_living feature
All polynomial models up to degree 15 iare included
The best model is selected using cross fold validation
"""


import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import tools
from project_tools import *


# highest power we consider:
MAXPOWER = 15
# load  data
train_data, test_data = get_train_test_data()

# add polynomial features to all data
train_data = tools.add_poly_feature(train_data, "sqft_living", MAXPOWER)
test_data = tools.add_poly_feature(test_data, "sqft_living", MAXPOWER)

# get poly cols
poly_cols = tools.get_poly_col_names(train_data, "sqft_living")

# find best deg poly using cross validation(10 fold)
deg_r_sqs = []
for deg in range(1, MAXPOWER):
    features = poly_cols[:deg]
    r_sq = tools.linear_k_fold_cross_validation(10,train_data,features, "price")
    deg_r_sqs.append(r_sq)

best_deg = np.argmax(deg_r_sqs) + 1
print("Using cross validation the best degree is", best_deg)


# find r_sq on test data with best model
# fit the model to best degree poly on training data
best_model = LinearRegression(normalize=True)
features = poly_cols[:best_deg]
X_train = train_data[features]
y_train = train_data["price"]
best_model.fit(X_train, y_train)
# get r squared on test data
X_test = test_data[features]
y_test = test_data["price"]
r_sq = best_model.score(X_test, y_test)
print("Test R squared with this model is:", np.round(r_sq,3))










