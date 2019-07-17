"""
peforms one nearest neighbour method on the houses using sqft_living as the distance
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from ipywidgets.widgets.widget_bool import _Bool
from sklearn.linear_model import LinearRegression
import tools
from project_tools import *


def distance(house1,house2):
    return np.abs((house1["sqft_living"] - house2["sqft_living"]))

def find_prediction(train_d, house_sqft):
    train_d["distance"] = np.abs(train_d["sqft_living"] - house_sqft)
    min_idx = train_d["distance"].idxmin()
    estimated_price = (train_data.iloc[min_idx])["price"]
    return estimated_price


train_data, test_data = get_train_test_data()


test_data["predicted"] = test_data["sqft_living"].map(lambda x:find_prediction(train_data,x))

test_r_sq = tools.r_squared(test_data,"price", test_data["predicted"])

print("Test r squared is", test_r_sq)