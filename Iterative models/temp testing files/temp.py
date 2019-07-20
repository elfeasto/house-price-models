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


pts = dict()
pts[1] = 1
pts[2] = 4
pts[3] = 8
print(pts.keys())

print(pts.values())
plt.plot(pts.keys(), pts.values())
plt.show()