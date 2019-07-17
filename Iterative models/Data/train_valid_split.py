"""
Split training data into two for validation data
"""

import pandas as pd
from sklearn.model_selection import train_test_split

"""
Split training data into validation and leftovers
"""


####################################################
## load data and get prediction using sqft_living ##
####################################################

# load train data
train_data = pd.read_csv("kc_house_train_data.csv")
both_data = train_test_split(train_data, train_size = 0.75)
print(type(both_data))
train, valid = both_data
print(len(train))
print(len(valid))
print(type(train))
train.to_csv("training_less_valid.csv")
valid.to_csv("validation_data.csv")