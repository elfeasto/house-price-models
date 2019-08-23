"""
Check to see if sqft_living is equal to sqft_above plus sqft_basement
"""
from project_tools import *
train_data,  test_data = get_train_test_data()

relevant = ["sqft_living", "sqft_above", "sqft_basement"]

train_data = train_data[relevant]
train_data["above_below"] = train_data["sqft_above"] + train_data["sqft_basement"]

equivalence = all(train_data["sqft_living"] == train_data["above_below"])
if equivalence:
    print("IN ALL CASES")
    print("Sqft_living = sqft_below + sqft_basement")
else:
    print("IN SOME CASES")
    print("Sqft_living != sqft_below + sqft_basement")