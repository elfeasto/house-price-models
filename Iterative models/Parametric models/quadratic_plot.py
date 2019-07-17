import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import tools


def poly_pt(xvalue, model):
    intercept = model.intercept_
    coeffs = model.coef_
    ans = intercept
    for n in range(len(coeffs)):
        ans += (xvalue ** (n + 1)) * coeffs[n]
    return ans


#load train data
train_data = pd.read_csv("kc_house_train_data.csv")
train_data.sort_values(by = ["sqft_living"])

# add sqft_living ** 2
train_data["sqft_living_squared"] = train_data["sqft_living"] ** 2

#pick feature
features = ["sqft_living", "sqft_living_squared"]
# fit our model
sqft_model = LinearRegression()
X = tools.get_features_matrix(train_data,features)
y = train_data["price"]
sqft_model.fit(X,y)

print("intercept of model is", sqft_model.intercept_)
print("coefficients of model are", sqft_model.coef_)
# get test data r sq
test_data = pd.read_csv("kc_house_test_data.csv")
test_data["sqft_living_squared"] = test_data["sqft_living"] ** 2
X = tools.get_features_matrix(test_data, features)
y = test_data["price"]
test_r_sq = sqft_model.score(X,y)
print(test_r_sq)


plt.scatter(train_data["sqft_living"], train_data["price"], s =1)
x_values  = list(range(train_data["sqft_living"].max()))
y_values  = [poly_pt(x, sqft_model) for x in x_values]

plt.plot(x_values, y_values, c = "y")
plt.show()