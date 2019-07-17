"""
Attempting to put a number on the quality of area that a house is in.

In this module THREE different areas are used
They are named green(high rating), yellow(medium rating) and red(low rating)
(hence the traffic lights name)

It is assumed that the main two factors for house price are house size and
house location.
Using this assumption we estimate the house price using simple linear
regression on the house size(sqft_living)
Then the price difference between price and estimated price is examined
(this value is called diff)
Diff is used to map a heat map (maps location using long and lat to diff)

This heat map is used to make three different areas corresponding to the three
types of area(each area is a combination of sqaure grids)
Currently this is done manually by making the areas based on eyeballing the
heat map

The are ratings are assigned by taking the average diff in each area

The new feature is created which maps house location to area rating
This new feature and sqft_living are then in a multiple linear regression
formula to give a new way of estimating house price
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
import tools

"""
Make a color coded scatter plot of house location with color relating to price
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import tools
from classes import *
from project_tools import *



def traffic_light_heatmap(data, target_col, areas):
    """
    attempt to clear up the heatmap by using red,yellow and green for colors
    :param data:
    :param target_col:
    :param grids:
    :return:
    """
    # get axis object
    fig, ax = plt.subplots(1)
    # quantiled the target column
    tar_quantiled = get_quantile_values(train_data[target_col], 3).astype(float)

    # round values for convenience
    tar_quantiled = tar_quantiled.map(lambda x: np.round(x,1))
    # set the color values
    def traffic_colors_map(x):
        """
        maps the three house price quantiles to their
        corresponding color
        """
        #map 0 to red
        if x == 0:
            return (1,0,0)
        #map 0.3 to yellow
        elif x ==0.3:
            return (1,1,0)
        #map 0.7 to green
        elif x == 0.7:
            return (0,1,0)
        else:
            raise
    data_colors = tar_quantiled.map(traffic_colors_map)
    # draw heat map
    plt.scatter(data["long"], data["lat"], c=data_colors, s=1)

    # draw grids to map
    for area in areas:
        area.draw(ax)

    # show heat map
    plt.show()


def get_diff_series(training_data):
    """
    return the difference between the predicted value and the actual price
    the predicted value is the linear sqft model
    :param training_data:
    :return:
    """
    # get prediction using sqft_living
    lm = LinearRegression()
    lm.fit(training_data[["sqft_living"]], training_data["price"])
    predicted = lm.predict(training_data[["sqft_living"]])
    diff = training_data["price"] - predicted
    return diff


def make_areas():
    """
    rougly divide the map into three zones of postive diff
    negative diff and the remainder should be neutral diff
    :return:
    """
    grid1 = Grid((-122.425, 47.6), 0.26, 0.1)
    grid2 = Grid((-122.25, 47.53), 0.04, 0.07)
    grid3 = Grid((-122.21, 47.579), 0.04, 0.021)
    grid4 = Grid((-122.425, 47.51), 0.04, 0.09)
    green_grids = [grid1, grid2, grid3, grid4]
    green_area = Area(green_grids, interior=True, color="green")

    # bad area
    red = "red"
    grid5 = Grid((-122.4, 47.25), 0.5, 0.15)
    grid6 = Grid((-121.89, 47.5), 0.05, 0.05)
    grid7 = Grid((-122.2, 47.4), .1, .12)
    red_grids = [grid5, grid6, grid7]
    red_area = Area(red_grids, interior=True, color="red")

    # medium area is everywhere outside the good or bad area
    # will take the complement of thses grids below
    yellow_grids = green_grids + red_grids
    yellow_area = Area(yellow_grids, interior=False, color="yellow")
    # make a list of all areas
    areas = [green_area, yellow_area, red_area]
    return areas


#load the data
train_data,  test_data = get_train_test_data()
#caculate the difference column
train_data["diff"] = get_diff_series(train_data)
#get the areas
areas = make_areas()
#add in area rating based on diff
for area in areas:
    area.add_area_diff_rating(train_data)
#add a column to our data representing the estimated
#price increase of living in different areas
train_data["house_area_cat"] = area_rating_series(train_data,areas)
test_data["house_area_cat"] = area_rating_series(test_data, areas)
#get linear regression model using new area category #
features = ["sqft_living", "house_area_cat"]
lm = LinearRegression()
lm.fit(train_data[features], train_data["price"])
#find the test r squared
test_r_sq = lm.score(train_data[features], train_data["price"])
print("Test r squared is", test_r_sq)


##########################
# debugging area #########
##########################

#traffic_light_heatmap(train_data, "diff", areas)
