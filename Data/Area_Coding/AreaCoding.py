"""
Attempting to put a number on the quality of area that a house is in.

This is to be done by looking at the price difference between price and
estimated price of a house and making a heat map
Then grids are made to group houses with similar differences together
These grids are assigned a numeric value
A new feature is created which maps house location to grid value
This new feature and sqft_living are then in a multiple linear regression
formula to give a new way of estimating house price

Currently there are two areas both with ad hoc assigned values
"""



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




def show_heatmap(data, target_col, areas, num_quantiles = 10):
    """
    Show heat map of each house mapped to its location on the grid
    The color represents the difference between its predicted price
    and it's actual price
    """

    #### Get the color that each house will be
    # quantile the target column
    tar_quantiled = get_quantile_values(train_data[target_col], num_quantiles).astype(float)
    # g sets the color value of each house based on it which quantile it is in
    g = lambda x: (1 - x, x, 0)
    data_colors = tar_quantiled.map(g)

    ### draw the heat map (with areas included) ###
    # get axis object
    fig, ax = plt.subplots(1)
    #label our plot
    plt.xlabel("long")
    plt.ylabel("lat")
    # draw areas to map
    for area in areas:
        area.draw(ax)
    # draw heat map of houses
    plt.scatter(data["long"], data["lat"], c=data_colors, s=1)
    # show heat map
    plt.show()

####################################################
## load data and get prediction using sqft_living ##
####################################################

# load data

#load the data
train_data, valid_data, test_data = get_train_valid_test_data()

# get prediction using sqft_living
lm = LinearRegression()
lm.fit(train_data[["sqft_living"]], train_data["price"])
train_data["predicted"] = lm.predict(train_data[["sqft_living"]])
# gett difference between price and predicted price
train_data["diff"] = train_data["price"] - train_data["predicted"]


#########################################
####### Make the areas ##################
#########################################

# all grids below are "nice" areas of the city
grid = Grid((-122.425, 47.6), 0.26, 0.1)
grid2 = Grid((-122.25, 47.53) ,0.04, 0.07)
grid3 = Grid((-122.21, 47.579), 0.04, 0.021)
grid4 = Grid((-122.425, 47.51), 0.04, 0.09)
high_value_grids = [grid, grid2, grid3, grid4]

HIGH_RATING = 1
high_value_area =  Area(high_value_grids, True, HIGH_RATING)

#currenly low are is everywhere outside the nice area
LOW_RATING = 0
low_value_area = Area(high_value_grids, False, LOW_RATING)

areas = [high_value_area, low_value_area]

###################################################
### show heatmap ##################################
###################################################

NUM_QUANTILES = 3
show_heatmap(train_data, "diff", [high_value_area], NUM_QUANTILES)

############################################################
# add a column to our data representing the estimated      #
# price increase of living in different areas              #
############################################################


add_area_rating(train_data, areas)
add_area_rating(valid_data, areas)


#######################################################################
# get linear regression score using new area category and sqft_living#
#######################################################################
features = ["sqft_living", "house_area_rating"]
lm = LinearRegression()
lm.fit(train_data[features], train_data["price"])
valid_r_sq = lm.score(valid_data[features], valid_data["price"])
print("Validation r squared is", valid_r_sq)

##########################
# debugging area #########
##########################

low_area_diff = low_value_area.avg_diff_value(train_data)
high_area_diff = high_value_area.avg_diff_value(train_data)
print( "Average difference in low area is", np.round(low_area_diff,0) )
print( "Average difference in high area is", np.round(high_area_diff,0) )