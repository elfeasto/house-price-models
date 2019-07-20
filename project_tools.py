"""
Tools for the coursea house price dataset (seattle area)
by Colm Gallagher
"""

import pandas as pd
import matplotlib.pyplot as plt

data_path = "C:\Python3.5\MyProgram\Pycharm Projects\kingcountyhouseprices\Data\\"

def get_train_test_data():
    dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float, 'grade': int,
                  'yr_renovated': int, 'price': float, 'bedrooms': float, 'zipcode': str, 'long': float,
                  'sqft_lot15': float, 'sqft_living': float, 'floors': float, 'condition': int, 'lat': float,
                  'date': str,
                  'sqft_basement': int, 'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}


    train_file = data_path + "kc_house_train_data.csv"
    test_file = data_path + "kc_house_test_data.csv"
    train_data = pd.read_csv(train_file, dtype=dtype_dict)
    test_data = pd.read_csv(test_file, dtype=dtype_dict)
    return train_data, test_data


def get_train_valid_test_data():
    dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float, 'grade': int,
                  'yr_renovated': int, 'price': float, 'bedrooms': float, 'zipcode': str, 'long': float,
                  'sqft_lot15': float, 'sqft_living': float, 'floors': float, 'condition': int, 'lat': float,
                  'date': str,
                  'sqft_basement': int, 'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}

    train_file = data_path + "training_less_valid.csv"
    valid_file = data_path + "validation_data.csv"
    test_file =  data_path + "kc_house_test_data.csv"
    train_data = pd.read_csv(train_file,dtype= dtype_dict)
    valid_data = pd.read_csv(valid_file, dtype= dtype_dict)
    test_data = pd.read_csv(test_file, dtype= dtype_dict)
    return train_data, valid_data, test_data


def get_quantile_values(x, num_quantiles):
    """
    Enter in an array of values x and a number of quantiles to
    split the values of x into
    It will then return the series with each element as a fraction
    in [0,1) representing its quantile:
    x -> quantile(x)/num_quantiles
    where 0 <= quant(x) < num_quantiles
    so  0 <= f(x) < 1

    :param x (Iterable) : Series to be split in quantiles
    :param num_quantiles(N) : number of quantiles needed

    :return (pd.series): A series of values representing the fraction of quantiles each
    value is greater than
    """
    ### helper function ###
    def f(interval, bins):
        """
        helper function for get_quantile_value fn
        :param interval:
        :param bins:
        :return:
        """
        return bins[bins == interval].index.values.astype(int)[0]

    # get all values for quantiles
    quant_values = x.quantile([i / num_quantiles for i in range(num_quantiles + 1)])
    # put a handier index on the series
    quant_values.index = range(num_quantiles + 1)
    # subtract one from lowest value for later use of semi open intervals
    quant_values[0] = quant_values[0] - 1
    # create bins as tuples
    bin_tuples = [(quant_values[i], quant_values[i + 1]) for i in range(num_quantiles)]
    # convert tuples to bins(half open intervals)
    bins = pd.IntervalIndex.from_tuples(bin_tuples)
    # map each element of x to its corresponding interval
    cuts = pd.cut(x, bins)
    bins = pd.Series(bins)
    x_quantiled = cuts.map(lambda a: f(a,bins)/num_quantiles)
    return x_quantiled


def house_area_rating(house, areas):
    """
    The areas should cover all of the map
    (have one as an exterior area)
    In the case of a house being in multiple areas it will
    return the rating of the first
    :param house:
    :param areas:
    :return:
    """
    # find which area the house is in
    # get its value
    for area in areas:
        if area.contains_house(house):
            return area.rating

    raise "House is not in any of the areas"


def area_rating_series(data, areas):
    """
    return a series with the area rating of each house in the data
    :param data:
    :param areas:
    :return:
    """
    f = lambda h: house_area_rating(h, areas)
    houses_area = data.apply(func=f, axis="columns")
    return houses_area


def add_area_rating(data, areas):
    """
    create a new column in the data with an area rating
    called "house_area_rating"
    :param data:
    :param areas:
    :return:
    """
    f = lambda h: house_area_rating(h, areas)
    data["house_area_rating"] = data.apply(func=f, axis="columns")


def show_heatmap(data, target_col, areas = None, num_quantiles = 10):
    """
    Show heat map of each house mapped to its location on the grid
    The color represents the difference between its predicted price
    and it's actual price
    """

    #### Get the color that each house will be
    # quantile the target column
    tar_quantiled = get_quantile_values(data[target_col], num_quantiles).astype(float)
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
    if areas:
        for area in areas:
            area.draw(ax)
    # draw heat map of houses
    plt.scatter(data["long"], data["lat"], c=data_colors, s=1)
    # show heat map
    plt.show()