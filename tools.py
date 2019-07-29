"""
Generaly tools for machine learning
by Colm Gallagher
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,Ridge

def pred_r_squared(data, target_col_name, features_list):
    """
    Predicted r squared is got by removing a sample(row) fitting the model
    then finding the errors on this term then finding the sum of squares of this vector
    :param data:
    :param features_list:
    :return:
    """
    features = [data[feature] for feature in features_list]
    features_matrix = np.zeros((data.shape[0], len(features_list)))
    for idx in range(len(features_list)):
        features_matrix[:, idx] = features[idx]
    # create the vector that is the row of predictions
    predictions = np.zeros(data.shape[0])
    for row_idx in range(data.shape[0]):
        adjusted_matrix = np.delete(features_matrix, row_idx, axis=0)
        lm = LinearRegression()
        # remove target col entry at row idx


        lm.fit(adjusted_matrix, data[target_col_name].drop(row_idx))
        predictions[row_idx] = lm.predict(features_matrix[row_idx].reshape(1,-1))

    error = predictions - data[target_col_name]
    #predicted sum of squares
    PSS = np.dot(error, error)
    TSS = total_sum_of_squares(data, target_col_name)
    pred_r_sq = 1 - PSS/TSS
    return pred_r_sq


def total_sum_of_squares(data, target_col_name):
    assert type(data) == pd.DataFrame
    assert type(target_col_name) == str

    avg = data[target_col_name].mean()
    errors = data[target_col_name] - avg
    TSS = np.dot(errors, errors)
    return TSS


def residual_sum_of_squres(data, target_col_name, predictions):
    assert type(data) == pd.DataFrame
    assert type(target_col_name) == str

    errors = (data[target_col_name] - predictions)
    RSS = np.dot(errors, errors)
    return RSS


def r_squared(data, target_col_name, predictions):
    TSS = total_sum_of_squares(data, target_col_name)
    RSS = residual_sum_of_squres(data, target_col_name, predictions)

    return 1 - RSS/TSS


def ME(data, target_col_name,predictions):
    """
    Returns the mean of the errors(absolute error)
    :param data:
    :param target_col_name:
    :param predictions:
    :return:
    """

    predictions = pd.Series(predictions)
    num_samples = data.shape[0]
    total_errors = np.abs(data[target_col_name] - predictions).sum()
    return total_errors/num_samples


def MSE(data, target_col_name,predictions):
    num_samples = data.shape[0]
    RSS = residual_sum_of_squres(data, target_col_name, predictions)
    return RSS/num_samples


def normalise(series):
    assert type(series) == pd.Series
    return ( series - series.min() )/( series.max()- series.min() )


def normalise_wrt(target_series, reference_series):
    """
    normalises the target series with respect to the reference series
    IE uses the min(m) and max(M) of the reference series and then translates
    the target series (x - m)/ (M - m)
    :param target_series:
    :param reference_series:
    :return:
    """
    m = reference_series.min()
    M = reference_series.max()
    ans = (target_series - m) / (M - m)
    return ans


def get_features_matrix(data, features_list):
    features = [data[feature] for feature in features_list]
    features_matrix = np.zeros((data.shape[0], len(features_list)))
    for idx in range(len(features_list)):
        features_matrix[:, idx] = features[idx]
    return  features_matrix


def polynomial_dataframe(feature, degree):
    """
    Takes a normalized(values between 0 and 1) pandas series and returns
    a dataframe with its powers
    :param feature:
    :param degree:
    :return:
    """
    assert type(feature) == pd.Series
    normed = feature.min() >= 0 and feature.max() <= 1
    assert normed or feature.dtype == float
    # assume that degree >= 1
    # initialize the dataframe:
    poly_dataframe = pd.DataFrame()
    # and set poly_dataframe['power_1'] equal to the passed feature
    poly_dataframe[feature.name + '_power_1'] = pd.Series(feature, dtype=np.float64)
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        for power in range(2, degree+1):
            # first we'll give the column a name:
            name = feature.name + '_power_' + str(power)
            # assign poly_dataframe[name] to be feature^power; use apply(*)
            poly_dataframe[name] = poly_dataframe[feature.name + '_power_1'] ** power
    return poly_dataframe


def add_poly_feature(data, feature, degree, normed = False):
    """
    Adds powers of a feature to the dataframe
    Returns the appended dataframe
    :param data:
    :param feature: name of df column or else a pd series
    :return:
    """
    if type(feature) ==  str:
        feature = data[feature]
    assert type(feature) == pd.Series

    if normed:
        print("Caution when using norm on only some of the data")
        print("Can lead to error due to norm being different in test and train data")
        feature = normalise(feature)
        feature.name = feature.name + "_normed"
    else:
        feature = feature.map(float)
    poly_df = polynomial_dataframe(feature, degree)
    data = pd.concat([data, poly_df], axis = 1)
    return data


def get_poly_col_names(data, feature_name, normed = False):
    if normed:
        normed_name  = feature_name + "_normed"
        poly_cols = []
        for col_name in data.columns:
            if col_name[:len(normed_name)] == normed_name:
                poly_cols.append(col_name)
    else:
        poly_cols = []
        for col_name in data.columns:
            if col_name[:len(feature_name) +7 ] == feature_name + "_power_":
                poly_cols.append(col_name)

    return poly_cols


def linear_k_fold_cross_validation(k, data, features_list, target):
    """
    Return the average r squared
    :param k: number of sets to split data into(num repitions of the process)
    :param data:
    :param features_list:
    :param target:
    :return:
    """
    assert type(target) == str
    pass
    n = len(data)
    # get the start and end indices for the validation sets
    indices = []
    for i in range(k):
        start = int((n * i) / k)
        end = int((n * (i + 1)) / k - 1)
        indices.append((start, end))

    R_sq_total = 0
    # iterate over all the validation sets to find the avg R_sq
    for i in range(k):
        start = indices[i][0]
        end = indices[i][1]
        valid_set = data.iloc[start: end + 1]
        before = data.iloc[0:start]
        after = data.iloc[end + 1: n]
        train_set = pd.concat([before, after])

        X_train = get_features_matrix(train_set, features_list)
        y_train = train_set[target]
        X_valid = get_features_matrix(valid_set, features_list)
        lm = LinearRegression(normalize=True)
        lm.fit(X_train, y_train)
        valid_predictions = lm.predict(X_valid)
        R_sq = r_squared(valid_set, target, valid_predictions)
        R_sq_total += R_sq

    avg_R_sq = R_sq_total / k
    return avg_R_sq


def ridge_k_fold_cross_validation(k, l2_penalty, data, features_list, target):
    """
    Return the average r squared
    :param k: number of sets to split data into(num repitions of the process)
    :param l2_penalty:
    :param data:
    :param features_list:
    :param target:
    :return:
    """
    assert type(target) == str
    pass
    n = len(data)
    # get the start and end indices for the validation sets
    indices = []
    for i in range(k):
        start = int((n * i) / k)
        end = int((n * (i + 1)) / k - 1)
        indices.append((start, end))

    R_sq_total = 0
    # iterate over all the validation sets to find the avg R_sq
    for i in range(k):
        start = indices[i][0]
        end = indices[i][1]
        valid_set = data.iloc[start: end + 1]
        before = data.iloc[0:start]
        after = data.iloc[end + 1: n]
        train_set = pd.concat([before, after])

        X_train = get_features_matrix(train_set, features_list)
        y_train = train_set[target]
        X_valid = get_features_matrix(valid_set, features_list)
        rm = Ridge(alpha=l2_penalty, normalize=True)
        rm.fit(X_train, y_train)
        valid_predictions = rm.predict(X_valid)
        R_sq = r_squared(valid_set, target, valid_predictions)
        R_sq_total += R_sq

    avg_R_sq = R_sq_total / k
    return avg_R_sq


def is_numeric(series):
    for elt in series:
        try:
            float(elt)
        except:
            return False
    return True


def train_valid_k_fold_sets(data, k):
    """

    :param data:
    :param k:
    :return: list of (train,valid) tuples
    """

    n = len(data)
    # get the start and end indices for the validation sets
    indices = []
    for i in range(k):
        start = int((n * i) / k)
        end = int((n * (i + 1)) / k - 1)
        indices.append((start, end))

    ans_sets = []
    for i in range(k):
        start = indices[i][0]
        end = indices[i][1]
        valid_set = data.iloc[start: end + 1]
        before = data.iloc[0:start]
        after = data.iloc[end + 1: n]
        train_set = pd.concat([before, after])
        ans_sets.append((train_set, valid_set))

    return ans_sets


def general_CFV(data, prediction_function, target_column, param, k=10):
    """

    :param data: data to perform CFV on
    :param prediction_function: prediction_function: maps (training data, testing data, k)
           to a prediction series of testing data
    :param target_column: column of df that is being predicted
    :param param: parameter of the model being used
    :param k: k to be used in k fold CFV
    :return: CFV r squared
    """
    train_valid_splits = train_valid_k_fold_sets(data, k)

    total_r_sq = 0
    # iterate through each (train,valid) tuple finding r squared for each
    # add this r squared to the running total r squared
    for train_set, valid_set in train_valid_splits:
        # find predictions of valid_set based on train_set
        valid_set_predictions = prediction_function(train_set, valid_set, param)
        valid_set_r_sq = r_squared(valid_set, target_column,
                                         valid_set_predictions)
        total_r_sq += valid_set_r_sq

    # find the average r squared over the different validation sets
    avg_r_sq = total_r_sq / len(train_valid_splits)
    return avg_r_sq

