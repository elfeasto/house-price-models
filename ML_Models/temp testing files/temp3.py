import pandas as pd

def add_yr_built_dummies(data):
    def f(x):
        if x < 1950:
            return 0
        elif x < 1975:
            return 1
        elif x < 1997:
            return 2
        else:
            return 3

    data['yr_built'] = data['yr_built'].map(f)

    yr_built_dummies = pd.get_dummies(data['yr_built'], prefix='yr_built_cat')
    data = pd.concat([data, yr_built_dummies], axis=1)
    data.drop(columns='yr_built', inplace=True)
    return  data


def add_zipcode_dummies(data):
    zipcode_dummies = pd.get_dummies(data['zipcode'])
    data = pd.concat([data, zipcode_dummies], axis=1)
    data.drop(columns = 'zipcde', inplace = True)
    return data