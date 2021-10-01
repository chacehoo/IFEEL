import numpy as np
import pandas as pd
from scipy.stats import norm
import string
from datetime import time, datetime

def feature_transformation(df_test, alphabet_size, time_business_start, time_business_end):
    # Symbolic Aggregate ApproXimation (SAX) Transformation
    def discretizer(ts, breakpoints):
        return np.where(breakpoints > float(ts))[0][0]

    def stringizer(row):
        return ''.join(string.ascii_letters[int(row['level'])])

    # add "is_business_hour" to the column, forming a MultiIndex
    is_busi_hour_all = []
    for i in np.arange(0, df_test.shape[1]):
        time_str = df_test.columns[i]
        t_test = datetime.strptime(time_str, '%H:%M:%S').time()
        is_busi_hour = lambda t: t >= time(time_business_start, 00) and t <= time(time_business_end, 00)
        boolean = is_busi_hour(t_test)
        is_busi_hour_all.append(boolean)

    tuples = list(zip(df_test.columns, is_busi_hour_all))
    df_test.columns = pd.MultiIndex.from_tuples(tuples, names=('Time', 'business_hour'))

    # SAX representation
    # SAX words are used for temporal feature extraction
    df_znorm_all_house = pd.DataFrame()
    df_SAX_number_all_house = pd.DataFrame()
    df_SAX_all_house = pd.DataFrame()

    for i in np.arange(0, df_test.shape[0]):
        df_SAX_each = df_test.iloc[i]
        x = df_SAX_each.fillna(method='ffill')
        y = (x - np.mean(x)) / np.std(x)
        y = pd.DataFrame(y.values)
        y.columns = ["normalized_power"]
        breakpoints = norm.ppf(np.linspace(1. / alphabet_size, 1 - 1. / alphabet_size, alphabet_size - 1))    # ppf: Percent point function (inverse of cdf)
        breakpoints = np.concatenate((breakpoints, np.array([np.Inf])))
        y['level'] = y.apply(discretizer, axis=1, args=[breakpoints])
        y['letter'] = y.apply(stringizer, axis=1)
        # save
        df_znorm_all_house = df_znorm_all_house.append(y['normalized_power'], ignore_index=True)
        df_SAX_number_all_house = df_SAX_number_all_house.append(y['level'], ignore_index = True)
        df_SAX_all_house = df_SAX_all_house.append(y['letter'], ignore_index = True)


    df_znorm_all_house.index = df_test.index
    df_SAX_number_all_house.index = df_test.index
    df_SAX_all_house.index = df_test.index

    df_znorm_all_house.columns = df_test.columns
    df_SAX_number_all_house.columns = df_test.columns
    df_SAX_all_house.columns = df_test.columns

    df_SAX_number_diff_pivot = df_SAX_number_all_house.diff(periods=1, axis=1)

    df_raw_diff = df_test.diff(periods=1, axis=1)

    return df_test, df_raw_diff, df_SAX_number_all_house, df_SAX_all_house, df_SAX_number_diff_pivot