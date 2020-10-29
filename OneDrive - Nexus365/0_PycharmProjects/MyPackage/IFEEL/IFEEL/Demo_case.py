import numpy as np
import pandas as pd
import os
from IFEEL import ifeel_transformation, ifeel_extraction

os.getcwd()  # get current working directory

### Notes ###
# Download test dataset (test_data_for_IFEEL.csv) from https://github.com/chacehoo/IFEEL/tree/master/OneDrive%20-%20Nexus365/0_PycharmProjects/MyPackage/IFEEL/Test_Data
# Place the downloaded "test_data_for_IFEEL.csv" under current working directory

# Read data
df_test = pd.read_csv("test_data_for_IFEEL.csv", header=0,index_col=0, parse_dates=False)

# Parameter setting
# Business hours are from 9 am to 5 pm
time_business_start = 9
time_business_end = 17
alphabet_size = 7    # alphabet size of SAX representation

# Data transformation
[df_raw, df_raw_diff, df_SAX_number, df_SAX_alphabet, df_SAX_number_diff] = ifeel_transformation.feature_transformation(df_test, alphabet_size,time_business_start,time_business_end)

# Global feature extraction for each daily profile
feature_global_all_days = pd.DataFrame()
for i in np.arange(0, df_raw.shape[0]):
    ts = df_raw.iloc[i]
    ts_diff = df_raw_diff.iloc[i]
    feature_global_all_each = ifeel_extraction.feature_global(ts, ts_diff).global_all().T
    feature_global_all_days = feature_global_all_days.append(feature_global_all_each, ignore_index=True)

feature_global_all_days.columns = ifeel_extraction.feature_name_global
feature_global_all_days.head()

# Peak feature extraction for each daily profile
feature_peak_period_all_days = pd.DataFrame()
for i in np.arange(0, df_raw.shape[0]):
    ts_sax = df_SAX_number.iloc[i]
    ts_sax_diff = df_SAX_number_diff.iloc[i]
    feature_peak_all_each = ifeel_extraction.feature_peak_period(ts_sax, ts_sax_diff,alphabet_size).T
    feature_peak_period_all_days = feature_peak_period_all_days.append(feature_peak_all_each, ignore_index=True)

feature_peak_period_all_days.columns = ifeel_extraction.feature_name_peak
feature_global_all_days.head()
