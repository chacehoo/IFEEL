
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import itertools

# Python will not import methods whose names are with a leading underscore
def _get_length_sequences_where(x):
    """
    This method calculates the length of all sub-sequences where the array x is either True or 1.
    Examples:
    x = [0,1,0,0,1,1,1,0,0,1,0,1,1]
    _get_length_sequences_where(x)
    [1, 3, 1, 2]
    x: An iterable containing only 1, True, 0 and False values
    return: A list with the length of all sub-sequences where the array is either True or False. If no ones or Trues
    contained, the list [0] is returned.
    """
    if len(x) == 0:
        return [0]
    else:
        res = [len(list(group)) for value, group in itertools.groupby(x) if value == 1]
        return res if len(res) > 0 else [0]

###################################
#### global feature extraction ####
###################################
# Class is a “template” / “blueprint” that is used to create objects.
class feature_global(object):

    # Attribute references
    # Class variables shared by all instances
    feature_name_global = [
        'Mean',
        'Std',
        'Max',
        'Min',
        'Range (i.e., max-min)',
        'Percentage above mean',
        'Sum of net loads during business hours',
        'Sum of net loads during non-business hours',
        'Skewness',
        'Kurtosis',
        'Mode of 5-bin histogram',
        'Longest period above mean',
        'Longest period of successive increase'
    ]

    # class instantiation automatically invokes __init__() for the newly-created class instance.
    # instance variables unique to each instance
    def __init__(self, ts, ts_diff):
        self.ts = ts
        self.ts_diff = ts_diff

    # a method is an action which an object is able to perform.
    def global_mean(self):
        return np.mean(self.ts)

    def global_std(self):
        return np.std(self.ts)

    def global_max(self):
        return np.max(self.ts)

    def global_min(self):
        return np.min(self.ts)

    def global_range(self):
        return np.max(self.ts)-np.min(self.ts)

    def global_percentage_above_mean(self):
        percentage = sum(i > np.mean(self.ts) for i in self.ts) / len(self.ts)
        return percentage

    def global_sum_net_loads_busi(self):
        x = self.ts
        ts_busi = x.loc[x.index.get_level_values('business_hour') == True]
        sum_busi = np.sum(ts_busi)
        return sum_busi

    def global_sum_net_loads_nonbusi(self):
        x = self.ts
        ts_non_busi = x.loc[x.index.get_level_values('business_hour') == False]
        sum_non_busi = np.sum(ts_non_busi)
        return sum_non_busi

    def global_skewness(self):
        skewness = skew(self.ts)
        return skewness

    def global_kurtosis(self):
        kurtosis_value = kurtosis(self.ts)
        return kurtosis_value

    def global_mode_histogram_bins5(self):
        hist, edges = np.histogram(self.ts, bins=5)
        index_max = np.argmax(hist)
        mode_range = [edges[index_max], edges[index_max + 1]]
        mode = np.mean(mode_range)
        return mode

    def global_longest_period_above_mean(self):
        # Returns the length of the longest consecutive subsequence in x that is bigger than the mean of x
        x = self.ts
        if not isinstance(x, (np.ndarray, pd.Series)):
            x = np.asarray(x)     # Convert the input to an array
        return np.max(_get_length_sequences_where(x > np.mean(x))) if x.size > 0 else 0

    def global_longest_period_of_successive_increase(self):
        # Returns the length of the longest consecutive increase using SAX difference word (df_SAX_number_diff_pivot)
        x = self.ts_diff
        if not isinstance(x, (np.ndarray, pd.Series)):
            x = np.asarray(x)

        return np.max(_get_length_sequences_where(x > 0)) if x.size > 0 else 0

    def global_all(self):
        # Get all time domain features in one function
        # return: all time domain features in a list
        feature_all = list()
        feature_all.append(self.global_mean())
        feature_all.append(self.global_std())
        feature_all.append(self.global_max())
        feature_all.append(self.global_min())
        feature_all.append(self.global_range())
        feature_all.append(self.global_percentage_above_mean())
        feature_all.append(self.global_sum_net_loads_busi())
        feature_all.append(self.global_sum_net_loads_nonbusi())
        feature_all.append(self.global_skewness())
        feature_all.append(self.global_kurtosis())
        feature_all.append(self.global_mode_histogram_bins5())
        feature_all.append(self.global_longest_period_above_mean())
        feature_all.append(self.global_longest_period_of_successive_increase())
        return pd.DataFrame(feature_all)


#################################
#### peak feature extraction ####
#################################
def feature_peak_period(ts_sax_number, ts_sax_number_diff, alphabet_size):

    peak = alphabet_size-1
    ts_to_boolean = ts_sax_number == peak
    peak_index_connected = np.where(ts_to_boolean)[0]  # where 返回的是个 tuple，用 [0] 获得真正的索引数组

    is_peak_exist = float(peak) in ts_sax_number.values
    if is_peak_exist==0:
        peak_number = 0
        peak_time = np.nan
        peak_time_diff_shortest = np.nan
        peak_duration = np.nan
        peak_longest_time = np.nan
        peak_longest_duration = np.nan
        peak_longest_slope_upward = np.nan
        peak_longest_slope_downward = np.nan
    else:
        # Group consecutive integers in an array?
        # 搜索不等于1的点，将其定位为断点，使用 where 返回不等于1的索引即为断点
        # != means not equal
        peak_index_separated = np.split(peak_index_connected, np.where(np.diff(peak_index_connected) != 1)[0] + 1)
        peak_number = len(peak_index_separated)
        peak_time = [np.mean(i) for i in peak_index_separated]

        if peak_number==1:
            peak_time_diff_shortest = np.nan
        else:
            peak_time_diff_shortest = np.min(np.diff(peak_time))

        peak_duration = [len(i) for i in peak_index_separated]
        peak_longest = [i for i in peak_index_separated if len(i) == max(peak_duration)][0]  # 1) If Condition in Python List   2) if there are multiple peaks with the same length, then also take the first one
        peak_longest_time = np.mean(peak_longest)
        peak_longest_duration = len(peak_longest)
        peak_longest_slope_upward = ts_sax_number_diff[peak_longest[0]]

        if peak_longest[-1] == 23:
            peak_longest_slope_downward = -1
        else:
            peak_longest_slope_downward = ts_sax_number_diff[peak_longest[-1]+1]

    return pd.DataFrame([peak_number,
                        peak_time,
                        peak_time_diff_shortest,
                        peak_duration,
                        peak_longest_time,
                        peak_longest_duration,
                        peak_longest_slope_upward,
                        peak_longest_slope_downward
                         ])

feature_name_peak = [
        'Peak_all: number',
        'Peak_all: time',
        'Peak_all: shortest interval between two peaks',
        'Peak_all: duration',
        'Peak_longest: occurrence time',
        'Peak_longest: duration',
        'Peak_longest: upward slope',
        'Peak_longest: downward slope']
