#!/usr/bin/env python
# detect_EDK-means.py
# The python script implements our Prediction-driven Anomaly Detection.
# @Author  : wwt
# @Date    : 2018-12-23

import numpy as np
from common import common_funcs
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


''' global params '''
# the range of data to learn
data_file = "data_std.txt"
data_form = 'std'
JZLogFrame_type = np.dtype([('timestamp', 'S13'), ('CROND', 'f8'), ('RSYSLOGD', 'f8'),
                            ('SESSION', 'f8'), ('SSHD', 'f8'), ('SU', 'f8')])
start_date = "2018-06-29-00"
start_h = 0
end_h = 3457
slot_size = 3
num_models = 24/slot_size if 24%slot_size == 0 else 24/slot_size + 1

''' model params '''
w = 0.5  # trades off statistical distance and prediction distance
boxes = 24
cr = 0.1  # contamination rate that determines anomaly threshold in each box

save_path = "pred_PAS_" + "alpha" + str(w) + '_cr' + str(cr) + '_' \
            + data_form + '_' + str(start_h) + "to" + str(end_h) + \
            '_slotSize' + str(slot_size) + ".txt"

# read data from the cleaned, normalized/standardized data set
# timestamps inclusive
data = common_funcs.readData(data_file, skips=1, cols=(0,1,2,3,4,5), datatype=JZLogFrame_type)
data = data[start_h:end_h]  # with datatype specified, data is an 1-d array


def getHourlyStats(data, stats_to_get='mean'):
    """
    # First get hourly frame sets by calling getHourlyFrameSetList(data),
    # then compute means & standard deviation in each set
    :param data: set of data in format: <timestamp, feature_1,...,feature_m>
    :param stats_to_get: stats to get - 'mean' or 'std_var'
    :return: hourly statistics, each of which is a list of num_features-dimensional vector
    # e.g., [mean_f1, mean_f2, mean_f3,..., mean_fm]
    """
    # hourly stats lists contain 24 * [f1_aggregated, f2_aggregated, f3_aggregated,...]
    hourly_means = []
    hourly_stdvars = []

    hourly_frameset_list = common_funcs.getHourlyFrameSets(data);


    # get average, h=hour, c=channel
    for h in range(0, 24, 1):
        hourly_means.append(hourly_frameset_list[h].mean(axis=0))  # average over this hourly set for each feature
        hourly_stdvars.append(hourly_frameset_list[h].std(axis=0))  # without Bessel's correction

    # return hourly statistics, each of which is a list of num_features-dimensional vector
    # e.g., [mean_f1, mean_f2, mean_f3,..., mean_fm]
    stats_dict = {}
    stats_dict['mean'] = hourly_means
    stats_dict['std_var'] = hourly_stdvars

    return stats_dict.get(stats_to_get, 'N/A')


def showHourlyAvg(data):
    """
    # show channel averages by hours
    :param data: set of data in format: <timestamp, feature_1,...,feature_m>
    :return: N/A
    """
    means = getHourlyStats(data, stats_to_get='mean')
    means = np.array(means)  # for easy slicing
    print means

    tline = range(0, 24, 1)  # a day
    plt.figure(1)  # figure 1

    # event-CROND
    plt.subplot(321)
    plt.plot(tline, means[:, 0], 'k', linewidth=0.3, markersize=0.4)
    plt.xticks(np.arange(0, 24, 3))
    plt.title("CROND")

    # event-RSYSLOGD
    plt.subplot(322)
    plt.plot(tline, means[:, 1], 'm', linewidth=0.3, markersize=0.4)
    plt.xticks(np.arange(0, 24, 3))
    plt.title("RSYSLOGD")

    # event-SESSSION
    plt.subplot(323)
    plt.plot(tline, means[:, 2], 'c', linewidth=0.3, markersize=0.4)
    plt.xticks(np.arange(0, 24, 3))
    plt.title("SESSION")

    # event-SSHD
    plt.subplot(324)
    plt.plot(tline, means[:, 3], 'y', linewidth=0.3, markersize=0.4)
    plt.xticks(np.arange(0, 24, 3))
    plt.title("SSHD")

    # event-SU
    plt.subplot(325)
    plt.plot(tline, means[:, 4], 'g', linewidth=0.3, markersize=0.4)
    plt.xticks(np.arange(0, 24, 3))
    plt.title("SU")

    plt.show()

# unfinished
def boxing():
    return

# unfinished
def buildPredictiveModelForOneSlot():
    return

# unfinished
def buildPredictiveModels():
    return

# unfinished
def getPredPositions(models, period_to_pred):
    return


def getStatsPositions(data_train):
    """
    # get hourly statistical positions from the training data
    Statistical position (SP) is the expected frame position from the distribution of frame vectors of the same hour,
    # that is, the hourly mean.
    :param data_train: 1-D array of dtype([(timestamp,'S13'), (feature_1, 'f8'), (feature_2, 'f8'),...,(feature_m, 'f8')])
    :return: statistical positions as a 24-dimensional list of (avg_f1, avg_f2,...,avg_fm) tuples
    """
    hourly_means = getHourlyStats(data_train, stats_to_get='mean')
    #hourly_stdv = getHourlyStats(data_train, stats_to_get='std_var')

    return hourly_means


# unfinished
def getRefPositions(data_train, pred_models, period_to_pred, w):
    """
    # The core of Prediction-driven anomaly detection - estimating the reference position (RP) in which a 'normal'
    # frame is most likely to appear. Mainly used for evaluation.
    # Reference position is determined by 'statistical position'(SP) and 'predicted position'(PP).
    # 1) Statistical position is the expected frame position from the distribution of frame vectors of the same hour,
    # that is, the hourly mean.
    #   Get SP by invoking getStatsPosition(data_train)
    # 2) Predicted position is the speculated frame position based on a window (slot) of previous frames
    #   Get PP by invoking getPredPositions(models, period_to_pred)
    # Note: predicted position is only applicable when the predictive model is already trained.
    #
    # The desired Reference Position is the weighed combination of 'statistical position' and 'predicted position'
    # i.e., RP = w*SP + (1-w)*PP,
    # where w is the tunable parameter to trade off the impact of historical distribution and local trend.
    :param data_train: training/historical data
    :param pred_models: predictive models - already trained
    :param period_to_pred: a given period to calculate reference positions in - continuous
    :param w: weight parameter - singular
    :return: Reference position as a frame vector
    """
    timestamps = data_train[0]

    num_channels = len(data_train[0]) - 1  # number of features with timestamp exclusive
    ref_pos = [0] * num_channels
    stats_positions = []

    # get statistical positions for the given period/a sequence of timestamp
    stats_pos_set = getStatsPositions(data_train)
    for k in period_to_pred:
        hourstamp = period_to_pred[k]
        stats_positions.append(stats_pos_set[hourstamp])

    # get predicted positions
    pred_pos = getPredPositions(pred_models, period_to_pred)

    # combine
    ref_pos = w * stats_pos + (1 - w) * pred_pos





    return





### test block
#print(getHourlyFrameSetList(data)[0])  # nice
#showHourlyAvg(data)
#print getStatsPositions(data)[3]  # fine

