#!/usr/bin/env python
# detect_EDK-means.py
# The python script implements our Prediction-driven Anomaly Detection.
# @Author  : wwt
# @Date    : 2018-12-23

import numpy as np
from common import common_funcs
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plot


''' global params '''
data_form = 'std'
data_file = "data_std.txt"
# the range of data to learn
start_h = 0
end_h = 1621
# define data structure for Jiuzhou Log data (normalized/standardized)
# i.e., timestamp(str), CROND(float),  RSYSLOGD(float), SESSION(float), SSHD(float), SU(float)
JZLogFrame_type = np.dtype([('timestamp', 'S13'), ('CROND', 'f8'), ('RSYSLOGD', 'f8'),
                            ('SESSION', 'f8'), ('SSHD', 'f8'), ('SU', 'f8')])


''' model params '''
w = 0.5  # trades off statistical distance and prediction distance
boxes = 24
cr = 0.1  # contamination rate that determines anomaly threshold in each box

save_path = "pred_PAS_" + "alpha" + str(w) + '_cr' + str(cr) + '_' \
            + data_form + '_' + str(start_h) + "to" + str(end_h) + ".txt"

# read data from the cleaned, normalized/standardized data set
# timestamps inclusive
data = common_funcs.readData(data_file, skips=1, cols=(0,1,2,3,4,5), datatype=JZLogFrame_type)
data = data[start_h:end_h]  # with datatype specified, data is an 1-d array



def getHourlyFrameSetList(data):
    """
    # divide the the dataset by hour and return a 24-hour-size list of it
    :param data: set of data in format: <timestamp, feature_1,...,feature_m>
    :return: a list(size=24) of np.array, with the size of each array = num_frames_hourly * features
    """
    # hourly frame set is a set storing [frames_in_a_hour] frames with sample hour-stamp
    hourly_frame_set = []

    # hourly_frame_set_list is a 24-size list of hourly_frame_set
    hourly_frame_set_list = []
    for h in range(0,24,1):
        hourly_frame_set_list.append([])  # cannot append variable hourly_frame_set, it would be a pointer

    # dive into the data frame by frame
    for frame in data:
        stamp = frame[0]  # timestamp
        hour = int(stamp.split('-')[3])

        # add the stamp-stripped record to the corresponding set
        hourly_frame_set_list[hour].append((frame[1], frame[2], frame[3], frame[4], frame[5]))

    # convert each hourly set to numpy.ndarray
    for h in range(0,24,1):
        hourly_frame_set_list[h] = np.array(hourly_frame_set_list[h])

    return hourly_frame_set_list


# unfinished
def getHourlyStats(data):
    # hourly stats are [24 * [num_channels]] lists
    hourly_means = []
    hourly_stds = []
    hourly_num_records = [0] * 24

    hourly_frameset_list = getHourlyFrameSetList(data);

    # aggregate by hour
    hourly_counters = [0, 0, 0, 0, 0]  # sum of 5 channels for a specified hour period
    hourly_counters_list = []  # 24 hours list
    for h in range(0, 24, 1):
        hourly_counters_list.append(hourly_counters)
        hourly_means.append(hourly_counters)
        hourly_stds.append(hourly_counters)

    # for each frame record, parse timestamp (frame[0]) and count value (frame[1 to 5]) from each channel
    # into the corresponding vector of counters
    for frame in data:
        stamp = frame[0]  # timestamp
        hour = int(stamp.split('-')[3])

        hourly_counters_list[hour][0] += frame[1]  # CROND = frame[1]
        hourly_counters_list[hour][1] += frame[2]  # RSYSLOGD = frame[2]
        hourly_counters_list[hour][2] += frame[3]  # SESSION = frame[3]
        hourly_counters_list[hour][3] += frame[4]  # SSHD = frame[4]
        hourly_counters_list[hour][4] += frame[5]  # SU = frame[5]

        hourly_num_records[hour] += 1

    # get average, h=hour, c=channel
    for h in range(0, 24, 1):
        hourly_means[h] = hourly_frameset_list[h]
        for c in range(0, 5):
            hourly_means[h][c] = hourly_counters_list[h][c] / hourly_num_records[h]
            hourly_stds[h][c] =



    return hourly_means, hourly_stds


# test block
print(getHourlyFrameSetList(data)[1])  # nice
