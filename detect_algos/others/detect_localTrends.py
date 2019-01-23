#!/usr/bin/env python
# detect_localTrends.py
# The python script implements our proactive anomaly detection based on fixed-size local trends.
# @Author  : wwt
# @Date    : 2019-01-11

import numpy as np
from common import common_funcs
import math
import sys
from queue import Queue
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


''' global params '''
data_file = "data_std.txt"
data_form = 'std'
# define data structure for Jiuzhou Log data (normalized/standardized)
# i.e., timestamp(str), CROND(float), RSYSLOGD(float), SESSION(float), SSHD(float), SU(float)
JZLogFrame_type = np.dtype([('timestamp', 'U13'), ('CROND', 'f8'), ('RSYSLOGD', 'f8'),
                            ('SESSION', 'f8'), ('SSHD', 'f8'), ('SU', 'f8')])
if sys.version_info[0] < 3:  # for py2
    JZLogFrame_type = np.dtype([('timestamp', 'S13'), ('CROND', 'f8'), ('RSYSLOGD', 'f8'),
                                ('SESSION', 'f8'), ('SSHD', 'f8'), ('SU', 'f8')])
# the range of data to learn
start_date = "2018-06-29-00"
start_h = 0
end_h = 3457
# slotting
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
data = common_funcs.read_data(data_file, skips=1, cols=(0, 1, 2, 3, 4, 5), datatype=JZLogFrame_type)
data = data[start_h:end_h]  # with datatype specified, data is an 1-d array


# TODO: member functions to be finished
class LocalTrend:
    """
    # The object for storing the local trend (the slope between two adjacent frames)
    # Local trend is a vector in multivariate time series starting at some point (origin)
    # Member:
    #   dimension: the dimension/number of channels
    #   origin: the starting time which is a timestamp/sequence number
    #   local_trend: the vector of slopes, e.g., (0.2, 0.3, 0.4)
    """
    def __init__(self, dimension, origin, local_trend):
        self.dimension = dimension
        self.origin = origin
        self.local_trend = local_trend

    def get_dimension(self):
        return self.dimension



class Trend:
    """
    # The object for storing trend vector(multi-channel slopes, essentially) and its center
    # Trend can be formatted as [local_trend_obj0, local_trend_obj1, ...]
    # Member:
    #   dimension: the dimension of the trend (channels per local trend)
    #   span: the length of the trend, i.e., number of intervals covered
    #   origin: the starting time of the trend which is a timestamp/sequence number
    #   trend: a list of LocalTrends, e.g., [(0.1,0.7,0.3), (0.2,0.3,0.4),..., ()]
    """
    def __init__(self, dimension, span, origin, trend=None):
        self.dimension = dimension
        self.span = span
        self.origin = origin
        self.trend = trend

        self.origin = trend[0].get_origin()  # set trend origin to its 1st local trend's origin

    def get_dimension(self):
        return self.dimension

    def get_span(self):
        return self.length

    def get_origin(self):
        return self.origin

    def get_trend(self):
        return self.trend

    def get_end_time(self):
        """
        # only applicable to sequence numbered trends
        :return: End time of the trend as a sequence number
        """
        return self.get_origin() + math.floor(self.get_span())


# TODO: member functions like get_trend are to be finished
class TrendsBuffer:
    """
    # The class is defined to store already-predicted trend objects in a queue
    # Member:
    #   buffer: the buffer implemented as a queue
    #   buffer_length: the max length of the buffer, Number of trends (i.i., buffer length) need to store
    #       is equal to the math.ceil(local_trend_len / 2)
    """
    def __init__(self, length):
        self.buffer_length = length
        self.buffer = Queue(maxsize=length)

    def get_buffer_size(self):
        """
        # get the size (length) of the trend buffer
        :return: buffer size
        """
        return self.buffer.qsize()

    def get_trends_buffer(self):
        """
        # return the buffer
        :return: buffer
        """
        return self.buffer

    def print_trends_buffer(self):
        """
        # print the buffer
        :return: N/A
        """
        for x in self.buffer.queue:
            print(x)

    def add_trend(self, trend):
        """
        # Store a new trend object into the buffer after popping the head (if not empty)
        :return: N/A
        """
        if not self.buffer.empty():
            self.buffer.get()  # pop the oldest trend stored
        self.buffer.put(trend)  # enqueue the new trend object


# TODO:
def build_trend_pred_model():
    return


# TODO:
def get_pred_trend(model, frame_t):
    return


# TODO:
def get_cosine_similarity(x, y):
    return


# TODO:
def get_future_trends_from_frame(frame_t):
    """
    # get the future trends (slopes) for frame t, which must contain timestamp
    # Timestamp of frame t is required so as to obtain previous frames t-1, t-2, ...
    :param frame_t:
    :return:
    """
    return


# TODO:
def get_anomaly_score(frame_t):
    """
    # Compute and return the anomaly score of given frame based on the set of predicted future trends starting
    # from his frame
    :param frame_t: the frame to test abnormality
    :return: anomaly score
    """
    return





### test block
#print(getHourlyFrameSetList(data)[0])  # nice
#showHourlyAvg(data)
#print getStatsPositions(data)[3]  # fine

