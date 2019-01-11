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
data = common_funcs.readData(data_file, skips=1, cols=(0,1,2,3,4,5), datatype=JZLogFrame_type)
data = data[start_h:end_h]  # with datatype specified, data is an 1-d array


class Trend:
    """
    # The Trend object for storing trend vector(multi-channel slopes, essentially) and its center
    # Trend can be formatted as [trend_center_1 : trend_vector_1()]
    # Member:
    #   dimension: the dimension of the trend
    #   span: the length of the trend, i.e., number of frames covered
    #   center: the center of the trend which is a timestamp/sequence number
    #   trend: the vector of slopes (or sig(slope)), e.g., (0.1, 0.7, 0.3)
    """
    def __init__(self, dimension, length, center, span):
        self.dimension = dimension
        self.length = length
        self.center = center
        self.span = span

    def get_dimension(self):
        return self.dimension

    def get_span(self):
        return self.length

    def get_center(self):
        return self.center

    def get_trend_vector(self):
        return self.span

    def get_start_time(self):
        """
        # only applicable to sequence numbered trends
        :return: Etarting time of the trend as a sequence number
        """
        return self.get_center() - math.floor(self.get_span())

    def get_end_time(self):
        """
        # only applicable to sequence numbered trends
        :return: End time of the trend as a sequence number
        """
        return self.get_center() + math.floor(self.get_span())


# TODO: functions like get_trend are to be finished
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

