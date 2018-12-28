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
''' model params '''
w = 0.5  # trades off statistical distance and prediction distance
boxes = 24
cr = 0.1  # contamination rate that determines anomaly threshold in each box

save_path = "pred_PAS_" + "alpha" + str(w) + '_cr' + str(cr) + '_' \
            + data_form + '_' + str(start_h) + "to" + str(end_h) + ".txt"

# read data from the cleaned, normalized/standardized data set
data = common_funcs.readData(data_file, skips=1, cols=(1, 2, 3, 4, 5))
data = data[start_h:end_h, :]


def getHourlyStats():
    return means, stds
