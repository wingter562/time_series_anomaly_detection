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


