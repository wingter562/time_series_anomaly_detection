#!/usr/bin/env python
# detect_EDK-means.py
# The python script is used to detect abnormal frames in event time series using Isolation Forest.
# @Author  : wwt
# @Date    : 2018-12-25

import numpy as np
from common import common_funcs
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plot
import sys

''' global params '''
data_file = "../preprocessed_data/data_std.txt"
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
model_name = 'Isolation Forest'
trees = 100
samples_tree = 256 # 'auto'
features_tree = 1
cr = 0.1 # contamination rate
save_path = "pred_isoForest_" + "trees" + str(trees) + '_cr' + str(cr) + '_' \
            + data_form + '_' + str(start_h) + "to" + str(end_h) + \
            '_slotSize' + str(slot_size) + ".txt"


# read data from the cleaned, normalized/standardized data set
data = common_funcs.read_data(data_file, skips=1, cols=(0, 1, 2, 3, 4, 5), datatype=JZLogFrame_type)
data = data[start_h:end_h]

# get slot-wise data
# build slots, totally 24/slot_size slots
slots = common_funcs.get_fixed_slot_frame_sets(data, slot_size, True, 'date')
# training_set_decisions set, should be temporally sequential from start_h to end_h
glob_decisions_map = list(range(start_h, end_h - start_h))
# invoke in-built k-means to build a model for each time slot
# Thus totally num_slots Models are to be built
model_set = []

# fit/train model one by one for each slot
for slot in slots:
    time_seq = np.array(slot)[:, 0].tolist()  # get timestamp sequence and transform it to hour-index sequence
    for k in range(len(time_seq)):
        time_seq[k] = common_funcs.count_hours_from_str(time_seq[k])  # convert t to absolute time
        time_seq[k] -= common_funcs.count_hours_from_str(start_date)  # now t is the hour index

    # invoke to train in-built iForest model
    # [1]Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation forest."
    #     Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on.
    # [2]Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation-based anomaly detection."
    #     ACM Transactions on Knowledge Discovery from Data (TKDD) 6.1 (2012): 3.
    #
    # @n_estimator number of isolation trees
    # @max_samples number of samples to be isolated in each tree, 'auto'=256(2**8) as demonstrated on p.9, [2]
    # @contamination contamination rate, directly determines the threshold
    # @max_features features used in a tree, optimal value=1 as shown in Fig.11, [2]
    # @bootstrap sampling with/without replacement, 'False'=without replacement
    # @behavior
    isoForest_model = IsolationForest(n_estimators=trees, max_samples='auto', contamination=cr,
                                      max_features=features_tree, bootstrap=False, behaviour='new')
    isoForest_model.fit(np.delete(np.array(slot), 0, 1))  # feed timestamp-stripped slot data

    # force minor class label to be '-1', and positive label '1'
    local_decisions_map = isoForest_model.decision_function(np.delete(np.array(slot), 0, 1)).tolist()

    # mapping training_set_decisions of this slot-local model to the global decision map
    for idx in range(len(time_seq)):
        glob_decisions_map[time_seq[idx]] = local_decisions_map[idx]  # store training_set_decisions

    model_set.append(isoForest_model)  # store the model
    print("Info@isoForest: model id%s stored." % id(isoForest_model))

#common_funcs.saveData(save_path, glob_decisions_map, delim=',', linenum=False)

plot.scatter(range(start_h, end_h), glob_decisions_map, s=1 ** 2)
plot.hlines(y=0, xmin=start_h, xmax=end_h, linestyles='dashed')
plot.title(model_name + "\ndecision map (+/- denotes normality/anomaly)")
plot.show()
