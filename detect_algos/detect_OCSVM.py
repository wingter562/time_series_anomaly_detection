#!/usr/bin/env python
# detect_EDK-means.py
# The python script is used to detect abnormal frames in event time series using One-Class-Support Vector Machine.
# @Author  : wwt
# @Date    : 2018-12-24

import numpy as np
from common import common_funcs
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plot

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
model_name = 'One Class SVM'
SVMKernel = 'rbf'
gamma = 'scale'
contamination = 0.1
save_path = "pred_OCSVM_" + SVMKernel + '_nu' + str(contamination) + '_' \
            + data_form + '_' + str(start_h) + "to" + str(end_h) + \
            '_slotSize' + str(slot_size) + ".txt"


# read data from the cleaned, normalized/standardized data set
data = common_funcs.readData(data_file, skips=1, cols=(0, 1, 2, 3, 4, 5), datatype=JZLogFrame_type)
data = data[start_h:end_h]


# get slot-wise data
# build slots, totally 24/slot_size slots
slots = common_funcs.getFixedSlotFrameSets(data, slot_size, True, 'date')
# decisions set, should be temporally sequential from start_h to end_h
glob_decisions_map = range(start_h, end_h - start_h)
# invoke in-built k-means to build a model for each time slot
# Thus totally num_slots Models are to be built
model_set = []

# fit/train model one by one for each slot
for slot in slots:
    time_seq = np.array(slot)[:, 0].tolist()  # get timestamp sequence and transform it to hour-index sequence
    for k in range(len(time_seq)):
        time_seq[k] = common_funcs.countHoursFromStr(time_seq[k])  # convert t to absolute time
        time_seq[k] -= common_funcs.countHoursFromStr(start_date)  # now t is the hour index

    OCSVM_model = OneClassSVM(kernel=SVMKernel, gamma='scale', nu=contamination)
    OCSVM_model.fit(np.delete(np.array(slot), 0, 1))  # feed timestamp-stripped slot data

    # force minor class label to be '-1', and positive label '1'
    local_decisions_map = OCSVM_model.decision_function(np.delete(np.array(slot), 0, 1)).tolist()

    # mapping decisions of this slot-local model to the global decision map
    for idx in range(len(time_seq)):
        glob_decisions_map[time_seq[idx]] = local_decisions_map[idx]  # store decisions

    model_set.append(OCSVM_model)  # store the model
    print("Info@OCSVM: model id%s stored." % id(OCSVM_model))

common_funcs.saveData(save_path, glob_decisions_map, delim=',', linenum=False)

plot.scatter(range(start_h, end_h), glob_decisions_map, s=1 ** 2)
plot.hlines(y=0, xmin=start_h, xmax=end_h, linestyles='dashed')
plot.title(model_name + "\ndecision map (+/- denotes normality/anomaly)")
plot.show()




