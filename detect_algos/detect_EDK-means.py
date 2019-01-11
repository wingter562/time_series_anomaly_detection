#!/usr/bin/env python
# detect_EDK-means.py
# The python script is used to cluster event time series using Euclidean Distance K-means.
# @Author  : wwt
# @Date    : 2018-12-20

import numpy as np
from common import common_funcs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plot
import sys

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
model_name = "Euclidean Distance K-means"
n_cls = 2
save_path = "pred_Kmeans_" + '_NClass' + str(n_cls) + \
            '_' + str(start_h) + "to" + str(end_h) + data_form + \
            '_slotSize' + str(slot_size) + ".txt"

# read data from the cleaned, normalized/standardized data set
data = common_funcs.readData(data_file, skips=1, cols=(0, 1, 2, 3, 4, 5), datatype=JZLogFrame_type)
print(data)
data = data[start_h:end_h]


# get slot-wise data
# build slots, totally 24/slot_size slots
slots = common_funcs.getFixedSlotFrameSets(data, slot_size, True, 'date')
# decisions set, should be temporally sequential from start_h to end_h
glob_decisions_map = list(range(start_h, end_h - start_h))
# invoke in-built k-means to build a model for each time slot
# Thus totally num_slots Models are to be built
model_set = []

# fit/train model one by one for each slot
for slot in slots:
    time_seq = np.array(slot)[:, 0].tolist()  # get timestamp sequence and transform it to hour-index sequence
    for k in range(len(time_seq)):
        time_seq[k] = common_funcs.countHoursFromStr(time_seq[k])  # convert t to absolute time
        time_seq[k] -= common_funcs.countHoursFromStr(start_date)  # now t is the hour index

    kmeans_model = KMeans(n_clusters=n_cls)
    kmeans_model.fit(np.delete(np.array(slot), 0, 1))  # feed timestamp-stripped slot data

    # force minor class label to be '-1', and positive label '1'
    local_decisions_map = kmeans_model.labels_.tolist()  # 0 or 1 originally
    zeroes = local_decisions_map.count(0)
    ones = local_decisions_map.count(1)
    minor_class = 0 if zeroes < ones else 1
    for k in range(len(local_decisions_map)):
        local_decisions_map[k] = -1 if local_decisions_map[k] == minor_class else 1

    # mapping decisions of this slot-local model to the global decision map
    for idx in range(len(time_seq)):
        glob_decisions_map[time_seq[idx]] = local_decisions_map[idx]  # store decisions

    model_set.append(kmeans_model)  # store the model
    print("Info@EDK-means: model id%s stored." % id(kmeans_model))

print("Info@EDK-means: pos_count=%d, neg_count=%d" % (glob_decisions_map.count(1), glob_decisions_map.count(-1)))

#common_funcs.saveData(save_path, glob_decisions_map, delim=',', linenum=False)

plot.scatter(range(start_h, end_h), glob_decisions_map, s=1 ** 2)
plot.hlines(y=0, xmin=start_h, xmax=end_h, linestyles='dashed')
plot.title(model_name + "\ndecision map (+1/-1 denotes normality/anomaly)")
plot.show()
