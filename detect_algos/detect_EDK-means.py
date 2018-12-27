#!/usr/bin/env python
# detect_EDK-means.py
# The python script is used to cluster event time series using Euclidean Distance K-means.
# @Author  : wwt
# @Date    : 2018-12-20

import numpy as np
from common import common_funcs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plot

''' global params '''
# the range of data to learn
data_form = 'std'
start_h = 0
end_h = 1621
''' model params '''
n_cls = 2
save_path = "pred_Kmeans_" + '_NClass' + str(n_cls) + '_' \
            + data_form + '_' + str(start_h) + "to" + str(end_h) + ".txt"

# read data from the cleaned, normalized/standardized data set
data = common_funcs.readData("data_std.txt", skips=1)
data = data[start_h:end_h, :]

# invoke in-built k-means
kmeans_model = KMeans(n_clusters=n_cls)
kmeans_model.fit(data)


print(kmeans_model.cluster_centers_)

# clustering result, i.e., decision function
dec_func = kmeans_model.labels_
# convert label==0 to -1, for intuitive comparison with other detectors
conv = lambda x: x if(x == 1) else -1
for k in range(0, len(dec_func)):
    dec_func[k] = conv(dec_func[k])


common_funcs.saveData(save_path, dec_func, delim=',', linenum=False)

plot.scatter(range(start_h, end_h), dec_func, s=1**2)
plot.show()