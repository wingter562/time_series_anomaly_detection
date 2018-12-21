#!/usr/bin/env python
# detect_EDK-means.py
# The python script is used to classify event time series using Euclidean Distance K-means.
# @Author  : wwt
# @Date    : 2018-12-20

import numpy as np
from common import common_funcs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plot

### global params
n_cls = 2
#max_loop = 30000
# the range of data to learn
start_h = 0
end_h = 2129

# read data from the cleaned, normalized/standardized data set
data = common_funcs.readData("data_std.txt")
data = data[start_h:end_h, :]

# invoke in-built k-means
kmeans_model = KMeans(n_clusters=n_cls)
kmeans_model.fit(data)


print(kmeans_model.cluster_centers_)
# for l in kmeans_model.labels_:
#    print(l)
plot.scatter(range(start_h,end_h), kmeans_model.labels_, s=1**2)
plot.show()