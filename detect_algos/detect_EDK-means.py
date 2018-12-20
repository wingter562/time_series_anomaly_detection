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
max_loop = 30000

# read data from the cleaned, normalized data set - "data_norm.txt"
data = common_funcs.readData("data_norm.txt")
data = data[0:2500,:]

# invoke in-built k-means
kmeans_model = KMeans(n_clusters=n_cls)
kmeans_model.fit(data)


print(kmeans_model.cluster_centers_)
# for l in kmeans_model.labels_:
#    print(l)
plot.scatter(range(0,2500), kmeans_model.labels_)
plot.show()