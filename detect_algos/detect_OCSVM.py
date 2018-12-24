#!/usr/bin/env python
# detect_EDK-means.py
# The python script is used to detect abnormal frames in event time series using One-Class-Support Vector Machine.
# @Author  : wwt
# @Date    : 2018-12-24

import numpy as np
from common import common_funcs
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plot

### global params
# the range of data to learn
start_h = 2523
end_h = 3457

# read data from the cleaned, normalized/standardized data set
data = common_funcs.readData("data_std.txt")
data = data[start_h:end_h, :]

# invoke to train in-built One Class SVM
# @kernel Kernel function
# @gamma Kernel coefficient for 'rbf', 'poly', 'sigmoid' (default=1/n_features, 'scale'=1/(n_feature*X_std)
# @nu contamination rate
OCSVM_model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
OCSVM_model.fit(data)


print(OCSVM_model.decision_function(data))
# for l in kmeans_model.labels_:
#    print(l)
plot.scatter(range(start_h,end_h), OCSVM_model.decision_function(data), s=1**2)
#plot.ylim((-2,2))
plot.show()
