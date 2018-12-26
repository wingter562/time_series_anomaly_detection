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
data_form = 'std'
start_h = 0
end_h = 1621
SVMKernel = 'rbf'
gamma = 'scale'
contamination = 0.1
save_path = "pred_OCSVM_" + SVMKernel + '_nu' + str(contamination) + '_' \
            + data_form + '_' + str(start_h) + "to" + str(end_h) + ".txt"


# read data from the cleaned, normalized/standardized data set
data = common_funcs.readData("data_std.txt", skips=1)
data = data[start_h:end_h, :]

# invoke to train in-built One Class SVM
# @kernel Kernel function
# @gamma Kernel coefficient for 'rbf', 'poly', 'sigmoid' (default=1/n_features, 'scale'=1/(n_feature*X_std)
# @nu contamination rate
OCSVM_model = OneClassSVM(kernel=SVMKernel, gamma='scale', nu=contamination)
OCSVM_model.fit(data)


dec_func = OCSVM_model.decision_function(data)
print(dec_func)

# save the result to file
common_funcs.saveData(save_path, dec_func, delim=',', linenum=False)

plot.scatter(range(start_h,end_h), dec_func, s=1 ** 2)
#plot.ylim((-2,2))

# discrimination horizon
plot.hlines(y=0, xmin=start_h, xmax=end_h, linestyles='dashed')

plot.show()

