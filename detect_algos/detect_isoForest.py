#!/usr/bin/env python
# detect_EDK-means.py
# The python script is used to detect abnormal frames in event time series using Isolation Forest.
# @Author  : wwt
# @Date    : 2018-12-25

import numpy as np
from common import common_funcs
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plot
import matplotlib.axes as ax

### global params
# the range of data to learn
data_form = 'std'
start_h = 0
end_h = 1621
trees = 100
samples_tree = 256 # 'auto'
features_tree = 1
cr = 0.1 # contamination rate
save_path = "pred_isoForest_" + "trees" + str(trees) + '_cr' + str(cr) + '_' \
            + data_form + '_' + str(start_h) + "to" + str(end_h) + ".txt"

# read data from the cleaned, normalized/standardized data set
data = common_funcs.readData("data_std.txt", skips=1)
data = data[start_h:end_h, :]

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
isoForest_model.fit(data)

dec_func = isoForest_model.decision_function(data)
print(dec_func)

# save the result to file
common_funcs.saveData(save_path, dec_func, delim=',', linenum=False)

plot.scatter(range(start_h,end_h), dec_func, s=1**2)

# discrimination horizon
plot.hlines(y=0, xmin=start_h, xmax=end_h, linestyles='dashed')

#plot.ylim((-2,2))
plot.show()
