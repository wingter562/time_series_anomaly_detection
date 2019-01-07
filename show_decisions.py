#!/usr/bin/env python
# show_decisions.py
# The python script is used to present decision functions from different anomaly detection methods
# @Author  : wwt
# @Date    : 2018-12-26

import numpy as np
from common import common_funcs
import matplotlib.pyplot as plot
from sklearn.preprocessing import StandardScaler


# plot the decisions(i.e., cluster labels or anomaly scores)
# @file_list a list of files storing the decisions of detection methods apiece
# @start_h starting timestamp of the samples under test
# @legends legends for each plot, tuple
def plotDecisions(file_list, start_h, legends):
    # count candidates
    num_predictors = len(legends)
    # store the decisions of methods in a list
    decs_mat = []
    colour_map = ['c','m','g','r','y','b']
    #colour_map = np.random.rand(num_predictors)
    for f in file_list:
        # read decisions from each file
        decs_mat.append(common_funcs.readData(f, skips=0, cols=(0,)))

    # rescale the matrix to make intuitive comparison
    for row in range(0, len(decs_mat)):
        row_max = decs_mat[row].max()
        decs_mat[row] = decs_mat[row]/row_max


    n = decs_mat[0].shape[0]
    t = range(start_h, start_h+n, 1)


    # plot handles
    h = []
    # plot decision of each frame by each predictor
    for k in range(0, num_predictors):
        h.append(plot.scatter(t, decs_mat[k], c=colour_map[k], s=1**2))

    # legends
    plot.legend(tuple(h), legends)
    plot.title("decision functions('anomalies'<0)")

    # discrimination horizon
    plot.hlines(y=0, xmin=start_h, xmax=start_h+n, linestyles='dashed')
    #plot.ylim((-1, 1))
    plot.show()



### test block
# params
start_h = 0
end_h = 50
legends = ("K-Means", "OCSVM", "isoForest")


pred_files = ["detect_algos/pred_Kmeans__NClass2_0to3457std_slotSize3.txt",
              "detect_algos/pred_OCSVM_rbf_nu0.1_std_0to3457_slotSize3.txt",
              "detect_algos/pred_isoForest_trees100_cr0.1_std_0to3457_slotSize3.txt"]

plotDecisions(pred_files, start_h, legends)
