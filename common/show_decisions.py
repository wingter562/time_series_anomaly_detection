#!/usr/bin/env python
# show_decisions.py
# The python script is used to present decision functions from different anomaly detection methods
# @Author  : wwt
# @Date    : 2018-12-26

import numpy as np
from common import common_funcs
import matplotlib.pyplot as plot
from sklearn.preprocessing import StandardScaler



def plot_decisions(file_list, span=(-1, -1), legends=None):
    """
    # plot the training_set_decisions(i.e., cluster labels or anomaly scores)
    :param file_list: a list of files storing the training_set_decisions of detection methods apiece
    :param span: starting timestamp of the samples under test, default = full range
    :param legends: legends for each plot, tuple
    :return: N/A

    ### example 1: Plot the first 50 decision results stored separately in three files by three different detectors
    # range to plot
    start_h = 0
    end_h = 50
    legends = ("K-Means", "OCSVM", "isoForest")  # legends, optional

    # files containing results
    pred_files = ["../training_set_decisions/pred_Kmeans__NClass2_0to3457std_slotSize3.txt",
                  "../training_set_decisions/pred_OCSVM_rbf_nu0.1_std_0to3457_slotSize3.txt",
                  "../training_set_decisions/pred_isoForest_trees100_cr0.1_std_0to3457_slotSize3.txt"]

    plot_decisions(pred_files, start_h, legends)
    """
    # count candidates
    num_predictors = len(file_list)
    # store the training_set_decisions of methods in a list
    decs_mat = []
    colour_map = ['c','m','g','r','y','b']
    #colour_map = np.random.rand(num_predictors)
    for f in file_list:
        # read training_set_decisions from each file
        decs_mat.append(common_funcs.read_data(f, skips=0, cols=(0,)))

    # rescale the matrix to make intuitive comparison
    for row in range(0, len(decs_mat)):
        row_max = decs_mat[row].max()
        decs_mat[row] = decs_mat[row]/row_max

    # define range
    n = decs_mat[0].shape[0]
    if span[0] == -1 or span[0] > span[1] or span[1] == -1:
        print("Err@show_decisions: invalid range given, show all the decisions instead...")
        show_from = 0
        show_to = n
    else:
        show_from = span[0]
        show_to = span[1]

    t = range(show_from, show_to, 1)


    # plot handles
    h = []
    # plot decision of each frame by each predictor
    for k in range(0, num_predictors):
        h.append(plot.scatter(t, decs_mat[k][show_from:show_to], c=colour_map[k], s=1**2))

    # draw legends
    if legends:  # if legends assigned
        if len(legends) == num_predictors:
            plot.legend(tuple(h), legends)
        else:
            print("Err@show_decisions: number of legends given incorrect, should be %d" % num_predictors)

    plot.title("decision functions('anomalies'<0)")

    # discrimination horizon
    plot.hlines(y=0, xmin=show_from, xmax=show_to, linestyles='dashed')
    #plot.ylim((-1, 1))
    plot.show()



## example 1
# start_h = 0
# end_h = 50
# legends = ("K-Means", "OCSVM", "isoForest")
#
#
# pred_files = ["../training_set_decisions/pred_Kmeans__NClass2_0to3457std_slotSize3.txt",
#               "../training_set_decisions/pred_OCSVM_rbf_nu0.1_std_0to3457_slotSize3.txt",
#               "../training_set_decisions/pred_isoForest_trees100_cr0.1_std_0to3457_slotSize3.txt"]
#
# plot_decisions(pred_files, span=(0, 3457), legends=legends)
