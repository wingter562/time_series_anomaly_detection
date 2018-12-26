# This module consists of common functions for anomaly detection algorithms.
# @Author  : wwt
# @Date    : 2018-12-20
#

import math as math
import numpy as np


# read data from a cleaned data set
# @fname data file path, abs path
def readData(fname, skips=1):
    data = np.loadtxt(fname, dtype=float, delimiter=',', skiprows=skips)
    return data


# save mem data to local disk
# @output_f target file path for saving
# @data data to save
# @delim delimiter, default=','
# @headline headline to print, default=''
# @linenum enable/disable line number, default=True
def saveData(output_f, data, delim=',', headline='', linenum=True):
    # add linenum column
    if linenum:
        data = np.array(data)
        # for one-column data, transform it to matrix and transpose
        if len(data.shape) == 1:
            data = np.array([data]).T
        nums = np.array([range(0, data.shape[0], 1)]).T

        data = np.concatenate((nums, data), axis=1)

    with open(output_f, 'w') as f:
        np.savetxt(output_f, data, delimiter=delim, header=headline)


# calculate the Euclidean distance between two vectors
# @x vector x
# @y vector y
def EDist(x, y):
    # get #dimensions
    d_x = np.size(x)
    d_y = np.size(y)

    # check validity
    if(d_x != d_y):
        print "Err@EDist: unequal dimensions of input vectors"
        return -1

    sum = 0
    for k in range(0, d_x):
        sum = sum + (x[k] - y[k])**2

    return math.sqrt(sum)


# calculate Area Under receiver operating characteristic Curve (AUC)
# @labels real data labels, 1-normal, -1-abnormal
# @preds decision functions or labels from detectors
def calculateAUC(labels, preds):
    return

### test block
#data = readNormData("../data_norm.txt")
#print data

#print EDist([1,2,3.4], [1,2,3.0])

#print np.mean(np.array([[0,0,0],[3,3,3],[4,4,4]]), axis=0) # along rows


