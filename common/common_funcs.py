# This module consists of common functions for anomaly detection algorithms.
# @Author  : wwt
# @Date    : 2018-12-20
#

import math as math
import numpy as np

# read data from a cleaned data set
# @fname data file path, abs path
def readData(fname):
    data = np.loadtxt(fname, dtype=float, delimiter=',', skiprows=1)
    return data


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


### test block
#data = readNormData("../data_norm.txt")
#print data

#print EDist([1,2,3.4], [1,2,3.0])

#print np.mean(np.array([[0,0,0],[3,3,3],[4,4,4]]), axis=0) # along rows


