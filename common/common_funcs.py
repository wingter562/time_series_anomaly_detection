# This module consists of common functions for anomaly detection algorithms.
# @Author  : wwt
# @Date    : 2018-12-20
#

import math as math
import numpy as np


def readData(fname, skips=1, cols=(0,1,2,3,4,5), datatype = float):
    """
    # read data from a cleaned data set, with column 0 supposed to be timestamp
    :param fname: data file path, abs path
    :param skips: num of lines to skip from beginning
    :param cols: indexes of columns of data to use, begins from 0
    :param datatype: the structure of a row in the data, containing several fields
    :return: data in np.array
    """
    data = np.loadtxt(fname, dtype=datatype, delimiter=',', skiprows=skips, usecols=cols)
    return data


def saveData(output_f, data, delim=',', headline='', linenum=True):
    """
    # save mem data to local disk
    :param output_f: target file path for saving
    :param data: data to save
    :param delim: delimiter, default=','
    :param headline: headline to print, default=''
    :param linenum: enable/disable line number, default=True
    :return: N/A
    """
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


def EDist(x, y):
    """
    calculate the Euclidean distance between two vectors
    :param x: vector x
    :param y: vector y
    :return: Euclidean distance between the given x and y
    """
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


def calculateAUC(labels, preds):
    """
    # calculate Area Under receiver operating characteristic Curve (AUC), based on
    # [] Hand, D. J., & Till, R. J. (2001). A simple generalisation of the area under the ROC curve
    #   for multiple class classification problems. Machine learning, 45(2), 171-186.
    # AUC ~= (Sa - Na*(Na+1)/2) / Na*Nn,
    # where Na, Nn are the number of labeled anomalies and normal points, respectively. And
    # Sa is the sum of ranks of anomalies sorted in ascending order by scoring from a detector.
    :param labels: real data labels, 1=normal, -1=abnormal
    :param preds: decision functions or labels given by detectors
    :return: AUC
    """
    # check length equality
    if len(labels) != len(preds):
        print("err@calculateAUC: label vec and preds vec must be of same length")
        return -1

    # count Na and Nn
    Na = 0
    Nn = 0
    for l in labels:
        if l < 0:
            Na = Na + 1
        else:
            Nn = Nn + 1

    # total samples
    N = Na + Nn

    # calculate Sa, use a list of tuples
    Sa = 0
    pl_list = []
    for k in range(0, len(labels)):
        pl_list.append( tuple((preds[k],labels[k])) )

    # sort by preds value in descending order
    # because anomaly probability is proportional to abs(score) in case of scores < 0
    pl_list.sort(key=lambda tup: tup[0], reverse=True)

    # sort so as to get ranks
    for rank in range(0, N):
        if pl_list[rank][1] < 0:  # '-' label denotes anomalies
            Sa = Sa + rank

    return (Sa - Na*(Na+1)/2) / Na*Nn





### test block
#data = readNormData("../data_norm.txt")
#print data

#print EDist([1,2,3.4], [1,2,3.0])

#print np.mean(np.array([[0,0,0],[3,3,3],[4,4,4]]), axis=0) # along rows


