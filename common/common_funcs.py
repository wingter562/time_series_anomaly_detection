# This module consists of common functions for anomaly detection algorithms.
# @Author  : wwt
# @Date    : 2018-12-20
#

import math as math
import numpy as np
import matplotlib as plt


def count_hours_from_ints(year, month, day, hour):
    """
    # Count how many hours past so far for a given time point from Jan. 1 that year
    :param year: the year
    :param month: the month
    :param day:  the day
    :param hour: the hour
    :return: number of hours past in the given year
    """
    # build a monthly day-counter first
    days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]; # from Jan. to Dec.
    # check leap year
    if year % 4 == 0:
        days_in_month[2] = 29

    # count days
    days = 0
    for k in range(0, month, 1):
        days += days_in_month[k]
    days = days + day - 1

    # count hours
    return days * 24 + hour


def count_hours_from_str(date):
    """
    # Count how many hours past so far for a given time point from Jan. 1 that year
    :param date: date in format "yyyy-mm-dd-hh"
    :return: number of hours past in the given year
    """
    y, m, d, h = date.split('-')
    return count_hours_from_ints(int(y), int(m), int(d), int(h))


def read_data(fname, skips=1, cols=(0, 1, 2, 3, 4, 5), datatype=float):
    """
    # read data from a cleaned data set, with column 0 supposed to be timestamp
    :param fname: data file path, abs path
    :param skips: num of lines to skip from beginning
    :param cols: indexes of columns of data to use, begins from 0
    :param datatype: the structure of a row in the data, containing several fields
    :return: data in np.array
    """
    # bytes would be return in python 3
    data = np.loadtxt(fname, dtype=datatype, delimiter=',', skiprows=skips, usecols=cols)  # OK for py2
    return data


def save_data(output_f, data, delim=',', headline='', linenum=True):
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


def get_hourly_frame_sets(data, with_timestamp=False):
    """
    # Divide the the dataset by hour and return a 24-hour-size list of it
    :param data: set of data in format: <timestamp, feature_1,...,feature_m>
    :param with_timestamp: if True, retain timestamp as the first column. Default = False
    :return: a list(size=24) of lists of tuples, with each tuple in format ( [timestamp], f1, f2,..., fm )
    """
    # hourly frame set is a set storing [frames_in_a_hour] frames with sample hour-stamp
    hourly_frame_set = []

    # hourly_frame_set_list is a 24-size list of hourly_frame_set
    hourly_frame_set_list = []
    for h in range(0,24,1):
        hourly_frame_set_list.append([])  # cannot append variable hourly_frame_set, it would be a pointer

    # dive into the data frame by frame
    for frame in data:
        stamp = frame[0]  # timestamp, read as np.bytes_ in python 3, but str in py 2
        # if type(stamp) == bytes or type(stamp) == np.bytes_:
        #     stamp = stamp.decode('utf-8')
        hour = int(stamp.split('-')[3])

        if with_timestamp:
            hourly_frame_set_list[hour].append((frame[0], frame[1], frame[2], frame[3], frame[4], frame[5]))
        else:  # add the stamp-stripped record to the corresponding set
            hourly_frame_set_list[hour].append((frame[1], frame[2], frame[3], frame[4], frame[5]))

    # convert each hourly set to numpy.ndarray
    # for h in range(0,24,1):
    #     hourly_frame_set_list[h] = np.array(hourly_frame_set_list[h])

    return hourly_frame_set_list


def get_fixed_slot_frame_sets(data, slot_size, with_timestamp=False, slot_sort_by="hour"):
    """
    # divide the time series data by time slots whose sizes are identical as specified. The last slot may be smaller
    # if 24 % slot_size != 0. Say, data is partitioned into [0, slot_size-1], [slot_size, 2*slot_size-1],...
    # [..., num_slots*slot_size-1].
    :param slot_size: the size of time slot, e.g. 3 hours
    :param data: raw data to be partitioned. Raw data is in format: <timestamp, feature_1,...,feature_m>
    :param with_timestamp: if True, retain timestamp as the first column. Default = False
    :param slot_sort_by: sort each slot by day/hour Default = 'date'
    :return: a list(size=24) of lists of tuples, with each tuple in format ( [timestamp], f1, f2,..., fm )
    """
    # the func to check whether an extra slot is needed
    #hasAnExtraSlot = lambda x: 1 if x > 0 else 0
    num_slots = math.ceil(24 / slot_size)  # num of slots totally
    fixed_slot_frame_sets = [[] for _ in range(0, num_slots)]  # initialize the list of lists

    last_slot = 24 % slot_size

    hourly_frame_sets = get_hourly_frame_sets(data, with_timestamp=True)
    # loop through the frame sets to add them into corresponding slots
    for h in range(0, 24, 1):
        slot_idx = math.floor(h/slot_size)
        fixed_slot_frame_sets[slot_idx] += hourly_frame_sets[h]

    # sort each slot. Nothing to do if slot_sort_by == 'hour'
    # '2011-09-01-09' ->2011090109
    date_to_int_for_comp = \
        lambda frame: int(frame[0].replace('-', ''))
    for slot in fixed_slot_frame_sets:
        if slot_sort_by == "date":
            slot.sort(key=date_to_int_for_comp)

    # strip timestamp if required
    if not with_timestamp:
        for slot_idx in range(0, len(fixed_slot_frame_sets)):
            fixed_slot_frame_sets[slot_idx] = np.delete(np.array(slot), 0, 1).tolist()

    return fixed_slot_frame_sets


def get_hourly_stats(data, stats_to_get='mean'):
    """
    # First get hourly frame sets by calling getHourlyFrameSetList(data),
    # then compute means & standard deviation in each set
    :param data: set of data in format: <timestamp, feature_1,...,feature_m>
    :param stats_to_get: stats to get - 'mean' or 'std_var'
    :return: hourly statistics, each of which is a list of num_features-dimensional vector
    # e.g., [mean_f1, mean_f2, mean_f3,..., mean_fm]
    """
    # hourly stats lists contain 24 * [f1_aggregated, f2_aggregated, f3_aggregated,...]
    hourly_means = []
    hourly_stdvars = []

    hourly_frameset_list = get_hourly_frame_sets(data);


    # get average, h=hour, c=channel
    for h in range(0, 24, 1):
        hourly_means.append(hourly_frameset_list[h].mean(axis=0))  # average over this hourly set for each feature
        hourly_stdvars.append(hourly_frameset_list[h].std(axis=0))  # without Bessel's correction

    # return hourly statistics, each of which is a list of num_features-dimensional vector
    # e.g., [mean_f1, mean_f2, mean_f3,..., mean_fm]
    stats_dict = {}
    stats_dict['mean'] = hourly_means
    stats_dict['std_var'] = hourly_stdvars

    return stats_dict.get(stats_to_get, 'N/A')


def show_hourly_avg(data):
    """
    # show channel averages by hours
    :param data: set of data in format: <timestamp, feature_1,...,feature_m>
    :return: N/A
    """
    means = get_hourly_stats(data, stats_to_get='mean')
    means = np.array(means)  # for easy slicing
    #print(means)

    tline = range(0, 24, 1)  # a day
    plt.figure(1)  # figure 1

    # event-CROND
    plt.subplot(321)
    plt.plot(tline, means[:, 0], 'k', linewidth=0.3, markersize=0.4)
    plt.xticks(np.arange(0, 24, 3))
    plt.title("CROND")

    # event-RSYSLOGD
    plt.subplot(322)
    plt.plot(tline, means[:, 1], 'm', linewidth=0.3, markersize=0.4)
    plt.xticks(np.arange(0, 24, 3))
    plt.title("RSYSLOGD")

    # event-SESSSION
    plt.subplot(323)
    plt.plot(tline, means[:, 2], 'c', linewidth=0.3, markersize=0.4)
    plt.xticks(np.arange(0, 24, 3))
    plt.title("SESSION")

    # event-SSHD
    plt.subplot(324)
    plt.plot(tline, means[:, 3], 'y', linewidth=0.3, markersize=0.4)
    plt.xticks(np.arange(0, 24, 3))
    plt.title("SSHD")

    # event-SU
    plt.subplot(325)
    plt.plot(tline, means[:, 4], 'g', linewidth=0.3, markersize=0.4)
    plt.xticks(np.arange(0, 24, 3))
    plt.title("SU")

    plt.show()


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
        print("Err@EDist: unequal dimensions of input vectors")
        return -1

    sum = 0
    for k in range(0, d_x):
        sum = sum + (x[k] - y[k])**2

    return math.sqrt(sum)


def calculate_AUC(labels, preds):
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
# JZLogFrame_type = np.dtype([('timestamp', 'S13'), ('CROND', 'f8'), ('RSYSLOGD', 'f8'),
#                             ('SESSION', 'f8'), ('SSHD', 'f8'), ('SU', 'f8')])
# data = readData("../data_std.txt", skips=1, cols=(0,1,2,3,4,5), datatype=JZLogFrame_type)
# #print getHourlyFrameSets(data, with_timestamp=True)[23]  # good
#
# print np.array(getFixedSlotFrameSets(data, 3, False, 'date')[0])  # good
#print EDist([1,2,3.4], [1,2,3.0])

#print np.mean(np.array([[0,0,0],[3,3,3],[4,4,4]]), axis=0) # along rows


