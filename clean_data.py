#!/usr/bin/env python
# clean_data.py
# The python script is used to present views of the event sequences, after performing data cleaning
# @Author  : wwt
# @Date    : 2018-10-19

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt



def countHours(year, month, day, hour):
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


def buildSeq_v1(fname):
    """
    # DATA CLEANING v.1
    # read data from the raw file
    # remove daily/monthly summary values
    # to finally build a sequence (series)
    :param fname: raw log file
    :return:
    """
    # designed for files like 'datasets/jiuzhouLog/event-crond.txt'
    with open(fname, 'rb') as file_crond:
        # parse head line
        name = file_crond.readline().split('\t')[0];
        print "Cleaning event log: %s" % name

        # work on the data
        line = file_crond.readline()
        last_hour = -1
        seq = []
        while line:
            # parse line
            name, date, hour, count = line.split('/')
            count = int(count)
            # parse hour
            hour = int(hour[0:1])
            # parse date
            year, month, day = date.split('-')
            year = int(year)
            month = int(month)
            day = int(day)


            # first data line or no data missing at this interval
            if (last_hour == -1
                    or (hour-last_hour == 1 and day == last_day)
                    or (hour-last_hour == -23 and day-last_day == 1 and month == last_month)
                    or (hour-last_hour == -23 and day == 1 and month-last_month == 1)):
                seq.append(count)
            else:
                # otherwise count missing data and fill them up
                # case 1: missing hours in a day
                if day == last_day and month == last_month and year == last_year:
                    infill = [0] * (hour - last_hour -1) # e.g., add "07h/0" between "06h/3" and "08h/4"

                # case 2: missing hours across several days in a month
                if month == last_month and year == last_year:
                    infill = [0] * ((day-last_day)*24 + hour - last_hour -1)

                # case 3: missing hours across several months
                if year == last_year:
                    infill = [0] * ((day - last_day) * 24 + hour - last_hour - 1)

                seq.append(infill)
                seq.append(count)

            last_year = year
            last_month = month
            last_day = day
            last_hour = hour
            line = file_crond.readline()


def buildSeq_v2(fname):
    """
    # DATA CLEANING v.2
    # read data from the raw file in format: event-crond/2018-06-27/00h/90
    # remove daily/monthly summary values
    # to finally build a sequence (series)
    :param fname: raw log file
    :return: the built sequence, a list of tuples of (timestamp, count)
    """
    # designed for files like 'datasets/jiuzhouLog/event-crond.txt'
    # for stripping
    start_idx = -1
    end_idx = -1

    with open(fname, 'rb') as f:
        # parse head line
        name = f.readline().split('\t')[0];
        print "Cleaning event log: %s" % name

        # build an all-zero sequence, hourly-based, one year long (enough in our case)
        # resulting in a temporally continuous sequence
        seq = [0] * 366 * 24
        #stamps = ["yyyy-mm-dd-hh"] * 366 * 24

        # work on the data
        line = f.readline()
        while line:
            # split line
            name, date, hour, count = line.split('/')
            count = int(count)

            # parse hour
            hour = int(hour[0:2])

            # parse date
            year, month, day = date.split('-')
            year = int(year)
            month = int(month)
            day = int(day)

            # count hours and fill in the sequence
            idx = countHours(year, month, day, hour)
            #stamps[idx] = date + ':' + str(hour)
            seq[idx] = count

            # for stripping
            end_idx = idx
            if start_idx == -1:
                start_idx = idx

            # move on to the next line
            line = f.readline()

    # return the stamps and count sequence without stripping
    return seq


def buildStamps(start, end):
    """
    # build a temporally continuous timestamp sequence from the given start to the given end (inclusive)
    :param start: starting timestamp, yyyy-mm-dd-hh
    :param end: ending timestamp, yyyy-mm-dd-hh
    :return: a continuous timestamp sequence
    """
    days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];  # from Jan. to Dec., 2018

    # define time-increment functions
    incre_hour = lambda h: h+1 if h < 23 else 0
    incre_day = lambda d, m: d + 1 if d < days_in_month[m] else 1
    incre_month = lambda m: m + 1 if m < 12 else 1
    incre_year = lambda y: y + 1  # deprecated

    # define formatting functions
    formatted = lambda t: str(t) if t > 9 else '0' + str(t)

    stamps_seq = []
    y, m, d, h = start.split('-')
    y = int(y)
    m = int(m)
    d = int(d)
    h = int(h)
    e_year, e_month, e_day, e_hour = end.split('-')

    stamp = start
    while stamp != end:
        stamps_seq.append(stamp)
        # increment
        h = incre_hour(h)
        if h == 0:  # new day
            d = incre_day(d, m)
            if d == 1:
                m = incre_month(m)
                if m == 1:
                    y = incre_year(y)

        # update timestamp
        stamp = str(y) + '-' + formatted(m) + '-' + formatted(d) + '-' + formatted(h)

    # append the ending timestamp
    stamps_seq.append(end)

    return stamps_seq


def alignSeqs(start, end, seq):
    """
    # align all sequences within a designated length of period
    :param start: start time point ("yyyy-mm-dd-hh")
    :param end: end time point ("yyyy-mm-dd-hh")
    :param seq: target sequence
    :return: sequence excerpted
    """
    # parse times passed in
    y1, m1, d1, h1 = start.split('-')
    y1 = int(y1)
    m1 = int(m1)
    d1 = int(d1)
    h1 = int(h1)

    y2, m2, d2, h2 = end.split('-')
    y2 = int(y2)
    m2 = int(m2)
    d2 = int(d2)
    h2 = int(h2)
    start_idx = countHours(y1, m1, d1, h1)
    end_idx = countHours(y2, m2, d2, h2)

    return seq[start_idx:end_idx+1]


def normalize(seq):
    """
    # Normalization
    :param seq: target sequence to rescale
    :return: Min-Max Normalized sequence
    """
    scaler = MinMaxScaler()
    scaler.fit(seq)
    return scaler.transform(seq)


def zStandardize(seq):
    """
    # z-score standardization
    :param seq: target sequence to rescale
    :return: Z-score standardized sequence
    """
    scaler = StandardScaler()
    scaler.fit(seq)
    return scaler.transform(seq)


def saveCleanedData(output_f, seq_mat):
    """
    # save cleaned, aligned, normalized data to file, with 1st column as timestamps
    :param output_f: file path to save at
    :param seq_mat: data to save
    :return: N/A
    """
    # output_f = "data_norm.txt"
    with open(output_f, 'w') as f:
        np.savetxt(output_f, seq_mat, delimiter=",", fmt='%s, %s, %s, %s, %s, %s',
                   header="time, CROND, RSYSLOGD, SESSION, SSHD, SU")


def showData(seq_mat, start_h=0, range=999999):
    """
    # show data
    :param seq_mat: data to show
    :param start_h: beginning time stamp
    :param range:  maximum length of period, in hour
    :return: N/A
    """

    # data length and show range
    end_h = min(range, seq_mat.shape[0])
    # time-line
    t = np.arange(start_h, end_h, 1)
    # use matplotlib to visualize the sequences
    plt.figure(1) # figure 1

    # event-CROND
    plt.subplot(321)
    plt.plot(t, seq_mat[start_h:end_h, 0], 'k', linewidth=0.3, markersize=0.4)
    plt.title("CROND")

    # event-RSYSLOGD
    plt.subplot(322)
    plt.plot(t, seq_mat[start_h:end_h, 1], 'k', linewidth=0.3, markersize=0.4)
    plt.title("RSYSLOGD")
    plt.ylim(ymax=max(seq_mat[:, 1]))

    # event-SESSSION
    plt.subplot(323)
    plt.plot(t, seq_mat[start_h:end_h, 2], 'k', linewidth=0.3, markersize=0.4)
    plt.title("SESSION")

    # event-SSHD
    plt.subplot(324)
    plt.plot(t, seq_mat[start_h:end_h, 3], 'k', linewidth=0.3, markersize=0.4)
    plt.title("SSHD")

    # event-SU
    plt.subplot(325)
    plt.plot(t, seq_mat[start_h:end_h, 4], 'k', linewidth=0.3, markersize=0.4)
    plt.title("SU")

    plt.show()


''' main '''

# built sequences & align them with given time scopes
begin_t = "2018-06-29-00"
end_t = "2018-11-20-00"

# build timestamps seq
seq_time = buildStamps(begin_t, end_t)
# convert it to a column
time_column = np.array([seq_time]).T

# build one seq from each channel and align them
seq_crond = alignSeqs(begin_t, end_t, buildSeq_v2('datasets/jiuzhouLog/event-crond.txt'))
seq_rsyslogd = alignSeqs(begin_t, end_t, buildSeq_v2('datasets/jiuzhouLog/event-rsyslogd.txt'))
seq_session = alignSeqs(begin_t, end_t, buildSeq_v2('datasets/jiuzhouLog/event-session.txt'))
seq_sshd = alignSeqs(begin_t, end_t, buildSeq_v2('datasets/jiuzhouLog/event-sshd.txt'))
seq_su = alignSeqs(begin_t, end_t, buildSeq_v2('datasets/jiuzhouLog/event-su.txt'))

# event-LOGIN, deprecated
# seq_login =buildSeq_v2('datasets/jiuzhouLog/event-login.txt')
# event-NETWORKMANAGER, deprecated
# seq_netman = buildSeq_v2('datasets/jiuzhouLog/event-netman.txt')
# missing months for event-USB, deprecated
# seq_usb = buildSeq_v2('datasets/jiuzhouLog/event-usb.txt')

# build a matrix by removing timestamps in each seq
seq_mat = np.array([seq_crond,
                    seq_rsyslogd,
                    seq_session,
                    seq_sshd,
                    seq_su], dtype=float)

seq_mat = np.transpose(seq_mat)

# normalize
seq_mat_norm = normalize(seq_mat)
# or, standardize
seq_mat_std = zStandardize(seq_mat)

# build the sequence matrix joint with timestamps
# it becomes an matrix of str for numpy
time_seq_mat_norm = np.concatenate((time_column, seq_mat_norm), axis=1)
time_seq_mat_std = np.concatenate((time_column, seq_mat_std), axis=1)

# print pure channels without stamps
print seq_mat_std
print seq_mat_norm
# plot
#showData(seq_mat_norm, start_h=0, range=3457)
#showData(seq_mat_std, start_h=0, range=3457)


# save cleaned, normalized/standardized data in text file
#saveCleanedData("data_norm.txt", time_seq_mat_norm)
#saveCleanedData("data_std.txt", time_seq_mat_std)





