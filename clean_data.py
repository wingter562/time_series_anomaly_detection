#!/usr/bin/env python
# clean_data.py
# The python script is used to present views of the event sequences, after performing data cleaning
# @Author  : wwt
# @Date    : 2018-10-19

import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import matplotlib.pyplot as plt


# Count how many hours past so far for a given time point from Jan. 1 that year
def countHours(year, month, day, hour):
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


# DATA CLEANING v.1
# read data from the raw file
# remove daily/monthly summary values
# to finally build a sequence (series)
def buildSeq_v1(fname):
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


# DATA CLEANING v.2
# read data from the raw file
# remove daily/monthly summary values
# to finally build a sequence (series)
def buildSeq_v2(fname):
    # designed for files like 'datasets/jiuzhouLog/event-crond.txt'
    # for stripping
    start_idx = -1
    end_idx = -1

    with open(fname, 'rb') as f:
        # parse head line
        name = f.readline().split('\t')[0];
        print "Cleaning event log: %s" % name

        # build an all-zero sequence, hourly-based, one year long (enough in our case)
        seq = [0] * 366 * 24

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
            seq[idx] = count

            # for stripping
            end_idx = idx
            if start_idx == -1:
                start_idx = idx

            # move on to the next line
            line = f.readline()

    # strip the sequence and return
    # return seq[start_idx:end_idx]
    # return without stripping
    return seq


# align all sequences with a designated length of period
# @start start time point ("yyyy-mm-dd-hh")
# @end end time point ("yyyy-mm-dd-hh")
# @seq target sequence
def alignSeqs(start, end, seq):
    # parse times passed in
    y1, m1 ,d1, h1 = start.split('-')
    y1 = int(y1)
    m1 = int(m1)
    d1 = int(d1)
    h1 = int(h1)

    y2, m2, d2, h2 = end.split('-')
    y2 = int(y2)
    m2 = int(m2)
    d2 = int(d2)
    h2 = int(h2)
    start_idx = countHours(y1, m1 ,d1, h1)
    end_idx = countHours(y2, m2, d2, h2)

    return seq[start_idx:end_idx+1]



#np.loadtxt(, delimiter='/', skiprows=1)
#raw_event_rsyslogd = np.loadtxt(open('datasets/jiuzhouLog/event-rsyslogd.txt', 'rb'), delimiter='/', skiprows=1)


# Normalization
# @seq target sequence to rescale
def normalize(seq):
    scaler = MinMaxScaler()
    scaler.fit(seq)
    return scaler.transform(seq)

# z-score standardization
# @seq target sequence to rescale
def zStandardize(seq):
    scaler = StandardScaler()
    scaler.fit(seq)
    return scaler.transform(seq)


# save cleaned, aligned, normalized data to file
# @output_f file path to save at
def saveCleanedData(output_f, seq_mat):
    # output_f = "data_norm.txt"
    with open(output_f, 'w') as f:
        np.savetxt(output_f, seq_mat, delimiter=",", header="CROND, RSYSLOGD, SESSION, SSHD, SU")

# show data
def showData(seq_mat):
    # time-line
    t = np.arange(0, seq_mat.shape[0], 1)
    # use matplotlib to visualize the sequences
    plt.figure(1) # figure 1

    # event-CROND
    plt.subplot(321)
    plt.plot(t, seq_mat[:,0], 'k')
    plt.title("CROND")

    # event-RSYSLOGD
    plt.subplot(322)
    plt.plot(t, seq_mat[:,1], 'k')
    plt.title("RSYSLOGD")

    # event-SESSSION
    plt.subplot(323)
    plt.plot(t, seq_mat[:,2], 'k')
    plt.title("SESSION")

    # event-SSHD
    plt.subplot(324)
    plt.plot(t, seq_mat[:,3], 'k')
    plt.title("SSHD")

    # event-SU
    plt.subplot(325)
    plt.plot(t, seq_mat[:,4], 'k')
    plt.title("SU")

    plt.show()



# main

# built sequences & align them with given time scopes
begin_t = "2018-06-29-00"
end_t = "2018-11-20-00"

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

print seq_mat_norm
print seq_mat_std

# plot
showData(seq_mat_norm)
showData(seq_mat_std)

# save cleaned, normalized/standardized data in text file
#saveCleanedData("data_std.txt", seq_norm_std)
#saveCleanedData("data_std.txt", seq_mat_std)





