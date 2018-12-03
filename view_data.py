#!/usr/bin/env python
# view_data.py
# The python script is used to present views of the event sequences.
# @Author  : wwt
# @Date    : 2018-10-19

import numpy as np
import matplotlib.pyplot as plt


# Count how many hours past so far for a given date from Jan. 1 that year
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



# DATA CLEANING
# read data from raw files
# remove daily/monthly summary values
# to finally build a sequence (series)
# version 1
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


# DATA CLEANING
# read data from raw files
# remove daily/monthly summary values
# to finally build a sequence (series)
# version 2
def buildSeq_v2(fname, year):
    # designed for files like 'datasets/jiuzhouLog/event-crond.txt'
    with open(fname, 'rb') as file_crond:
        # parse head line
        name = file_crond.readline().split('\t')[0];
        print "Cleaning event log: %s" % name

        # build an all-zero sequence, hourly-based, one year long (enough in our case)
        seq = [0] * 366 * 24

        # work on the data
        line = file_crond.readline()
        while line:
            # split line
            name, date, hour, count = line.split('/')
            count = int(count)

            # parse hour
            hour = int(hour[0:1])

            # parse date
            year, month, day = date.split('-')
            year = int(year)
            month = int(month)
            day = int(day)

            # count hours and fill in the sequence
            idx = countHours(year, month, day, hour)
            seq[idx] = count




#np.loadtxt(, delimiter='/', skiprows=1)
#raw_event_rsyslogd = np.loadtxt(open('datasets/jiuzhouLog/event-rsyslogd.txt', 'rb'), delimiter='/', skiprows=1)

'''
# Standardization
scaler = MinMaxScaler()
scaler.fit(raw_data)
raw_data = scaler.transform(raw_data)
# print('standardized data:\n', raw_data)
'''

#print(raw_event_crond)

'''
# time-line
t = np.arange(0, 24, 1);
# use matplotlib to visualize the sequences
plt.figure(1) # figure 1
# 2 rows, 1 column
plt.subplot(211)
plt.plot(t, raw_event_crond, 'k')
plt.subplot(212)
plt.plot(t, raw_event_rsyslogd, 'k')
plt.show()
'''


