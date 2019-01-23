#!/usr/bin/env python
# clean_data.py
# The python script is used to present views of the event sequences, after performing data cleaning
# by combining separate, raw event count files (e.g., ~/raw_datasets/raw_log_files/event-crond.txt) into
# an aggregated, normalized/standardized matrix (with each row as a frame containing timestamp and counts of events),
# which will finally be written to a disk file (e.g., ~/preprocessed_data/data_norm.txt)
# @Author  : wwt
# @Date    : 2018-10-19

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import common.common_funcs as common_funcs


''' Deprecated, use build_seq_v2 instead'''
def build_seq_v1(fname):
    """
    # DATA CLEANING v.1
    # read data from the raw file
    # remove daily/monthly summary values
    # to finally build a sequence (series)
    :param fname: raw log file
    :return:
    """
    # designed for files like 'raw_datasets/raw_log_files/event-crond.txt'
    with open(fname, 'rb') as file_crond:
        # parse head line
        name = file_crond.readline().split('\t')[0];
        print("Cleaning event log: %s" % name)

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


def build_seq_v2(fname, with_timestamp=False, headers=0):
    """
    # DATA CLEANING v.2
    # read data from the raw file in format: event-crond/2018-06-27/00h/90
    # remove daily/monthly summary values
    # to finally build a sequence (series)
    :param fname: raw log file
    :return: the built sequence, a list of tuples of (timestamp, count)
    """
    # designed for files like 'raw_datasets/raw_log_files/event-crond.txt'
    # for stripping
    start_idx = -1
    end_idx = -1

    with open(fname, 'r') as f:
        # skip head line
        for i in range(headers):
            f.readline()
        name = f.readline().split('/')[0]
        print("Cleaning event log: %s" % name)

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
            idx = common_funcs.count_hours_from_ints(year, month, day, hour)
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


def get_event_names_list(raw_data_files_list, headers=0):
    """
    # Extract event (channel) names and return a list of them
    :param raw_data_files_list: list of raw data files
    :return: a list of channel names, e.g., ['CROND', 'SSHD', ..., 'RSYSLOGD']
    """
    name_list = []

    # examine all raw data (channels)
    for file in raw_data_files_list:
        with open(file, 'r') as f:
            # skip head line
            for i in range(headers):
                f.readline()
            # from first line get starting time
            name = f.readline().split('/')[0]
            name_list.append(name)

    return name_list


def get_minimum_span(raw_data_files_list, headers=0):
    """
    # Examine all raw channels and determine the latest starting timestamp and earliest ending timestamp
    :param raw_data_files_list: a list of raw data files
    :return: a tuple of (start_time, end_time), e.g., ('2011-09-01-09', '2011-10-01-21')
    """
    global_lastest_start = -1
    global_lastest_start_str = ''
    global_earliest_end = 999999
    global_earliest_end_str = ''

    # examine all raw data (channels)
    for file in raw_data_files_list:
        with open(file, 'r') as f:
            # skip head line
            for i in range(headers):
                f.readline()
            # from first line get starting time
            lineone = f.readline()
            channel_start = lineone.split('/')[1] + '-' + lineone.split('/')[2][0:2]
            # from last line get ending time
            stamp = ''
            for line in f:
                stamp = line.split('/')[1] + '-' + line.split('/')[2][0:2]
            channel_end = stamp if stamp else channel_start

        # compare globally
        if common_funcs.count_hours_from_str(channel_start) >= global_lastest_start:
            global_lastest_start = common_funcs.count_hours_from_str(channel_start)
            global_lastest_start_str = channel_start
        if common_funcs.count_hours_from_str(channel_end) <= global_earliest_end:
            global_earliest_end = common_funcs.count_hours_from_str(channel_end)
            global_earliest_end_str = channel_end

    return global_lastest_start_str, global_earliest_end_str


def build_stamps(start, end):
    """
    # build a temporally continuous timestamp sequence from the given start to the given end (inclusive)
    :param start: starting timestamp, yyyy-mm-dd-hh
    :param end: ending timestamp, yyyy-mm-dd-hh
    :return: a continuous timestamp sequence
    """
    # check leap year
    days_in_month_reg_year = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # regular year, e.g., 2018
    days_in_month_leap_year = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # leap year, e.g., 2020
    if int(start.split('-')[0]) % 4 == 0:
        days_in_month = days_in_month_leap_year
    else:
        days_in_month = days_in_month_reg_year

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
    #e_year, e_month, e_day, e_hour = end.split('-')
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
                    if y % 4 == 0:  # stepping into a leap year
                        days_in_month = days_in_month_leap_year
                    elif y % 4 == 1:  # stepping into a regular year from a leap year
                        days_in_month = days_in_month_reg_year

        # update timestamp
        stamp = str(y) + '-' + formatted(m) + '-' + formatted(d) + '-' + formatted(h)

    # append the ending timestamp
    stamps_seq.append(end)

    return stamps_seq


def align_seqs(start, end, seq):
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
    start_idx = common_funcs.count_hours_from_ints(y1, m1, d1, h1)
    end_idx = common_funcs.count_hours_from_ints(y2, m2, d2, h2)

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

'''Deprecated, use save_data()@common_funcs.py module instead'''
def save_cleaned_data(output_f, seq_mat):
    """
    # save cleaned, aligned, normalized data to file, with 1st column as timestamps
    :param output_f: file path to save at
    :param seq_mat: data to save
    :return: N/A
    """
    # output_f = "data_norm.txt"
    with open(output_f, 'w+') as f:
        np.savetxt(output_f, seq_mat, delimiter=",", fmt='%s, %s, %s, %s, %s, %s',
                   header="time, CROND, RSYSLOGD, SESSION, SSHD, SU")


def show_data(seq_mat, start_h=0, range=999999):
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

'''The main API for data pre-processing'''
def preprocess_data(raw_files_list, out_fname, headers=0, rescale='std', begin_time='', end_time=''):
    """
    # Read raw data files (e.g., ~/raw_datasets/raw_log_files/event-crond.txt), combine them into an aggregated,
    # normalized/standardized matrix (with each row as a frame containing timestamp and counts of events) in memory,
    # and finally write to a specified disk file (e.g., ~/preprocessed_data/data_norm.txt)
    :param raw_files_list: raw data to be preprocessed, content should be in format: event-name/yyyy-mm-dd/hh/count
    for example:
        event-xxx/2018-06-27/00h/90
        event-xxx/2018-06-27/17h/23
        event-xxx/2018-06-27/18h/175
        ...
    :param out_fname: aggregated data matrix in format:
    for example:
        # time, CROND, RSYSLOGD, SESSION, SSHD, SU (header)
        2018-06-29-00, 0.829, 0.0, 0.796, 0.155, 0.884615
        2018-06-29-01, 0.804, 0.0, 0.805, 0.154, 0.903
        ...
    :param headers: number of headlines at the front
    :param rescale: apply Normalization ('norm') or (Z-scale) Standardization ('std') to data, default = 'std'
    :param begin_time: starting timestamp of the data after preprocessed,
        time span of raw data may be shrunk or extended,
        default = '', which means taking starting time of shortest channel
    :param end_time: ending timestamp of the data after preprocessed,
        time span of raw data may be shrunk or extended,
        default = '', which means taking ending time of shortest channel
    :return: preprocessed data, identical to that stored in the file 'out_fname'
    """
    print('Info@clean_data: data pre-processing starts...')
    # determine the time span if not given
    begin_t = begin_time
    end_t = end_time
    if begin_time == '' or end_time == '':
        begin_t, end_t = get_minimum_span(raw_files_list, headers=headers)

    # build a timestamps seq
    seq_time = build_stamps(begin_t, end_t)
    # convert it to a column
    time_column = np.array([seq_time]).T

    channels = []
    for file in raw_files_list:
        channels.append(align_seqs(begin_t, end_t, build_seq_v2(file, headers=headers)))

    # event-LOGIN, deprecated
    # seq_login =buildSeq_v2('raw_datasets/raw_log_files/event-login.txt')
    # event-NETWORKMANAGER, deprecated
    # seq_netman = buildSeq_v2('raw_datasets/raw_log_files/event-netman.txt')
    # missing months for event-USB, deprecated
    # seq_usb = buildSeq_v2('raw_datasets/raw_log_files/event-usb.txt')

    # build a matrix by removing timestamps in each seq
    channels_mat = np.array(channels, dtype=float)

    channels_mat = np.transpose(channels_mat)

    # normalize
    seq_mat_norm = normalize(channels_mat)
    # or, standardize
    seq_mat_std = zStandardize(channels_mat)

    # build the sequence matrix joint with timestamps
    # it becomes an matrix of str for numpy
    time_seq_mat_norm = np.concatenate((time_column, seq_mat_norm), axis=1)
    time_seq_mat_std = np.concatenate((time_column, seq_mat_std), axis=1)

    # print pure channels without stamps
    #print(seq_mat_std)
    #print(seq_mat_norm)
    # plot
    # showData(seq_mat_norm, start_h=0, range=3457)
    # showData(seq_mat_std, start_h=0, range=3457)

    # save cleaned, normalized/standardized data in text files
    names_list = get_event_names_list(raw_files_list, headers=0)
    if rescale == 'std':
        common_funcs.save_data(out_fname, time_seq_mat_std,
                               delim=',', headline='time, ' + ', '.join(names_list),
                               format='%s, %s, %s, %s, %s, %s', linenum=False)
        return seq_mat_std
    elif rescale == 'norm':
        common_funcs.save_data(out_fname, time_seq_mat_norm,
                               delim=',', headline='time, ' + ', '.join(names_list),
                               format='%s, %s, %s, %s, %s, %s', linenum=False)
        return seq_mat_norm



# test examples
# raw_files_list = ['../raw_datasets/raw_log_files/event-crond.txt',
#                  '../raw_datasets/raw_log_files/event-rsyslogd.txt',
#                  '../raw_datasets/raw_log_files/event-session.txt',
#                  '../raw_datasets/raw_log_files/event-sshd.txt',
#                  '../raw_datasets/raw_log_files/event-su.txt']
# print(preprocess_data(raw_files_list,
#                       '../preprocessed_data/data_norm.txt',
#                       0, 'norm', '2018-06-29-00', '2018-11-20-00'))






