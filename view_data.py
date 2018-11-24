#!/usr/bin/env python
# view_data.py
# The python script is used to present views of the event sequences.
# @Author  : wwt
# @Date    : 2018-10-19

import numpy as np
import matplotlib.pyplot as plt


# Read data from csv and shape it
raw_data = np.loadtxt(open('raw_data\\new_raw_data.csv', 'rb'), delimiter=',', skiprows=1)

'''
# Standardization
scaler = MinMaxScaler()
scaler.fit(raw_data)
raw_data = scaler.transform(raw_data)
# print('standardized data:\n', raw_data)
'''

# time-line
t = np.arange(0, 24, 1);
# use matplotlib to visualize the sequences
plt.figure(1) # figure 1
# 2 rows, 1 column
plt.subplot(211)
plt.plot(t, seq1, 'k')
plt.subplot(212)
plt.plot(t, seq2, 'k')
plt.show()