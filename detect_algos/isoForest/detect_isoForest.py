#!/usr/bin/env python
# detect_EDK-means.py
# The python script is used to detect abnormal frames in event time series using Isolation Forest.
# @Author  : wwt
# @Date    : 2018-12-25

import numpy as np
from common import common_funcs
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plot
import sys
import yaml
import pickle



def fit(train_file='data_std.txt', config='parameters.yaml', model_file='', slotting=True, plotting=False):
    """
    # train the model(s) on the given data set, within each time slot if enabled
    # the model will be saved to local disk as specified by the 'model_save_path' in cfg
    :param train_file: contains data to fit
    :param config: configuration file path
    :param model_file: model file containing pre-trained model(s), use the one specified in config if not given
    :param slotting: if True then load slotting configs from cfg(default = parameters.yaml), no slotting if False
    :param plotting: if True then plot the decisions on training data, no plotting if False
    :return: decision functions on training data
    """
    print("INFO@OCSVM: loading cfg file...")
    # load yaml config file
    with open(config, 'r') as cfg:
        params = yaml.load(cfg)

    ''' load input data config '''
    headlines = params['headlines']
    num_channels = params['num_channels']
    channel_fmt_list_py3 = []  # data format
    channel_fmt_list_py2 = []
    for ch in params['data_format_py3']:
        channel_fmt_list_py3.append(tuple(ch))
    for ch in params['data_format_py2']:
        channel_fmt_list_py2.append(tuple(ch))

    if train_file == '':  # use file path specified in config if not given as an argument
        train_file = params['data_file']
    data_rescaling = params['rescale']
    JZLogFrame_type = np.dtype(channel_fmt_list_py3)
    if sys.version_info[0] < 3:  # for py2
        JZLogFrame_type = np.dtype(channel_fmt_list_py2)

    # the range of data to learn
    start_hour_shift = 0
    training_data_range_limit = params['training_data_range_limit']

    ''' load slotting config '''
    slot_size = params['slot_size']
    if slot_size == -1 or not slotting:
        slot_size = 24
    num_models = 24 / slot_size if 24 % slot_size == 0 else 24 / slot_size + 1

    ''' load model params '''
    model_name = params['model_name']
    trees = params['trees']
    samples_tree = params['samples_tree']
    features_tree = params['features_tree']
    cr = params['cr']  # contamination rate
    decisions_save_path = params['decision_save_path']
    model_save_path = model_file if model_file else params['model_save_path']

    # read data from the cleaned, normalized/standardized data set
    train_data = common_funcs.read_data(train_file,
                                        skips=headlines,
                                        cols=tuple(range(num_channels + 1)),
                                        datatype=JZLogFrame_type)
    # determine left bound and right bound
    if training_data_range_limit == [-1, -1]:  # default setting in config file
        training_data_range_limit = [0, len(train_data)]
    else:
        training_data_range_limit[0] = max(training_data_range_limit[0], 0)  # left bound
        training_data_range_limit[1] = min(training_data_range_limit[1], len(train_data))  # right bound
    start_date = train_data[training_data_range_limit[0]][0]
    train_data = train_data[training_data_range_limit[0]: training_data_range_limit[1]]

    # print(train_data)
    print("INFO@IsoForest: Number of slots = %d" % num_models)
    print("INFO@IsoForest: Model training in each slot starts...")

    # slotting
    slots = common_funcs.get_fixed_slot_frame_sets(train_data, slot_size, True, 'date')
    # training_set_decisions set, should be temporally sequential from start_hour_shift to end_h
    glob_decisions_map = list(range(training_data_range_limit[0], training_data_range_limit[1]))
    model_set = []

    # fit/train model one by one for each slot
    model_id = 0
    for slot in slots:
        time_seq = np.array(slot)[:, 0].tolist()  # get timestamp sequence and transform it to hour-index sequence
        for k in range(len(time_seq)):
            time_seq[k] = common_funcs.count_hours_from_str(time_seq[k])  # convert stamp to absolute time
            time_seq[k] -= common_funcs.count_hours_from_str(start_date)  # further trans to hour index starting from 0

        # invoke to train in-built iForest model
        # [1]Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation forest."
        #     Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on.
        # [2]Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation-based anomaly detection."
        #     ACM Transactions on Knowledge Discovery from Data (TKDD) 6.1 (2012): 3.
        #
        # @n_estimator number of isolation trees
        # @max_samples number of samples to be isolated in each tree, 'auto'=256(2**8) as demonstrated on p.9, [2]
        # @contamination contamination rate, directly determines the threshold
        # @max_features features used in a tree, optimal value=1 as shown in Fig.11, [2]
        # @bootstrap sampling with/without replacement, 'False'=without replacement
        # @behavior
        isoforest = IsolationForest(n_estimators=trees, max_samples='auto', contamination=cr,
                                          max_features=features_tree, bootstrap=False, behaviour='new')
        isoforest.fit(np.delete(np.array(slot), 0, 1))  # feed timestamp-stripped slot data

        # cache each slot-wise model
        model_set.append(isoforest)
        print("INFO@IsoForest: model id%s cached." % id(isoforest))
        model_id += 1

        # force minor class label to be '-1', and positive label '1'
        local_decisions_map = isoforest.decision_function(np.delete(np.array(slot), 0, 1)).tolist()

        # mapping training_set_decisions of this slot-local model to the global decision map
        for idx in range(len(time_seq)):
            glob_decisions_map[time_seq[idx]] = local_decisions_map[idx]  # store training_set_decisions

    # dump models to disk as binary file using pickle
    print("INFO@IsoForest: Dumping models into %s" % model_save_path)
    with open(model_save_path, 'wb') as msp:
        pickle.dump(model_set, msp)

    # save training data decisions to disk
    print("INFO@IsoForest: Dumping decisions of training data into %s" % decisions_save_path)
    decisions_with_stamps = np.array([train_data, glob_decisions_map]).T
    common_funcs.save_data(decisions_save_path, decisions_with_stamps, linenum=False)

    # plot decisions on training data
    if plotting:
        plot.scatter(range(training_data_range_limit[0], training_data_range_limit[1]), glob_decisions_map, s=1 ** 2)
        plot.hlines(y=0, xmin=training_data_range_limit[0], xmax=training_data_range_limit[1], linestyles='dashed')
        plot.title(model_name + "\ndecision map (+1/-1 denotes normality/anomaly)")
        plot.show()

    return decisions_with_stamps


def detect(test_file='', config='parameters.yaml', model_file='saved_model.mdl', plotting=True):
    """
    # Detect anomalies in the specified data from a file using pre-trained models
    :param test_file: data file containing test data in the format like:
        # time, event-crond, event-rsyslogd, event-session, event-sshd, event-su
        2018-06-29-00, 0.147, -0.223, 0.571, -0.594, 1.298
        2018-06-29-01, -0.215, -0.223, 0.696, -0.597, 1.443
    :param config: configuration file path, identical to the one used in training(fitting)
    :param model_file: model file containing pre-trained model(s)
    :param plotting: if True then plot the decisions on test data, no plotting if False
    :return: decisions on the test data
    """
    print("INFO@IsoForest: loading cfg file...")
    # load yaml config file
    with open(config, 'r') as cfg:
        params = yaml.load(cfg)

    ''' load input data config '''
    headlines = params['headlines']
    num_channels = params['num_channels']
    channel_fmt_list_py3 = []  # data format
    channel_fmt_list_py2 = []
    for ch in params['data_format_py3']:
        channel_fmt_list_py3.append(tuple(ch))
    for ch in params['data_format_py2']:
        channel_fmt_list_py2.append(tuple(ch))

    if test_file == '':  # use file path specified in config if not given as an argument
        test_file = params['data_file']
    data_rescaling = params['rescale']
    JZLogFrame_type = np.dtype(channel_fmt_list_py3)
    if sys.version_info[0] < 3:  # for py2
        JZLogFrame_type = np.dtype(channel_fmt_list_py2)

    ''' load slotting config '''
    slot_size = params['slot_size']
    if slot_size == -1:
        slot_size = 24
    num_models = 24 / slot_size if 24 % slot_size == 0 else 24 / slot_size + 1

    ''' load model params '''
    model_name = params['model_name']
    trees = params['trees']
    samples_tree = params['samples_tree']
    features_tree = params['features_tree']
    cr = params['cr']  # contamination rate
    decisions_save_path = params['decision_save_path']
    model_save_path = params['model_save_path']

    test_data_range_lim = params['test_data_range_limit']
    # read data from the cleaned, normalized/standardized data set
    test_data = common_funcs.read_data(test_file,
                                       skips=headlines,
                                       cols=tuple(range(num_channels + 1)),
                                       datatype=JZLogFrame_type)
    # determine left bound and right bound
    if test_data_range_lim == [-1, -1]:  # default setting in config file
        test_data_range_lim = [0, len(test_data)]
    else:
        test_data_range_lim[0] = max(test_data_range_lim[0], 0)  # left bound
        test_data_range_lim[1] = min(test_data_range_lim[1], len(test_data))  # right bound
    start_date = test_data[test_data_range_lim[0]][0]
    end_date = test_data[test_data_range_lim[1] - 1][0]
    test_data = test_data[test_data_range_lim[0]: test_data_range_lim[1]]

    # load model(s) from file
    print("INFO@IsoForest: loading model(s)...")
    with open(model_file, 'rb') as mdl:
        models = pickle.load(mdl)
    print("INFO@IsoForest: %d model(s) are loaded into memory..." % len(models))

    # slotting
    slot_id = 0
    slots = common_funcs.get_fixed_slot_frame_sets(test_data, slot_size, True, 'date')
    glob_decisions_map = list(range(test_data_range_lim[0], test_data_range_lim[1]))  # decisions map
    for slot in slots:
        time_seq = np.array(slot)[:, 0].tolist()  # get timestamp sequence and transform it to hour-index sequence
        for k in range(len(time_seq)):
            time_seq[k] = common_funcs.count_hours_from_str(time_seq[k])  # convert stamp to absolute time
            time_seq[k] -= common_funcs.count_hours_from_str(start_date)  # further trans to hour index starting from 0

        # load the corresponding slot-wise model
        isoforest = models[slot_id]

        # make decisions for test data in this slot
        local_decisions_map = isoforest.decision_function(np.delete(np.array(slot), 0, 1)).tolist()  # timestamp stripped

        # mapping training_set_decisions of this slot-local model to the global decision map
        for idx in range(len(time_seq)):
            glob_decisions_map[time_seq[idx]] = local_decisions_map[idx]  # store training_set_decisions
        # ready for the next slot
        slot_id += 1

    # plot results
    if plotting:
        plot.scatter(range(test_data_range_lim[0], test_data_range_lim[1]), glob_decisions_map, s=1 ** 2)
        plot.hlines(y=0, xmin=test_data_range_lim[0], xmax=test_data_range_lim[1], linestyles='dashed')
        plot.title(model_name + "\ndecision map (+/- denotes normality/anomaly)")
        plot.xlabel('Timeline(hour)')
        plot.text(test_data_range_lim[0], 0, start_date, rotation=90)
        plot.text(test_data_range_lim[1], 0, end_date, rotation=90)
        plot.show()

    print("INFO@IsoForest: detection finished, decisions returned...")
    return glob_decisions_map


def get_model_params(model_file='saved_model.mdl'):
    """
    # return model parameters
    :param model_file: the file path where the model is saved
    :return: model parameters
    """
    params = {}
    with open(model_file, 'rb') as mf:
        models = pickle.load(mf)

    params['num_models(slots)'] = len(models)
    params['models'] = models[0].get_params()

    return params


### test examples
#fit('data_std.txt', 'parameters.yaml', slotting=True, plotting=True)
#print(detect(test_file='data_std.txt'))
#print(get_model_params())


