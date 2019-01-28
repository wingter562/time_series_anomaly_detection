#!/usr/bin/env python
# example1.py
# This python script is a simple demonstration of our anomaly detection APIs.
# @Author  : wwt
# @Date    : 2019-01-22

import sys
# module path, i.e., the absolute directory path of Unsupervised-Anomaly-Detection/, which needs to be re-configured
# before running on your machine
project_path = "F:/wwt/projects/codes/Unsupervised-Anomaly-Detection"
sys.path.append(project_path)

import common.common_funcs as cf  # common APIs for reading/saving/extracting/formating data, etc.
import common.clean_data as clean
import detect_algos.OCSVM.detect_OCSVM as ocsvm  # our implementation of One Class SVM anomaly detector



# execute only if run as a script
if __name__ == "__main__":
    # locate raw files and this demo’s path prefix. Need to re-configure them on your local machine.
    demo_path = 'F:/wwt/projects/codes/Unsupervised-Anomaly-Detection/examples/example1/'
    raw_files_list = ['F:/wwt/projects/codes/Unsupervised-Anomaly-Detection/examples/raw_log_files/event-crond.txt',
                      'F:/wwt/projects/codes/Unsupervised-Anomaly-Detection/examples/raw_log_files/event-rsyslogd.txt',
                      'F:/wwt/projects/codes/Unsupervised-Anomaly-Detection/examples/raw_log_files/event-session.txt',
                      'F:/wwt/projects/codes/Unsupervised-Anomaly-Detection/examples/raw_log_files/event-sshd.txt',
                      'F:/wwt/projects/codes/Unsupervised-Anomaly-Detection/examples/raw_log_files/event-su.txt']
    # content in even-xxx.txt:
    #   event-xxx/2018-06-27/00h/90
    #   event-xxx/2018-06-27/17h/23
    #   event-xxx/2018-06-27/18h/175
    #   event-xxx/2018-06-27/19h/206
    #   event-xxx/2018-06-27/20h/206
    #   event-xxx/2018-06-27/21h/206
    #   event-xxx/2018-06-27/22h/206
    #   event-xxx/2018-06-27/23h/218
    #   ...
    cfg_path = demo_path + 'parameters.yaml'
    model_path = demo_path + 'saved_model.mdl'
    training_file_path = demo_path + 'data_std.txt'
    test_file_path = demo_path + 'data_std.txt'

    # pre-process data and save to '.data_std.txt'
    print(clean.preprocess_data(raw_files_list,
                                demo_path + 'data_std.txt',
                                headers=0, rescale='std', begin_time='2018-06-29-00', end_time='2018-11-20-00'))


    # train OCSVM on the first 3000 data records, as specified in the config file (parameters.yaml)
    # i.e., 2018-06-29 to 2018-10-31
    # after training, the model(s) will be saved to the path specified by model_file argument

    ocsvm.fit(train_file=training_file_path, config=cfg_path, model_file=model_path,
              slotting=True, plotting=False)

    # test OCSVM on the last 457 data records, as per the config file (parameters.yaml)
    # i.e., 2018-11-01 to 2018-11-20
    res = ocsvm.detect(test_file=test_file_path, config=cfg_path, model_file=model_path,
                       plotting=True)
    print(res)
