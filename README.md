# Unsupervised Anomaly Detection

#### introduction
Multiple Unsupervised approaches to Detection in multivariate time series data.

#### dependencies
- python 3 (python 2 may be compatible but no guarantee)
- numpy-1.15.4
- scipy-1.2.0
- scikit-learn-0.20.2
- matplotlib-3.0.2
- pyyaml-3.13

#### code structure
    |Unsupervised-Anomaly-Detection
        |common
            |clean_data.py    # APIs for data preprocessing
            |common_funcs.py  # common APIs for reading/saving/extracting/formating data, etc.
            |show_decisions.py   # A simple API for showing detection results
        |detect_algos
            |EDK-means
                |data_norm.txt       # identical to that in preprocessed_data
                |data_std.txt        # identical to that in preprocessed_data
                |detect_EDK-means.py   # fitting and training APIs for K-means
                |parameters.yaml       # parameters of data input and model
                |saved_model.yaml    # persistence of the model on disk
                |train_res_EDKmeans.txt   # detections made in our training
            |isoForest
                |data_norm.txt       # identical to that in preprocessed_data
                |data_std.txt        # identical to that in preprocessed_data
                |detect_isoForest.py   # fitting and training APIs for Isolation Forest detector
                |parameters.yaml        # parameters of data input and model
                |saved_model.mdl     # persistence of the model on disk
                |train_res_isoforest.txt   # detections made in our training
            |OCSVM
                |data_norm.txt       # identical to that in preprocessed_data
                |data_std.txt        # identical to that in preprocessed_data
                |detect_isoForest.py   # fitting and training APIs for One Class SVM
                |parameters.yaml        # parameters of data input and model
                |saved_model.mdl     # persistence of the model on disk
                |train_res_OCSVM.txt   # detections made in our training
            |others
        |examples
            |example1   # an demonstration of how to detect anomalies using OCSVM detector
            |raw_log_files   # file containing raw log data
        |preprocessed_data
            |data_norm.txt    # aggregated, normalized data
            |data_std.txt     # aggregated, standardized data
        |raw_datasets
            |jiuzhouLog
                |event-xxx.txt   # raw data of log
        |training_set_decisions
            |pred_xxx.txt   # decisions made on our training process

