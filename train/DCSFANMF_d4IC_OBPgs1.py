import torch
import numpy as np
import random
import shutil
import os
import pickle as pkl
import argparse
import json
from itertools import product
import copy

import time

from general_utils.input_argument_utils import read_in_data_args, read_in_model_args 
from general_utils.model_utils import create_model_instance, get_data_for_model_training, call_model_fit_method
from general_utils.plotting import plot_reconstruction_comparisson



def kick_off_dCSFA_NMF_training_experiment(args_dict):
    print("kick_off_dCSFA_NMF_training_experiment: START")
    print("kick_off_dCSFA_NMF_training_experiment: args_dict['model_type'] == ", args_dict["model_type"])
    print("kick_off_dCSFA_NMF_training_experiment: args_dict['data_set_name'] == ", args_dict["data_set_name"])
    
    save_folder_name = "_".join([
        str(args_dict["model_type"]),
        str(args_dict["data_set_name"]),
    ])
    curr_run_save_dir = args_dict["save_root_path"]+os.sep+save_folder_name
    if not os.path.exists(curr_run_save_dir):
        os.mkdir(curr_run_save_dir)
    args_dict["save_path"] = curr_run_save_dir
    
    args_dict["dirspec_params"]["csd_params"]["nperseg"] = args_dict["num_node_features"]
    args_dict["dirspec_params"]["csd_params"]["noverlap"] = int(args_dict["num_node_features"]*0.5) # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html
    print("kick_off_dCSFA_NMF_training_experiment: using args_dict['dirspec_params']['csd_params'] == ", args_dict["dirspec_params"]["csd_params"])

    # load data
    print("kick_off_dCSFA_NMF_training_experiment: Loading data")
    [X_train, y_train, X_val, y_val] = get_data_for_model_training(args_dict, grid_search=False, dataset_category="DREAM4")
    args_dict["X_train"] = X_train
    args_dict["y_train"] = y_train
    args_dict["X_val"] = X_val
    args_dict["y_val"] = y_val

    print("kick_off_dCSFA_NMF_training_experiment: defining true_GC_tensor")
    GC = args_dict["true_GC_tensor"]

    print("kick_off_dCSFA_NMF_training_experiment: calling plot_all_signal_channels")
    x0 = X_train[0,:]
    n_flattened_features = len(x0)
    print("kick_off_dCSFA_NMF_training_experiment: sample0 origninal size is x0.shape == ", x0.shape)
    print("kick_off_dCSFA_NMF_training_experiment: sample0 origninal sum is np.sum(x0) == ", np.sum(x0))

    # Set up model
    print("kick_off_dCSFA_NMF_training_experiment: initializing dcsfa_nmf_model")
    dcsfa_nmf_model = create_model_instance(args_dict)

    # Train
    print("kick_off_dCSFA_NMF_training_experiment: training dcsfa_nmf_model")
    call_model_fit_method(dcsfa_nmf_model, args_dict)

    # save current param score
    print("kick_off_dCSFA_NMF_training_experiment: evaluating dcsfa_nmf_model")
    _ = dcsfa_nmf_model.evaluate(X_val, y_val, GC, args_dict["save_path"], threshold=False, ignore_features=True)
    
    print("kick_off_dCSFA_NMF_training_experiment: STOP")
    pass


def set_up_and_run_dCSFA_NMF_experiments(args_dict, files_of_cached_model_args, files_of_cached_data_args, POSSIBLE_MODEL_TYPES, POSSIBLE_DATA_SETS, shuffle_seed=0):
    MODEL_TYPES_IND = 0
    DATA_SETS_IND = 1

    parameters_to_be_parallelized = list(product(
        POSSIBLE_MODEL_TYPES, 
        POSSIBLE_DATA_SETS, 
    ))
    random.shuffle(parameters_to_be_parallelized) # shuffle so that queued jobs do not favor any particular setting over another in terms of execution order
    print("set_up_and_run_dCSFA_NMF_experiments: TOTAL NUM OF PARAMETER COMBOS BEING EXPLORED IS len(parameters_to_be_parallelized) == ", len(parameters_to_be_parallelized))
    
    taskID = int(os.environ['SLURM_ARRAY_TASK_ID'])
    task_param_settings = [x for x in parameters_to_be_parallelized[taskID-1]]

    args_dict["model_type"] = task_param_settings[MODEL_TYPES_IND]
    print("set_up_and_run_experiments: args_dict['model_type'] == ", args_dict["model_type"])
    print("set_up_and_run_experiments: [x for x in files_of_cached_model_args if args_dict['model_type'] in x] == ", [x for x in files_of_cached_model_args if args_dict["model_type"] in x])
    assert len([x for x in files_of_cached_model_args if args_dict["model_type"] in x]) == 1
    args_dict["model_cached_args_file"] = [x for x in files_of_cached_model_args if args_dict["model_type"] in x][0]
    
    args_dict["data_set_name"] = task_param_settings[DATA_SETS_IND]
    print("set_up_and_run_experiments: args_dict['data_set_name'] == ", args_dict["data_set_name"])
    print("set_up_and_run_experiments: [x for x in files_of_cached_data_args if args_dict['data_set_name'] in x] == ", [x for x in files_of_cached_data_args if args_dict["data_set_name"] in x])
    assert len([x for x in files_of_cached_data_args if args_dict["data_set_name"] in x]) == 1
    args_dict["data_cached_args_file"] = [x for x in files_of_cached_data_args if args_dict["data_set_name"] in x][0]
    
    args_dict = read_in_model_args(args_dict)
    args_dict = read_in_data_args(args_dict)
    
    print("set_up_and_run_experiments: pre-kickoff args_dict == ", args_dict)
    kick_off_dCSFA_NMF_training_experiment(args_dict)
    return taskID



if __name__=="__main__":
    parse=argparse.ArgumentParser(description='Default FullDCSFAModel parameter search')
    parse.add_argument(
        "-cached_args_file",
        default="DCSFANMF_d4IC_OBPgs1_cached_args.txt", # WARNING: edit this line when running new experiments on SLURM server
        help="/path/to/cached_args.txt should contain a json object with updated arg values",
    )
    args = parse.parse_args()

    # fix random seed(s) to 0 -- see https://discuss.pytorch.org/t/how-could-i-fix-the-random-seed-absolutely/45515
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic=True

    POSSIBLE_MODEL_TYPES = [
        "DCSFANMF", 
    ]
    files_of_cached_model_args = [
        "DCSFANMF_d4IC_OBPgs1_cached_args.txt",
    ]

    POSSIBLE_DATA_SETS = [
        "dream4_insilicoCombo_size10_LSNR_fold0",
        "dream4_insilicoCombo_size10_LSNR_fold1",
        "dream4_insilicoCombo_size10_LSNR_fold2",
        "dream4_insilicoCombo_size10_LSNR_fold3",
        "dream4_insilicoCombo_size10_LSNR_fold4",
        "dream4_insilicoCombo_size10_MSNR_fold0",
        "dream4_insilicoCombo_size10_MSNR_fold1",
        "dream4_insilicoCombo_size10_MSNR_fold2",
        "dream4_insilicoCombo_size10_MSNR_fold3",
        "dream4_insilicoCombo_size10_MSNR_fold4",
        "dream4_insilicoCombo_size10_HSNR_fold0",
        "dream4_insilicoCombo_size10_HSNR_fold1",
        "dream4_insilicoCombo_size10_HSNR_fold2",
        "dream4_insilicoCombo_size10_HSNR_fold3",
        "dream4_insilicoCombo_size10_HSNR_fold4",
    ]
    files_of_cached_data_args = [
        "cached_dataset_args/dream4_insilicoCombo_size10_LSNR_fold0_cached_args.txt",
        "cached_dataset_args/dream4_insilicoCombo_size10_LSNR_fold1_cached_args.txt",
        "cached_dataset_args/dream4_insilicoCombo_size10_LSNR_fold2_cached_args.txt",
        "cached_dataset_args/dream4_insilicoCombo_size10_LSNR_fold3_cached_args.txt",
        "cached_dataset_args/dream4_insilicoCombo_size10_LSNR_fold4_cached_args.txt",
        "cached_dataset_args/dream4_insilicoCombo_size10_MSNR_fold0_cached_args.txt",
        "cached_dataset_args/dream4_insilicoCombo_size10_MSNR_fold1_cached_args.txt",
        "cached_dataset_args/dream4_insilicoCombo_size10_MSNR_fold2_cached_args.txt",
        "cached_dataset_args/dream4_insilicoCombo_size10_MSNR_fold3_cached_args.txt",
        "cached_dataset_args/dream4_insilicoCombo_size10_MSNR_fold4_cached_args.txt",
        "cached_dataset_args/dream4_insilicoCombo_size10_HSNR_fold0_cached_args.txt",
        "cached_dataset_args/dream4_insilicoCombo_size10_HSNR_fold1_cached_args.txt",
        "cached_dataset_args/dream4_insilicoCombo_size10_HSNR_fold2_cached_args.txt",
        "cached_dataset_args/dream4_insilicoCombo_size10_HSNR_fold3_cached_args.txt",
        "cached_dataset_args/dream4_insilicoCombo_size10_HSNR_fold4_cached_args.txt",
    ]
    # the above yields 15 possible setting combinations
    
    print("__MAIN__: LOADING ARGS")
    with open(args.cached_args_file, 'r') as infile:
        load_dict = json.load(infile)
        save_root_path = load_dict["save_root_path"]
        args_dict = {"save_root_path": save_root_path, }
        
        args_dict["dirspec_params"] = {
            "fs": 100,
            "min_freq": 0.0,
            "max_freq": 50.0,
            "directed_spectrum": True,
            "csd_params": {
                "detrend": "constant",
                "window": "hann",
                "nperseg": None, # NOTICE: THIS WILL BE CHANGED LATER ON, DEPENDING ON task_id
                "noverlap": None, # NOTICE: THIS WILL BE CHANGED LATER ON, DEPENDING ON task_id
                "nfft": None,
            }, 
        }

    taskID = set_up_and_run_dCSFA_NMF_experiments(
        args_dict, 
        files_of_cached_model_args, 
        files_of_cached_data_args, 
        POSSIBLE_MODEL_TYPES, 
        POSSIBLE_DATA_SETS, 
    )
    
    print("__MAIN__: DONE RUNNING TASKID == ", taskID,"!!!")
    pass