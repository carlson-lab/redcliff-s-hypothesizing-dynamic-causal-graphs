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

from general_utils.input_argument_utils import read_in_data_args, read_in_model_args
from general_utils.model_utils import create_model_instance, get_data_for_model_training, call_model_fit_method



def kick_off_model_training_experiment(args_dict, resume_training=False):
    print("kick_off_model_training_experiment: START")
    print("kick_off_model_training_experiment: args_dict['model_type'] == ", args_dict["model_type"])
    print("kick_off_model_training_experiment: args_dict['data_set_name'] == ", args_dict["data_set_name"])
    
    save_folder_name = "_".join([
        str(args_dict["model_type"]), 
        str(args_dict["data_set_name"]), 
        "fc"+str(args_dict["coeff_dict"]["FORECAST_COEFF"]).replace(".","-"), 
        "fsc"+str(args_dict["coeff_dict"]["FACTOR_SCORE_COEFF"]).replace(".","-"), 
        "fcsc"+str(args_dict["coeff_dict"]["FACTOR_COS_SIM_COEFF"]).replace(".","-")[:8], 
        "fwl1c"+str(args_dict["coeff_dict"]["FACTOR_WEIGHT_L1_COEFF"]).replace(".","-"), 
        "al1c"+str(args_dict["coeff_dict"]["ADJ_L1_REG_COEFF"]).replace(".","-")[:8], 
    ])
    curr_run_save_dir = args_dict["save_root_path"]+os.sep+save_folder_name
    args_dict["save_path"] = curr_run_save_dir
    print("kick_off_model_training_experiment: curr_run_save_dir == ", curr_run_save_dir, flush=True)
    if not os.path.exists(curr_run_save_dir):
        os.mkdir(curr_run_save_dir)
    elif "final_best_model.bin" in os.listdir(curr_run_save_dir):
        assert "training_meta_data_and_hyper_parameters.pkl" in os.listdir(curr_run_save_dir)
        resume_training = True
    else:
        for file in os.listdir(curr_run_save_dir):
            os.remove(curr_run_save_dir+os.sep+file)

    # load data
    [X_train, y_train, X_val, y_val] = get_data_for_model_training(args_dict, grid_search=False, dataset_category="DREAM4")
    args_dict["X_train"] = X_train
    args_dict["y_train"] = y_train
    args_dict["X_val"] = X_val
    args_dict["y_val"] = y_val
    
    # Set up model
    print("kick_off_model_training_experiment: initializing model")
    model = None
    if not resume_training:
        model = create_model_instance(args_dict, employ_version_with_smoothing_loss=True)
    else:
        model = torch.load(curr_run_save_dir+os.sep+"final_best_model.bin")
        model.resume_training_from_checkpoint(curr_run_save_dir+os.sep+"training_meta_data_and_hyper_parameters.pkl")
    assert model is not None

    # train the model
    call_model_fit_method(model, args_dict)
    print("kick_off_model_training_experiment: STOP")
    pass


def set_up_and_run_experiments(args_dict, files_of_cached_model_args, files_of_cached_data_args, POSSIBLE_MODEL_TYPES, POSSIBLE_DATA_SETS, shuffle_seed=0):
    MODEL_TYPES_IND = 0
    DATA_SETS_IND = 1

    parameters_to_be_parallelized = list(product(
        POSSIBLE_MODEL_TYPES, 
        POSSIBLE_DATA_SETS, 
    ))
    random.Random(shuffle_seed).shuffle(parameters_to_be_parallelized) # shuffle so that queued jobs do not favor any particular setting over another in terms of execution order
    print("set_up_and_run_experiments: TOTAL NUM OF TRAINING INSTANCES BEING INITIATED IS len(parameters_to_be_parallelized) == ", len(parameters_to_be_parallelized))
    
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
    
    print("set_up_and_run_experiments: pre-model args read-in args_dict == ", args_dict)
    args_dict = read_in_model_args(args_dict)
    print("set_up_and_run_experiments: pre-data args read-in args_dict == ", args_dict)
    args_dict = read_in_data_args(args_dict)
    print("set_up_and_run_experiments: post args read-in args_dict == ", args_dict)
    
    print("set_up_and_run_experiments: OVER-WRITING FACTOR_COS_SIM_COEFF TO SCALE WITH SELECTED DATASET; SETTING IT TO ", args_dict["coeff_dict"]["FACTOR_COS_SIM_COEFF"]/sum([1.*i for i in range(1,args_dict["num_factors"])]))
    args_dict["coeff_dict"]["FACTOR_COS_SIM_COEFF"] = args_dict["coeff_dict"]["FACTOR_COS_SIM_COEFF"]/sum([1.*i for i in range(1,args_dict["num_factors"])])
    print("set_up_and_run_experiments: OVER-WRITING ADJ_L1_REG_COEFF TO SCALE WITH SELECTED PARAMETERS; SETTING IT TO ", args_dict["coeff_dict"]["ADJ_L1_REG_COEFF"]*(1./(1.*args_dict["num_factors"]))*(1./np.sqrt(args_dict["num_channels"]**2. - 1.)))
    args_dict["coeff_dict"]["ADJ_L1_REG_COEFF"] = args_dict["coeff_dict"]["ADJ_L1_REG_COEFF"]*(1./(1.*args_dict["num_factors"]))*(1./np.sqrt(args_dict["num_channels"]**2. - 1.))
    print("set_up_and_run_experiments: OVER-WRITING STOPPING-CRITERIA COEFFS TO MATCH LOSS COEFFS", flush=True)
    args_dict["stopping_criteria_forecast_coeff"] = args_dict["coeff_dict"]["FORECAST_COEFF"]
    args_dict["stopping_criteria_factor_coeff"] = args_dict["coeff_dict"]["FACTOR_SCORE_COEFF"]
    args_dict["stopping_criteria_cosSim_coeff"] = args_dict["coeff_dict"]["FACTOR_COS_SIM_COEFF"]
    
    print("set_up_and_run_experiments: post gc args over-write args_dict == ", args_dict)
    
    kick_off_model_training_experiment(args_dict)
    return taskID



if __name__=="__main__":
    parse=argparse.ArgumentParser(description='Default parameter grid search evaluation')
    parse.add_argument(
        "-cached_args_file",
        default="REDCLIFF_S_CMLP_Smooth_d4IC_BSCgs4ParSmo0_RespAblateUnsup_cached_args.txt", # WARNING: edit this line when running new experiments on SLURM server
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
        "REDCLIFF_S_CMLP", 
    ]
    files_of_cached_model_args = [
        "REDCLIFF_S_CMLP_Smooth_d4IC_BSCgs4ParSmo0_RespAblateUnsup_cached_args.txt",
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

    taskID = set_up_and_run_experiments(
        args_dict, 
        files_of_cached_model_args, 
        files_of_cached_data_args, 
        POSSIBLE_MODEL_TYPES, 
        POSSIBLE_DATA_SETS, 
    )
    
    print("__MAIN__: DONE RUNNING TASKID == ", taskID,"!!!")
    pass