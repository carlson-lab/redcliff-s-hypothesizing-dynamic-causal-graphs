import torch
import numpy as np
from torch.utils.data import DataLoader
import random
import shutil
import os
import pickle as pkl
import argparse
import json
from itertools import product
import copy
import time

from general_utils.plotting import plot_all_signal_channels, plot_x_wavelet_comparisson
from general_utils.time_series import perform_wavelet_decomposition
from general_utils.input_argument_utils import parse_input_list_of_ints
from general_utils.misc import make_kfolds_cv_splits, save_cv_split




# CLASSES AND SCRIPTS FOR LOADING/USING DREAM4 In-Silico DATASET(S) ######################################################################################
class DREAM4Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, signal_format="original", shuffle=True, shuffle_seed=0, wavelet_params=None):
        """
        Notes: 
         - ...
        """
        super().__init__()
        self.TEMPORAL_DIM = 0
        self.CHANNEL_DIM = 1
        self.X_IND = 0
        self.Y_LABEL_IND = 1

        self.data_path = data_path
        assert signal_format in ["original", "wavelet_decomp"]
        self.signal_format = signal_format
        self.num_chans = None
        self.num_time_steps = None
        with open(data_path, 'rb') as infile:
            orig_samples = pkl.load(infile)
            self.data = [s[self.X_IND] for s in orig_samples]
            self.labels = [s[self.Y_LABEL_IND] for s in orig_samples]

        print("DREAM4Dataset.__init__: TOTAL NUM SAMPLES IN LOADED DATASET == ", len(self.data))

        if self.signal_format == "wavelet_decomp":
            assert wavelet_params is not None
            print("DREAM4Dataset.__init__: PERFORMING WAVELET DECOMPOSITIONS ON ORIGINAL TIME SERIES WITH SETTINGS wavelet_params == ", wavelet_params)
            self.wavelet_params = wavelet_params
            self.wavelet_type = wavelet_params["wavelet_type"]
            self.wavelet_decomp_level = wavelet_params["wavelet_decomp_level"]
            self.wavelet_decomposition_type = wavelet_params["wavelet_decomposition_type"]
            self.data = [perform_wavelet_decomposition(X, self.wavelet_type, self.wavelet_decomp_level, self.wavelet_decomposition_type) for X in self.data]

        if shuffle:
            data_inds = [i for i in range(len(self.data))]
            random.Random(shuffle_seed).shuffle(data_inds)
            self.data = self.data[data_inds]
            self.labels = self.labels[data_inds]
        pass
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returned shape from __getitem__: x=(batch_size, num_time_steps, num_chans)
        """
        x = torch.from_numpy(self.data[index]).squeeze().float() # x has shape (1, num_time_steps, num_channels) originally
        y = torch.from_numpy(self.labels[index]).float()
        return x, y


def load_DREAM4_data(data_path, batch_size, signal_format="original", shuffle=True, shuffle_seed=0, wavelet_params=None):
    data_set = DREAM4Dataset(data_path, signal_format=signal_format, shuffle=shuffle, shuffle_seed=shuffle_seed, wavelet_params=wavelet_params)
    data_loader = DataLoader(data_set, batch_size=batch_size) # see https://blog.paperspace.com/dataloaders-abstractions-pytorch/
    return data_loader
    
##########################################################################################################################################################
# CLASSES AND METHODS FOR CONVERTING ORIGINAL DREAM4 DATA FORMAT TO THAT NEEDED FOR THIS REPOSITORY ######################################################

def make_dream4_combo_dataset(orig_data_path, save_path, FOLD_ID, split_name, NUM_FACTORS, DOMINANT_COEFF, BACKGROUND_COEFF):
    print("make_dream4_combo_dataset: NOW MAKING ", split_name, " SPLIT")
    # load DREAM4 datasets (both train and val) that are relevant to FOLD_ID as factors
    factor_folders = [orig_data_path+os.sep+x+os.sep+"fold_"+str(FOLD_ID)+os.sep+split_name for x in os.listdir(orig_data_path) if os.path.exists(orig_data_path+os.sep+x+os.sep+"fold_"+str(FOLD_ID)+os.sep+split_name)]
    factor_sample_paths = []
    for x in factor_folders:
        factor_sample_paths.append([x+os.sep+y  for y in os.listdir(x) if "subset" in y and ".pkl" in y])
    
    assert len(factor_sample_paths) == NUM_FACTORS
    print("make_dream4_combo_dataset: factor_sample_paths == ", factor_sample_paths)
    
    orig_data_samples = []
    num_factor_samples = None
    for factor_id in range(NUM_FACTORS):
        factor_data = []
        for samp_folder in factor_sample_paths[factor_id]:
            with open(samp_folder, 'rb') as infile:
                orig_samples = pkl.load(infile)
                factor_data = factor_data + [s[0] for s in orig_samples]
        orig_data_samples.append(factor_data)

        if num_factor_samples is None: 
            num_factor_samples = len(factor_data)
        else:
            assert num_factor_samples == len(factor_data) # sanity check

        # report orig data len stats
        print("make_dream4_combo_dataset: factor ", factor_id," has ", len(factor_data), " original samples")

    combined_dataset = []
    for factor_id in range(NUM_FACTORS):
        for samp_id in range(num_factor_samples):
            # create combo sample
            x = DOMINANT_COEFF*orig_data_samples[factor_id][samp_id]
            for background_id in range(NUM_FACTORS):
                if background_id != factor_id:
                    x = x + BACKGROUND_COEFF*orig_data_samples[background_id][samp_id]

            # generate label
            y = np.zeros((NUM_FACTORS, 1))
            y[factor_id] += DOMINANT_COEFF
            y[:factor_id] += BACKGROUND_COEFF
            y[factor_id+1:] += BACKGROUND_COEFF

            if factor_id == 0 and samp_id == 0:
                print("make_dream4_combo_dataset: combined x.shape == ", x.shape)
                print("make_dream4_combo_dataset: combined y.shape == ", y.shape)

            # record combo sample and label
            combined_dataset.append([x,y])

    # SHUFFLE DATA
    random.shuffle(combined_dataset)
    
    # report new data len stats
    print("make_dream4_combo_dataset: combined_dataset has ", len(combined_dataset), " total samples")

    # save data in appropriate folder
    time.sleep(FOLD_ID) # pause to ensure race conditions do not interfere with directory update
    curr_split_save_dir = save_path+os.sep+split_name
    if not os.path.exists(curr_split_save_dir):
        os.mkdir(curr_split_save_dir)

    with open(curr_split_save_dir+os.sep+"subset_0.pkl", 'wb') as outfile:
        pkl.dump(combined_dataset, outfile)

    print("make_dream4_combo_dataset: FINISHED MAKING ", split_name, " SPLIT")
    pass


##########################################################################################################################################################
# SCRIPT(S) FOR PARALLELIZED PREPROCESSING OF DREAM4 DATASET #############################################################################################

def kick_off_preprocessing_run(orig_data_path, save_path, FOLD_ID, network_size, superPositional_setting, state_label_setting, 
                               NUM_FACTORS, NUM_FOLDS, DOMINANT_COEFF, BACKGROUND_COEFF):
    print("kick_off_preprocessing_run: START")
    # sanity checks
    assert FOLD_ID < NUM_FOLDS
    assert network_size == 10
    assert superPositional_setting == "singleDominantSuperPositional"
    assert state_label_setting == True
    assert NUM_FACTORS == 5
    
    # update the save_path parameter to reflect input settings
    save_folder_name = "_".join([
        "netSize"+str(network_size),
        str(superPositional_setting), 
        "labeled"+str(state_label_setting), 
        "nk"+str(NUM_FACTORS), 
        "dc"+str(DOMINANT_COEFF).replace(".","-"), 
        "bc"+str(BACKGROUND_COEFF).replace(".","-"), 
    ])
    time.sleep(FOLD_ID) # pause to ensure race conditions do not interfere with directory update
    curr_run_save_dir = save_path+os.sep+save_folder_name
    if not os.path.exists(curr_run_save_dir):
        os.mkdir(curr_run_save_dir)
    save_path = curr_run_save_dir

    save_folder_name = "_".join([
        "fold",
        str(FOLD_ID),
    ])
    time.sleep(FOLD_ID) # pause to ensure race conditions do not interfere with directory update
    curr_run_save_dir = save_path+os.sep+save_folder_name
    if not os.path.exists(curr_run_save_dir):
        os.mkdir(curr_run_save_dir)
    save_path = curr_run_save_dir

    # CONSTRUCT THE TRAINING SET
    make_dream4_combo_dataset(orig_data_path, save_path, FOLD_ID, "train", NUM_FACTORS, DOMINANT_COEFF, BACKGROUND_COEFF)
    
    # CONSTRUCT THE VAL SET (same as before, but with different data)
    make_dream4_combo_dataset(orig_data_path, save_path, FOLD_ID, "validation", NUM_FACTORS, DOMINANT_COEFF, BACKGROUND_COEFF)

    print("kick_off_preprocessing_run: STOP")
    pass

def set_up_and_run_dream4_preprocessing(orig_data_path, save_path, network_sizes, superPositional_settings, state_label_settings, 
                                        POSSIBLE_NUM_FACTORS, POSSIBLE_NUM_FOLDS, POSSIBLE_DOMINANT_COEFFS, POSSIBLE_BACKGROUND_COEFFS):
    NET_SIZE_IND = 0
    SUPER_POS_IND = 1
    LABEL_SET_IND = 2
    DATA_SUBDIR_IND = 3
    NUM_FACTORS_IND = 4
    NUM_FOLDS_IND = 5
    DOMINANT_COEFFS_IND = 6
    BACKGROUND_COEFFS_IND = 7

    # create list of various combinations of network_sizes, superPositional_settings, state_label_settings
    parameters_to_be_parallelized = list(product(
        network_sizes, 
        superPositional_settings, 
        state_label_settings, 
        POSSIBLE_NUM_FACTORS, 
        POSSIBLE_NUM_FOLDS, 
        POSSIBLE_DOMINANT_COEFFS, 
        POSSIBLE_BACKGROUND_COEFFS
    ))
    print("set_up_and_run_dream4_preprocessing: len(parameters_to_be_parallelized) == ", len(parameters_to_be_parallelized))

    taskID = int(os.environ['SLURM_ARRAY_TASK_ID'])
    print("set_up_and_run_dream4_preprocessing: KICKING OFF TASKID == ", taskID)

    assert len(POSSIBLE_NUM_FOLDS) == 1
    PARAMS_ID = int((taskID-1) // POSSIBLE_NUM_FOLDS[0])
    FOLD_ID = int((taskID-1) % POSSIBLE_NUM_FOLDS[0])
    print("set_up_and_run_dream4_preprocessing: PARAMS_ID == ", PARAMS_ID)
    print("set_up_and_run_dream4_preprocessing: FOLD_ID == ", FOLD_ID)

    task_param_settings = [orig_data_path, save_path]+[FOLD_ID]+[x for x in parameters_to_be_parallelized[PARAMS_ID]]

    kick_off_preprocessing_run(*task_param_settings)
    return taskID


if __name__=="__main__":
    parse=argparse.ArgumentParser(description='Default .pkl/.npy data preprocessing')
    parse.add_argument(
        "-cached_args_file",
        default="dream4_insilicoCombo_cached_args.txt", # WARNING: edit this line when running new experiments on SLURM server
        help="/path/to/cached_args.txt should contain a json object with updated arg values",
    )
    args = parse.parse_args()

    # fix random seed(s) to 5 -- see https://discuss.pytorch.org/t/how-could-i-fix-the-random-seed-absolutely/45515
    torch.manual_seed(5)
    torch.cuda.manual_seed(5)
    np.random.seed(5)
    random.seed(5)
    torch.backends.cudnn.deterministic=True

    POSSIBLE_NETWORK_SIZES = [10]
    POSSIBLE_SUPERPOSITIONAL_SETTINGS = ["singleDominantSuperPositional"]
    POSSIBLE_STATE_LABEL_SETTINGS = [True]
    POSSIBLE_NUM_FACTORS = [5]
    POSSIBLE_NUM_FOLDS = [5]
    POSSIBLE_DOMINANT_COEFFS = [10.0]
    POSSIBLE_BACKGROUND_COEFFS = [0.0, 0.1, 1.0]

    print("__MAIN__: BEGINNING DATA CURRATION")
    coeff_dict = None
    with open(args.cached_args_file, 'r') as infile:
        new_args_dict = json.load(infile)
        orig_data_path = new_args_dict["orig_data_path"]
        data_save_path = new_args_dict["data_save_path"]
    
    taskID = set_up_and_run_dream4_preprocessing(
        orig_data_path, 
        data_save_path, 
        POSSIBLE_NETWORK_SIZES, 
        POSSIBLE_SUPERPOSITIONAL_SETTINGS, 
        POSSIBLE_STATE_LABEL_SETTINGS, 
        POSSIBLE_NUM_FACTORS, 
        POSSIBLE_NUM_FOLDS, 
        POSSIBLE_DOMINANT_COEFFS, 
        POSSIBLE_BACKGROUND_COEFFS
    )
    
    print("__MAIN__: DONE RUNNING TASKID == ", taskID,"!!!")
    pass