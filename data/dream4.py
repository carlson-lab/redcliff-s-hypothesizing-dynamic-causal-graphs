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

def parse_orig_DREAM4_time_series_file(orig_ts_file, apply_state_perspective=False):
    print("dream4.parse_orig_DREAM4_time_series_file: START")
    print("dream4.parse_orig_DREAM4_time_series_file: orig_ts_file == ", orig_ts_file)
    POSSIBLE_NUM_CHANNELS = [10, 100]
    POSSIBLE_NUM_TIME_POINTS = [21]
    time_series_in_file = [] # will eventually be list of np arrays of shape (num_time_steps, num_channels)
    num_channels = None
    channel_ids = None
    num_time_points = None
    time_points = None
    state_labels = []
    meta_data = {
        "num_channels": num_channels, 
        "channel_ids": channel_ids, 
        "time_points": time_points, 
        "apply_state_perspective": apply_state_perspective, 
    }
    print("dream4.parse_orig_DREAM4_time_series_file: meta_data == ", meta_data)

    # load/parse the text file
    curr_loaded_time_series = None
    with open(orig_ts_file, 'r') as infile: # read in the original text-format time series - see https://stackoverflow.com/questions/48124206/iterate-through-a-file-lines-in-python
        all_lines = [line for line in infile] # see https://www.geeksforgeeks.org/count-number-of-lines-in-a-text-file-in-python/
        total_num_lines = len(all_lines)
        for i, line in enumerate(all_lines):
            print("dream4.parse_orig_DREAM4_time_series_file: line i == ", i, " is line == ", line)
            # ensure \n chars aren't counted
            if "\n" == line[-1]:
                line = line[:-1]
            
            if len(line) > 0:
                if i != 0:
                    curr_state_values = [float(v) for v in line.split("\t")]
                    curr_state = np.array(curr_state_values[1:]).reshape(1, num_channels) # don't record the first element (which denotes the time step at which a measurement was taken) in curr_state
                    curr_loaded_time_series.append(curr_state)
                    if num_time_points is None:
                        time_points.append(int(curr_state_values[0]))
                    if i == total_num_lines-1: 
                        curr_loaded_time_series = np.concatenate(curr_loaded_time_series, axis=0) # see https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
                        if apply_state_perspective:
                            time_series_in_file.append(curr_loaded_time_series[:(num_time_points//2)+1]) # the first half of the series has a different stimuli/perturbation that the latter half - see "Time series" discussion at https://www.synapse.org/#!Synapse:syn3049712/wiki/74633
                            state_labels.append(np.array([1, 0]))
                            time_series_in_file.append(curr_loaded_time_series[(num_time_points//2)+1:])
                            state_labels.append(np.array([0, 1]))
                        else:
                            time_series_in_file.append(curr_loaded_time_series)
                            state_labels.append(np.array([1, 0]))
                else:
                    header = line.split("\t")
                    header = [x[1:-1] for x in header] # remove \" chars from header elements
                    print("dream4.parse_orig_DREAM4_time_series_file: header == ", header)
                    assert header[0] == "Time"
                    channel_ids = header[1:]
                    num_channels = len(channel_ids)
                    assert num_channels in POSSIBLE_NUM_CHANNELS
                    meta_data["num_channels"] = num_channels
                    meta_data["channel_ids"] = channel_ids
                    curr_loaded_time_series = [] # list storing system states of shape np.zeros((1, 1, num_channels))
            elif i > 1: # ignore the line separating the header from the recordings and focus solely on breaks between recordings
                if num_time_points is None:
                    num_time_points = len(time_points)
                    assert num_time_points == len(curr_loaded_time_series)
                    assert num_time_points in POSSIBLE_NUM_TIME_POINTS
                    meta_data["num_time_points"] = num_time_points
                    meta_data["time_points"] = time_points

                curr_loaded_time_series = np.concatenate(curr_loaded_time_series, axis=0) # see https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
                
                if apply_state_perspective:
                    time_series_in_file.append(curr_loaded_time_series[:(num_time_points//2)+1]) # the first half of the series has a different stimuli/perturbation that the latter half - see "Time series" discussion at https://www.synapse.org/#!Synapse:syn3049712/wiki/74633
                    state_labels.append(np.array([1, 0]))
                    time_series_in_file.append(curr_loaded_time_series[(num_time_points//2)+1:])
                    state_labels.append(np.array([0, 1]))
                else:
                    time_series_in_file.append(curr_loaded_time_series)
                    state_labels.append(np.array([1, 0]))
                
                curr_loaded_time_series = [] # reset curr_loaded_time_series

            elif i == 1:
                time_points = []

    print("dream4.parse_orig_DREAM4_time_series_file: STOP")
    return time_series_in_file, state_labels, meta_data


def make_dream4_individual_preprocessed_dataset(orig_data_path, save_path, state_label_setting):
    print("dream4.make_dream4_individual_preprocessed_dataset: START")
    print("dream4.make_dream4_individual_preprocessed_dataset: orig_data_path == ", orig_data_path)
    print("dream4.make_dream4_individual_preprocessed_dataset: save_path == ", save_path)
    print("dream4.make_dream4_individual_preprocessed_dataset: state_label_setting == ", state_label_setting)
    if "size10_" in save_path:
        num_kfolds = 5 # the size10 networks have 5 recordings each in the dream4 dataset
    elif "size100_" in save_path:
        num_kfolds = 10 # the size100 networks have 5 recordings each in the dream4 dataset
    else:
        raise ValueError("Network Size must be stated as 10 or 100 in save_path")

    time_series_in_file, state_labels, meta_data = parse_orig_DREAM4_time_series_file(orig_data_path, apply_state_perspective=state_label_setting)
    print("dream4.make_dream4_individual_preprocessed_dataset: len(time_series_in_file) == ", len(time_series_in_file))
    print("dream4.make_dream4_individual_preprocessed_dataset: len(state_labels) == ", len(state_labels))
    print("dream4.make_dream4_individual_preprocessed_dataset: meta_data == ", meta_data)
    kfolds = make_kfolds_cv_splits(time_series_in_file, state_labels, num_folds=num_kfolds)
    for cv_id in range(num_kfolds):
        save_cv_split(kfolds[cv_id]["train"], kfolds[cv_id]["validation"], cv_id, save_path)
    print("dream4.make_dream4_individual_preprocessed_dataset: STOP")
    pass


def make_dream4_singleDominantSuperPositional_preprocessed_dataset(orig_data_path, save_path, state_label_setting):
    print("dream4.make_dream4_singleDominantSuperPositional_preprocessed_dataset: START")
    print("dream4.make_dream4_singleDominantSuperPositional_preprocessed_dataset: orig_data_path == ", orig_data_path)
    print("dream4.make_dream4_singleDominantSuperPositional_preprocessed_dataset: save_path == ", save_path)
    print("dream4.make_dream4_singleDominantSuperPositional_preprocessed_dataset: state_label_setting == ", state_label_setting)
    DOMINANT_NET_COEFF = 5
    BACKGROUND_NET_COEFF = 0.1
    print("dream4.make_dream4_singleDominantSuperPositional_preprocessed_dataset: WARNING - USING DEFAULT SUPERPOSITION SETTINGS OF DOMINANT_NET_COEFF == ", DOMINANT_NET_COEFF, " AND BACKGROUND_NET_COEFF == ", BACKGROUND_NET_COEFF)
    if "size10_" in save_path:
        num_kfolds = 5 # the size10 networks have 5 recordings each in the dream4 dataset
    elif "size100_" in save_path:
        num_kfolds = 10 # the size100 networks have 5 recordings each in the dream4 dataset
    else:
        raise ValueError("Network Size must be stated as 10 or 100 in save_path")
    
    # gather info from independent networks
    kfolds_by_network = []
    meta_data = []
    network_folders = os.listdir(orig_data_path)
    for net_folder in network_folders:
        curr_net_time_rec_file = [x for x in os.listdir(orig_data_path+os.sep+net_folder) if "_timeseries.tsv" in x]
        assert len(curr_net_time_rec_file)==1
        curr_net_time_rec_file = curr_net_time_rec_file[0]
        assert len(curr_net_time_rec_file) > len("_timeseries.tsv")

        curr_data_path = os.sep.join([orig_data_path, net_folder, curr_net_time_rec_file])
        curr_time_series_in_file, curr_state_labels, curr_meta_data = parse_orig_DREAM4_time_series_file(curr_data_path, apply_state_perspective=state_label_setting)
        curr_kfolds = make_kfolds_cv_splits(curr_time_series_in_file, curr_state_labels, num_folds=num_kfolds)
        kfolds_by_network.append(curr_kfolds)
        meta_data.append(curr_meta_data)
    with open(save_path+os.sep+"meta_data.pkl", 'wb') as outfile:
        pkl.dump(meta_data, outfile)

    # make superpositional dataset(s)
    for i, dominant_net_kfolds in enumerate(kfolds_by_network):
        dom_net_name = network_folders[i]
        curr_save_path = save_path+os.sep+dom_net_name
        os.mkdir(curr_save_path)

        # initialize superpositional kfolds with dominant network and scale recording accordingly
        superPositional_kfolds = copy.deepcopy(dominant_net_kfolds)
        for cv_id in range(num_kfolds):
            for element_num in range(len(superPositional_kfolds[cv_id]["train"])):
                superPositional_kfolds[cv_id]["train"][element_num][0] =  DOMINANT_NET_COEFF*superPositional_kfolds[cv_id]["train"][element_num][0]
            for element_num in range(len(superPositional_kfolds[cv_id]["validation"])):
                superPositional_kfolds[cv_id]["validation"][element_num][0] =  DOMINANT_NET_COEFF*superPositional_kfolds[cv_id]["validation"][element_num][0]

        # add background noise to kfold recordings
        for j, background_net_kfolds in enumerate(kfolds_by_network):
            if i != j:
                for cv_id in range(num_kfolds):
                    for element_num in range(len(superPositional_kfolds[cv_id]["train"])):
                        superPositional_kfolds[cv_id]["train"][element_num][0] =  superPositional_kfolds[cv_id]["train"][element_num][0] + BACKGROUND_NET_COEFF*background_net_kfolds[cv_id]["train"][element_num][0]
                    for element_num in range(len(superPositional_kfolds[cv_id]["validation"])):
                        superPositional_kfolds[cv_id]["validation"][element_num][0] =  superPositional_kfolds[cv_id]["validation"][element_num][0] + BACKGROUND_NET_COEFF*background_net_kfolds[cv_id]["validation"][element_num][0]

        # save the current superpositional kfolds
        for cv_id in range(num_kfolds):
            save_cv_split(superPositional_kfolds[cv_id]["train"], superPositional_kfolds[cv_id]["validation"], cv_id, curr_save_path)
    print("dream4.make_dream4_singleDominantSuperPositional_preprocessed_dataset: STOP")
    pass


##########################################################################################################################################################
# SCRIPT(S) FOR PARALLELIZED PREPROCESSING OF DREAM4 DATASET #############################################################################################

def kick_off_preprocessing_run(orig_data_path, save_path, network_size, superPositional_setting, state_label_setting, data_subdir):
    print("kick_off_preprocessing_run: START")
    print("kick_off_preprocessing_run: orig_data_path == ", orig_data_path)
    print("kick_off_preprocessing_run: save_path == ", save_path)
    print("kick_off_preprocessing_run: network_size == ", network_size)
    print("kick_off_preprocessing_run: superPositional_setting == ", superPositional_setting)
    print("kick_off_preprocessing_run: state_label_setting == ", state_label_setting)
    print("kick_off_preprocessing_run: data_subdir == ", data_subdir)
    curr_net_id = None
    if superPositional_setting != "individual":
        assert len(data_subdir.split(os.sep)) == 1
    else:
        assert len(data_subdir.split(os.sep)) == 3
        assert data_subdir.split(os.sep)[1][-2:] in ["_1", "_2", "_3", "_4", "_5"] # ensure a specific in-silico subfolder is included in data_subdir
        assert "_timeseries.tsv" in data_subdir
        curr_net_id = data_subdir.split(os.sep)[1]
    orig_data_path = orig_data_path+os.sep+data_subdir

    # create the new directory for saving info to
    # see https://www.geeksforgeeks.org/python-os-path-exists-method/ and https://www.geeksforgeeks.org/python-os-mkdir-method/
    size_tag = "size"+str(network_size)
    superPos_tag = superPositional_setting
    label_tag = "withStateLabels" if state_label_setting else "noStateLabels"
    curr_run_folder_name = "_".join([size_tag,superPos_tag,label_tag])

    curr_run_root_save_dir = save_path + os.sep + curr_run_folder_name
    save_path = curr_run_root_save_dir
    if not os.path.exists(curr_run_root_save_dir):
        os.mkdir(save_path)
    save_path = curr_run_root_save_dir+os.sep+curr_net_id if curr_net_id is not None else curr_run_root_save_dir
    if curr_net_id is not None and not os.path.exists(curr_run_root_save_dir+os.sep+curr_net_id):
        os.mkdir(save_path)

    # train the upstream model
    if superPositional_setting == "individual":
        make_dream4_individual_preprocessed_dataset(orig_data_path, save_path, state_label_setting)
    elif superPositional_setting == "singleDominantSuperPositional":
        make_dream4_singleDominantSuperPositional_preprocessed_dataset(orig_data_path, save_path, state_label_setting)
    else:
        raise NotImplementedError()
    print("kick_off_preprocessing_run: STOP")
    pass

def set_up_and_run_dream4_preprocessing(orig_data_path, save_path, network_sizes, superPositional_settings, state_label_settings):
    NET_SIZE_IND = 0
    SUPER_POS_IND = 1
    LABEL_SET_IND = 2
    DATA_SUBDIR_IND = 3
    # create list of various combinations of network_sizes, superPositional_settings, state_label_settings
    parameters_to_be_parallelized_preview = list(product(network_sizes, superPositional_settings, state_label_settings))
    print("set_up_and_run_dream4_preprocessing: len(parameters_to_be_parallelized_preview) == ", len(parameters_to_be_parallelized_preview))

    # since we are interested in creating both non-superpositional AND superpositional versions of DREAM4 datasets, we need to match the right original data files 
    # to the right parameter combinations (in particular, superpositional preprocessing code must be able to see recordings from all independent networks at once)
    parameters_to_be_parallelized = []
    for i, params in enumerate(parameters_to_be_parallelized_preview):
        print("set_up_and_run_dream4_preprocessing: params i == ", i)
        print("set_up_and_run_dream4_preprocessing: \t params[SUPER_POS_IND] == ", params[SUPER_POS_IND])
        if params[SUPER_POS_IND] == "singleDominantSuperPositional": # if superpositional setting is selected, ensure the data directory passed is the parent of all relevant independent network recordings
            data_subdir = "DREAM4_InSilico_Size"+str(params[NET_SIZE_IND])
            print("set_up_and_run_dream4_preprocessing: \t SUPERPOSITIONAL data_subdir == ", data_subdir)
            parameters_to_be_parallelized.append([
                params[NET_SIZE_IND], 
                params[SUPER_POS_IND], 
                params[LABEL_SET_IND], 
                data_subdir
            ])
        elif params[SUPER_POS_IND] == "individual": # we are not making a superpositional dataset with the current setting, so the data directory passed should only refer to a single (independent) network recording
            task_folders = ["DREAM4_InSilico_Size"+str(params[NET_SIZE_IND])]
            print("set_up_and_run_dream4_preprocessing: \t INDIVIDUAL task_folders == ", task_folders)
            for j, tf in enumerate(task_folders):
                print("set_up_and_run_dream4_preprocessing: \t tf j == ", j)
                net_folders = [x for x in os.listdir(orig_data_path+os.sep+tf) if not os.path.isfile(orig_data_path+os.sep+tf+os.sep+x)]
                print("set_up_and_run_dream4_preprocessing: \t\t net_folders == ", net_folders)
                for k, nf in enumerate(net_folders): # we want to preprocess recordings from ALL independent networks, so we add a new setting combination for each network
                    print("set_up_and_run_dream4_preprocessing: \t\t nf k == ", k)
                    timeseries_file_name = [x for x in os.listdir(orig_data_path+os.sep+tf+os.sep+nf) if os.path.isfile(orig_data_path+os.sep+tf+os.sep+nf+os.sep+x) and "_timeseries.tsv" in x]
                    assert len(timeseries_file_name) == 1
                    timeseries_file_name = timeseries_file_name[0]
                    assert len(timeseries_file_name) > len("_timeseries.tsv")

                    data_subdir = tf+os.sep+nf+os.sep+timeseries_file_name
                    print("set_up_and_run_dream4_preprocessing: \t\t\t data_subdir == ", data_subdir)
                    parameters_to_be_parallelized.append([
                        params[NET_SIZE_IND], 
                        params[SUPER_POS_IND], 
                        params[LABEL_SET_IND], 
                        data_subdir
                    ])
        else:
            raise ValueError("Unrecognized superpositional format params[SUPER_POS_IND] == "+str(params[SUPER_POS_IND]))

    print("set_up_and_run_dream4_preprocessing: len(parameters_to_be_parallelized) == ", len(parameters_to_be_parallelized))
    taskID = int(os.environ['SLURM_ARRAY_TASK_ID'])
    print("set_up_and_run_dream4_preprocessing: KICKING OFF TASKID == ", taskID)
    task_param_settings = [orig_data_path, save_path]+[x for x in parameters_to_be_parallelized[taskID-1]]
    kick_off_preprocessing_run(*task_param_settings)
    return taskID


if __name__=="__main__":
    parse=argparse.ArgumentParser(description='Default .pkl/.npy data preprocessing')
    parse.add_argument(
        "-cached_args_file",
        default="dream4_cached_args.txt", # WARNING: edit this line when running new experiments on SLURM server
        help="/path/to/cached_args.txt should contain a json object with updated arg values",
    )
    args = parse.parse_args()

    # fix random seed(s) to 1337 -- see https://discuss.pytorch.org/t/how-could-i-fix-the-random-seed-absolutely/45515
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)
    torch.backends.cudnn.deterministic=True

    POSSIBLE_NETWORK_SIZES = [10, 100]
    POSSIBLE_SUPERPOSITIONAL_SETTINGS = ["individual", "singleDominantSuperPositional"]
    POSSIBLE_STATE_LABEL_SETTINGS = [True, False]

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
        POSSIBLE_STATE_LABEL_SETTINGS
    )
    
    print("__MAIN__: DONE RUNNING TASKID == ", taskID,"!!!")
    pass