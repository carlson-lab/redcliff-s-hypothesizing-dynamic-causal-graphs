import torch
import numpy as np
from torch.utils.data import DataLoader
import random
import shutil
import os
import pickle as pkl

from general_utils.plotting import plot_all_signal_channels, plot_x_wavelet_comparisson
from general_utils.time_series import perform_wavelet_decomposition, construct_signal_approx_from_wavelet_coeffs, make_high_level_signal_features
from general_utils.misc import flatten_directed_spectrum_features
from general_utils.metrics import get_number_of_connected_components



# CLASSES AND SCRIPTS FOR LOADING/USING DREAM4 In-Silico DATASET(S) ######################################################################################

class NormalizedDREAM4Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, signal_format, shuffle=True, shuffle_seed=0, max_num_features_per_series=None, dirspec_params=None, grid_search=True):
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
        self.signal_format = signal_format
        self.max_num_features_per_series = max_num_features_per_series
        self.dirspec_params = dirspec_params
        self.freq_bins = None
        self.num_chans = None
        self.num_time_steps = None
        self.data = []
        files_to_load = [x for x in os.listdir(data_path) if "subset_" in x and ".pkl" in x and "metadata" not in x]

        print("NormalizedDREAM4Dataset.__init__: loading files from path == ", data_path)
        print("NormalizedDREAM4Dataset.__init__: files_to_load == ", files_to_load)

        num_samps = 0
        num_skipped_samps = 0
        summed_samps = None
        for i, file_name in enumerate(files_to_load):
            curr_data_path = os.sep.join([data_path, file_name])

            with open(curr_data_path, 'rb') as infile:
                new_samples = pkl.load(infile)

                for j in range(len(new_samples)):
                    curr_data_samp = new_samples[j][self.X_IND]
                    curr_samp_label = new_samples[j][self.Y_LABEL_IND]
                    
                    if not np.isnan(np.sum(curr_data_samp)): # case where curr_sample is accepted into dataset (to be normalized later)
                        self.data.append((curr_data_path, j))
                        if self.num_chans is None:
                            summed_samps = curr_data_samp
                            self.num_chans = curr_data_samp.shape[self.CHANNEL_DIM]
                            self.num_time_steps = curr_data_samp.shape[self.TEMPORAL_DIM]
                        else:
                            summed_samps = summed_samps + curr_data_samp
                        num_samps += 1
                        pass

                    else: # case where curr smaple is rejected due to nan values
                        print("NormalizedDREAM4Dataset.__init__: WARNING - SKIPPING SAMPLE LOCATED AT ", (curr_data_path, j), " ON ACCOUNT OF np.isnan FLAG!!!")
                        num_skipped_samps += 1
            pass
            
        print("NormalizedDREAM4Dataset.__init__: TOTAL NUM OF SKIPPED SAMPLES (due to np.isnan flags) IN DATA SET == ", num_skipped_samps)
        print("NormalizedDREAM4Dataset.__init__: TOTAL NUM SAMPLES IN LOADED DATASET == ", len(self.data))
        
        # create variables necessary for normalization - see https://stats.stackexchange.com/questions/327192/the-correct-way-to-normalize-time-series-data
        assert len(summed_samps.shape) == 2
        self.channel_means = np.sum(summed_samps, axis=self.TEMPORAL_DIM) / (1.*num_samps*self.num_time_steps) # see https://numpy.org/doc/stable/reference/generated/numpy.sum.html
        self.channel_means = self.channel_means.reshape(1, self.num_chans)
        self.channel_std_devs = None # we now compute the std dev for each channel - see https://www.mathsisfun.com/data/standard-deviation-formulas.html
        
        for i in range(len(self.data)):
            curr_data_samp_path = self.data[i][0]
            
            with open(curr_data_samp_path, 'rb') as infile:
                curr_data_samp = pkl.load(infile)[self.data[i][1]][self.X_IND]
                if i == 0:
                    self.channel_std_devs = (curr_data_samp - self.channel_means)**2.
                else:
                    self.channel_std_devs = self.channel_std_devs + (curr_data_samp - self.channel_means)**2.
            pass

        assert len(self.channel_std_devs.shape) == 2
        self.channel_std_devs = torch.from_numpy(np.sqrt(np.sum(self.channel_std_devs[:,:], axis=self.TEMPORAL_DIM) / (1.*num_samps*self.num_time_steps)))
        self.channel_std_devs = self.channel_std_devs.reshape(1, self.num_chans)

        print("NormalizedDREAM4Dataset.__init__: self.channel_means.shape == ", self.channel_means.shape)
        print("NormalizedDREAM4Dataset.__init__: self.channel_means == ", self.channel_means)
        print("NormalizedDREAM4Dataset.__init__: self.channel_std_devs == ", self.channel_std_devs)

        if shuffle:
            random.Random(shuffle_seed).shuffle(self.data)
        pass
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index, label_each_time_step=False, torch_sample=False):
        """
        Returned shape from __getitem__: x=(batch_size, num_time_steps, num_chans)
        """
        x, y = None, None
        with open(self.data[index][0], 'rb') as infile:
            curr_data = pkl.load(infile)[self.data[index][1]]
            x = curr_data[self.X_IND]
            if torch_sample:
                x = x.squeeze().float() # x is a torch tensor of shape (1, num_time_steps, num_channels) originally
            else:
                x = torch.from_numpy(x).squeeze().float() # x has shape (1, num_time_steps, num_channels) originally

            x = (x - self.channel_means) / self.channel_std_devs

            if "directed_spectrum" in self.signal_format:
                assert self.dirspec_params is not None
                high_level_features = make_high_level_signal_features(
                    x[:self.max_num_features_per_series,:],
                    fs=self.dirspec_params["fs"],#1000,
                    min_freq=self.dirspec_params["min_freq"],#0.0,
                    max_freq=self.dirspec_params["max_freq"],#55.0,
                    directed_spectrum=self.dirspec_params["directed_spectrum"],#False,
                    csd_params=self.dirspec_params["csd_params"],#{"detrend": "constant","window": "hann","nperseg": 512,"noverlap": 256,"nfft": None,},
                )
                if self.freq_bins is None:
                    self.freq_bins = high_level_features["freq"]
                    print("NormalizedDREAM4Dataset.__getitem__: len(self.freq_bins) == ", len(self.freq_bins))
                    print("NormalizedDREAM4Dataset.__getitem__: self.freq_bins == ", self.freq_bins)
                if "vanilla" in self.signal_format:
                    num_regions = high_level_features["dir_spec"][0,:,:,:].shape[0]
                    num_features = high_level_features["dir_spec"][0,:,:,:].shape[-1]
                    x = torch.from_numpy(np.reshape(high_level_features["dir_spec"][0,:,:,:], (num_regions*num_regions*num_features)))
                else:
                    x = torch.flatten(torch.from_numpy(flatten_directed_spectrum_features(high_level_features["dir_spec"][0,:,:,:])))
            elif "power_features" in self.signal_format:
                raise NotImplementedError()
            elif "flattened" in self.signal_format:
                assert self.max_num_features_per_series is not None
                assert self.max_num_features_per_series > 0
                x = torch.flatten(x[:self.max_num_features_per_series,:]) # try and reduce the size of x prior to flattening to facilitate memory allocation (in future steps)

            y = torch.from_numpy(curr_data[self.Y_LABEL_IND]).float()

            if len(y.size()) != 2 and label_each_time_step: # copy y label for every time point in x
                assert len(y.size()) == 1, y.size()
                y = y.repeat(x.size()[self.TEMPORAL_DIM],1).T
            elif len(y.size()) == 1 and y.size()[-1] != x.size()[self.TEMPORAL_DIM] and label_each_time_step: # multiple labels present that haven't been assigned to time points
                raise NotImplementedError("NormalizedDREAM4Dataset.__init__: CAN'T CURRENTLY HANDLE y.size() == "+str(y.size())+" when x.size() == "+str(x.size()))
        
        return x.float(), y.float()


def load_normalized_DREAM4_data(data_path, batch_size, signal_format="original", shuffle=True, shuffle_seed=0, dirspec_params=None, grid_search=True):
    data_loader = None
    if data_path is not None:
        data_set = NormalizedDREAM4Dataset(data_path, signal_format=signal_format, shuffle=shuffle, shuffle_seed=shuffle_seed, dirspec_params=dirspec_params, grid_search=grid_search)
        data_loader = DataLoader(data_set, batch_size=batch_size) # see https://blog.paperspace.com/dataloaders-abstractions-pytorch/
    return data_loader


def load_normalized_DREAM4_data_train_test_split(data_root_path, batch_size, signal_format="original", shuffle=True, shuffle_seed=0, train_portion=0.8, dirspec_params=None, grid_search=True):
    train_data_path = data_root_path+os.sep+"train"
    val_data_path = data_root_path+os.sep+"validation"

    if not os.path.exists(train_data_path):
        assert not os.path.exists(val_data_path)
        os.mkdir(train_data_path)
        os.mkdir(val_data_path)
        # this code references https://www.tutorialspoint.com/How-to-copy-files-from-one-folder-to-another-using-Python
        data_files = [x for x in os.listdir(data_root_path) if "subset_" in x and ".pkl" in x]
        train_files = data_files[:int(train_portion*len(data_files))]
        val_files = data_files[int(train_portion*len(data_files)):]
        for tfile in train_files:
            shutil.copy(data_root_path+os.sep+tfile, train_data_path+os.sep+tfile)
        for vfile in val_files:
            shutil.copy(data_root_path+os.sep+vfile, val_data_path+os.sep+vfile)
    
    train_set = NormalizedDREAM4Dataset(train_data_path, signal_format=signal_format, shuffle=shuffle, shuffle_seed=shuffle_seed, dirspec_params=dirspec_params, grid_search=grid_search)
    train_loader = DataLoader(train_set, batch_size=batch_size) # see https://blog.paperspace.com/dataloaders-abstractions-pytorch/
    val_set = NormalizedDREAM4Dataset(val_data_path, signal_format=signal_format, shuffle=shuffle, shuffle_seed=shuffle_seed, dirspec_params=dirspec_params, grid_search=grid_search)
    val_loader = DataLoader(val_set, batch_size=batch_size) # see https://blog.paperspace.com/dataloaders-abstractions-pytorch/
    return train_loader, val_loader
    

def load_normalized_DREAM4_data_train_test_split_as_matrices(data_root_path, signal_format="original flattened", shuffle=True, shuffle_seed=0, 
                                                            max_num_features_per_series=25, train_portion=0.8, dirspec_params=None, grid_search=True):
    train_data_path = data_root_path+os.sep+"train"
    val_data_path = data_root_path+os.sep+"validation"

    if not os.path.exists(train_data_path):
        assert not os.path.exists(val_data_path)
        os.mkdir(train_data_path)
        os.mkdir(val_data_path)
        # this code references https://www.tutorialspoint.com/How-to-copy-files-from-one-folder-to-another-using-Python
        data_files = [x for x in os.listdir(data_root_path) if "subset_" in x and ".pkl" in x]
        train_files = data_files[:int(train_portion*len(data_files))]
        val_files = data_files[int(train_portion*len(data_files)):]
        for tfile in train_files:
            shutil.copy(data_root_path+os.sep+tfile, train_data_path+os.sep+tfile)
        for vfile in val_files:
            shutil.copy(data_root_path+os.sep+vfile, val_data_path+os.sep+vfile)
    
    train_set = NormalizedDREAM4Dataset(
        train_data_path, 
        signal_format=signal_format, 
        shuffle=shuffle, 
        shuffle_seed=shuffle_seed, 
        max_num_features_per_series=max_num_features_per_series, 
        dirspec_params=dirspec_params, 
        grid_search=grid_search
    )
    print("data.dream4_datasets.load_normalized_DREAM4_data_train_test_split_as_matrices: len(train_set) == ", len(train_set))
    X_train = None
    Y_train = None
    for i in range(len(train_set)):
        x, y = train_set[i]
        y = y.squeeze()
        if X_train is None:
            print("data.dream4_datasets.load_normalized_DREAM4_data_train_test_split_as_matrices: train x.size() == ", x.size())
            print("data.dream4_datasets.load_normalized_DREAM4_data_train_test_split_as_matrices: train y.size() == ", y.size())
            assert Y_train is None
            assert len(x.size()) == 1
            X_train = torch.zeros((len(train_set), len(x)))
            if len(y.size()) < 1:
                Y_train = torch.zeros((len(train_set), 1))
            else:
                assert len(y.size()) == 1
                Y_train = torch.zeros((len(train_set), len(y)))
        X_train[i,:] = X_train[i,:] + x
        if len(y.size()) >= 1:
            Y_train[i,:] = Y_train[i,:] + y
    
    val_set = NormalizedDREAM4Dataset(
        val_data_path, 
        signal_format=signal_format, 
        shuffle=shuffle, 
        shuffle_seed=shuffle_seed, 
        max_num_features_per_series=max_num_features_per_series, 
        dirspec_params=dirspec_params, 
        grid_search=grid_search
    )
    print("data.dream4_datasets.load_normalized_DREAM4_data_train_test_split_as_matrices: len(val_set) == ", len(val_set))
    X_val = None
    Y_val = None
    for i in range(len(val_set)):
        x, y = val_set[i]
        y = y.squeeze()
        if X_val is None:
            print("data.dream4_datasets.load_normalized_DREAM4_data_train_test_split_as_matrices: val x.size() == ", x.size())
            print("data.dream4_datasets.load_normalized_DREAM4_data_train_test_split_as_matrices: val y.size() == ", y.size())
            assert Y_val is None
            assert len(x.size()) == 1
            X_val = torch.zeros((len(val_set), len(x)))
            if len(y.size()) < 1:
                Y_val = torch.zeros((len(val_set), 1))
            else:
                assert len(y.size()) == 1
                Y_val = torch.zeros((len(val_set), len(y)))
        X_val[i,:] = X_val[i,:] + x
        if len(y.size()) >= 1:
            Y_val[i,:] = Y_val[i,:] + y
    
    return X_train.numpy(), Y_train.numpy(), X_val.numpy(), Y_val.numpy()


def load_normalized_DREAM4_data_train_test_split_as_tensors(data_root_path, signal_format="original", shuffle=True, shuffle_seed=0, 
                                                            max_num_features_per_series=None, train_portion=0.8, dirspec_params=None, grid_search=True):
    train_data_path = data_root_path+os.sep+"train"
    val_data_path = data_root_path+os.sep+"validation"

    if not os.path.exists(train_data_path):
        assert not os.path.exists(val_data_path)
        os.mkdir(train_data_path)
        os.mkdir(val_data_path)
        # this code references https://www.tutorialspoint.com/How-to-copy-files-from-one-folder-to-another-using-Python
        data_files = [x for x in os.listdir(data_root_path) if "subset_" in x and ".pkl" in x]
        train_files = data_files[:int(train_portion*len(data_files))]
        val_files = data_files[int(train_portion*len(data_files)):]
        for tfile in train_files:
            shutil.copy(data_root_path+os.sep+tfile, train_data_path+os.sep+tfile)
        for vfile in val_files:
            shutil.copy(data_root_path+os.sep+vfile, val_data_path+os.sep+vfile)
    
    train_set = NormalizedDREAM4Dataset(
        train_data_path, 
        signal_format=signal_format, 
        shuffle=shuffle, 
        shuffle_seed=shuffle_seed, 
        max_num_features_per_series=max_num_features_per_series, 
        dirspec_params=dirspec_params, 
        grid_search=grid_search
    )
    print("data.dream4_datasets.load_normalized_DREAM4_data_train_test_split_as_matrices: len(train_set) == ", len(train_set))
    X_train = None
    Y_train = None
    for i in range(len(train_set)):
        x, y = train_set[i]
        y = y.squeeze()
        if X_train is None:
            print("data.dream4_datasets.load_normalized_DREAM4_data_train_test_split_as_matrices: train x.size() == ", x.size())
            print("data.dream4_datasets.load_normalized_DREAM4_data_train_test_split_as_matrices: train y.size() == ", y.size())
            assert Y_train is None
            assert len(x.size()) == 2
            X_train = torch.zeros((len(train_set), x.size()[0], x.size()[1]))
            if len(y.size()) < 1:
                Y_train = torch.zeros((len(train_set), 1))
            else:
                assert len(y.size()) == 1
                Y_train = torch.zeros((len(train_set), len(y)))
        X_train[i,:,:] = X_train[i,:,:] + x
        if len(y.size()) >= 1:
            Y_train[i,:] = Y_train[i,:] + y
    
    val_set = NormalizedDREAM4Dataset(
        val_data_path, 
        signal_format=signal_format, 
        shuffle=shuffle, 
        shuffle_seed=shuffle_seed, 
        max_num_features_per_series=max_num_features_per_series, 
        dirspec_params=dirspec_params, 
        grid_search=grid_search
    )
    print("data.dream4_datasets.load_normalized_DREAM4_data_train_test_split_as_matrices: len(val_set) == ", len(val_set))
    X_val = None
    Y_val = None
    for i in range(len(val_set)):
        x, y = val_set[i]
        y = y.squeeze()
        if X_val is None:
            print("data.dream4_datasets.load_normalized_DREAM4_data_train_test_split_as_matrices: val x.size() == ", x.size())
            print("data.dream4_datasets.load_normalized_DREAM4_data_train_test_split_as_matrices: val y.size() == ", y.size())
            assert Y_val is None
            assert len(x.size()) == 2
            X_val = torch.zeros((len(val_set), x.size()[0], x.size()[1]))
            if len(y.size()) < 1:
                Y_val = torch.zeros((len(val_set), 1))
            else:
                assert len(y.size()) == 1
                Y_val = torch.zeros((len(val_set), len(y)))
        X_val[i,:,:] = X_val[i,:,:] + x
        if len(y.size()) >= 1:
            Y_val[i,:] = Y_val[i,:] + y
    
    return X_train.numpy(), Y_train.numpy(), X_val.numpy(), Y_val.numpy()