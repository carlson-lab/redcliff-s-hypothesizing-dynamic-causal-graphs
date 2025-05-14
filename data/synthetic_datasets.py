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



# CLASSES AND SCRIPTS FOR LOADING/USING SYNNTHETIC DATASET(S) ######################################################################################

class NormalizedSyntheticWVARDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, signal_format="original", shuffle=True, shuffle_seed=0, max_num_features_per_series=None, dirspec_params=None, grid_search=True, pad_X=None):
        """
        Notes: 
         - pad_X (default: None):: a list of 4 elements (front_pad, back_pad, top_pad, bottom_pad)
        """
        super().__init__()
        self.TEMPORAL_DIM = 0
        self.CHANNEL_DIM = 1
        self.X_IND = 0
        self.X_WAV_DECOMP_IND = 1
        self.X_APPROX_IND = 2
        self.Y_LABEL_IND = 3

        self.data_path = data_path
        self.signal_format = signal_format
        self.max_num_features_per_series = max_num_features_per_series
        self.dirspec_params = dirspec_params
        self.pad_X = pad_X
        self.freq_bins = None
        self.num_chans = None
        self.num_time_steps = None
        self.data = []
        files_to_load = [x for x in os.listdir(data_path) if ("_subset" in x or "subset_" in x) and ".pkl" in x and "metadata" not in x]

        print("NormalizedSyntheticWVARDataset.__init__: loading files from path == ", data_path)
        print("NormalizedSyntheticWVARDataset.__init__: files_to_load == ", files_to_load)
        
        num_samps = 0
        num_skipped_samps = 0
        summed_samps = None
        for i, file_name in enumerate(files_to_load):
            curr_data_path = os.sep.join([data_path, file_name])

            with open(curr_data_path, 'rb') as infile:
                new_samples = pkl.load(infile)

                for j in range(len(new_samples)):
                    curr_data_samp = None
                    if "original" in self.signal_format:
                        curr_data_samp = new_samples[j][self.X_IND]
                    elif "wavelet_decomp" in self.signal_format:
                        curr_data_samp = new_samples[j][self.X_WAV_DECOMP_IND]
                    elif "approximation" in self.signal_format:
                        curr_data_samp = new_samples[j][self.X_APPROX_IND]
                    else:
                        raise ValueError("NormalizedSyntheticWVARDataset.__init__: Unrecognized signal format == ", self.signal_format)

                    curr_samp_label = new_samples[j][self.Y_LABEL_IND]
                    
                    if not np.isnan(np.sum(curr_data_samp)): # case where curr_sample is accepted into dataset (to be normalized later)
                        self.data.append((curr_data_path, j))
                        if len(curr_data_samp.shape) > 2:
                            curr_data_samp = curr_data_samp[0,:,:] # remove extra dimension
                        if self.num_chans is None:
                            summed_samps = curr_data_samp
                            self.num_chans = curr_data_samp.shape[self.CHANNEL_DIM]
                            self.num_time_steps = curr_data_samp.shape[self.TEMPORAL_DIM]
                        else:
                            summed_samps = summed_samps + curr_data_samp
                        num_samps += 1
                        pass

                    else: # case where curr smaple is rejected due to nan values
                        print("NormalizedSyntheticWVARDataset.__init__: WARNING - SKIPPING SAMPLE LOCATED AT ", (curr_data_path, j), " ON ACCOUNT OF np.isnan FLAG!!!")
                        num_skipped_samps += 1
            pass

        print("NormalizedSyntheticWVARDataset.__init__: TOTAL NUM OF SKIPPED SAMPLES (due to np.isnan flags) IN DATA SET == ", num_skipped_samps)
        print("NormalizedSyntheticWVARDataset.__init__: TOTAL NUM SAMPLES IN LOADED DATASET == ", len(self.data))
        
        # create variables necessary for normalization - see https://stats.stackexchange.com/questions/327192/the-correct-way-to-normalize-time-series-data
        assert len(summed_samps.shape) == 2
        self.channel_means = np.sum(summed_samps, axis=self.TEMPORAL_DIM) / (1.*num_samps*self.num_time_steps) # see https://numpy.org/doc/stable/reference/generated/numpy.sum.html
        self.channel_means = self.channel_means.reshape(1, self.num_chans)
        self.channel_std_devs = None # we now compute the std dev for each channel - see https://www.mathsisfun.com/data/standard-deviation-formulas.html
        
        for i in range(len(self.data)):
            curr_data_samp_path = self.data[i][0]
            
            with open(curr_data_samp_path, 'rb') as infile:
                curr_data_samp = None
                if "original" in self.signal_format:
                    curr_data_samp = pkl.load(infile)[self.data[i][1]][self.X_IND]
                elif "wavelet_decomp" in self.signal_format:
                    curr_data_samp = pkl.load(infile)[self.data[i][1]][self.X_WAV_DECOMP_IND]
                elif "approximation" in self.signal_format:
                    curr_data_samp = pkl.load(infile)[self.data[i][1]][self.X_APPROX_IND]
                else:
                    raise ValueError("NormalizedSyntheticWVARDataset.__init__: Unrecognized signal format == ", self.signal_format)

                if i == 0:
                    self.channel_std_devs = (curr_data_samp - self.channel_means)**2.
                else:
                    self.channel_std_devs = self.channel_std_devs + (curr_data_samp - self.channel_means)**2.
            pass

        if len(self.channel_std_devs.shape) > 2:
            self.channel_std_devs = self.channel_std_devs[0,:,:]
        self.channel_std_devs = torch.from_numpy(np.sqrt(np.sum(self.channel_std_devs, axis=self.TEMPORAL_DIM) / (1.*num_samps*self.num_time_steps)))
        self.channel_std_devs = self.channel_std_devs.reshape(1, self.num_chans)

        print("NormalizedSyntheticWVARDataset.__init__: self.channel_means == ", self.channel_means)
        print("NormalizedSyntheticWVARDataset.__init__: self.channel_std_devs == ", self.channel_std_devs)

        if shuffle:
            random.Random(shuffle_seed).shuffle(self.data)
        
        if grid_search:
            print("NormalizedSyntheticWVARDataset.__init__: WARNING!!! ONLY GRABBING FIRST 1/4 ELEMENTS OF DATASET FOR GRID SEARCH PURPOSES!!!") # FOR GRID SEARCH / DEBUGGING PURPOSES
            self.data = self.data[:len(self.data)//4] # FOR GRID SEARCH / DEBUGGING PURPOSES
            print("NormalizedSyntheticWVARDataset.__init__: WARNING!!! SHORTENED len(self.data) == ", len(self.data), " !!!", flush=True) # FOR GRID SEARCH / DEBUGGING PURPOSES
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

            x = None
            if "original" in self.signal_format:
                x = curr_data[self.X_IND]
            elif "wavelet_decomp" in self.signal_format:
                x = curr_data[self.X_WAV_DECOMP_IND]
            elif "approximation" in self.signal_format:
                x = curr_data[self.X_APPROX_IND]
            else:
                raise ValueError("NormalizedSyntheticWVARDataset.__init__: Unrecognized signal format == ", self.signal_format)

            if torch_sample:
                x = x.squeeze().to(torch.float32)#.float() # x is a torch tensor of shape (1, num_time_steps, num_channels) originally
            else:
                x = torch.from_numpy(x).squeeze().to(torch.float32)#.float() # x has shape (1, num_time_steps, num_channels) originally

            x = (x - self.channel_means) / self.channel_std_devs

            if "directed_spectrum" in self.signal_format:
                assert self.dirspec_params is not None
                if self.pad_X is not None:
                    if self.pad_X[0] is None or self.pad_X[1] is not None or self.pad_X[2] is not None or self.pad_X[3] is not None:
                        raise NotImplemented("CURRENT Dirspec FORMATTING ONLY ALLOWS FOR FRONT-END PADDING")
                    padded_x_versions = []
                    for pad_index in range(self.pad_X[0]):
                        curr_padded_x = None
                        if x.size()[0] >= self.max_num_features_per_series + self.pad_X[0]:
                            curr_padded_x = x[pad_index:self.max_num_features_per_series+pad_index,:]
                        else:
                            curr_padded_x = torch.zeros(x[:self.max_num_features_per_series,:].size())
                            curr_padded_x[self.pad_X[0]-pad_index:,:] = curr_padded_x[self.pad_X[0]-pad_index:,:] + x[:self.max_num_features_per_series-(self.pad_X[0]-pad_index),:]

                        high_level_features = make_high_level_signal_features(
                            curr_padded_x,
                            fs=self.dirspec_params["fs"],#1000,
                            min_freq=self.dirspec_params["min_freq"],#0.0, 
                            max_freq=self.dirspec_params["max_freq"],#55.0,
                            directed_spectrum=self.dirspec_params["directed_spectrum"],#False,
                            csd_params=self.dirspec_params["csd_params"],#{"detrend": "constant","window": "hann","nperseg": 512,"noverlap": 256,"nfft": None,},
                        )
                        if self.freq_bins is None:
                            self.freq_bins = high_level_features["freq"]
                            print("NormalizedSyntheticWVARDataset.__getitem__: len(self.freq_bins) == ", len(self.freq_bins))
                            print("NormalizedSyntheticWVARDataset.__getitem__: self.freq_bins == ", self.freq_bins)
                        if "vanilla" in self.signal_format:
                            num_regions = high_level_features["dir_spec"][0,:,:,:].shape[0]
                            num_features = high_level_features["dir_spec"][0,:,:,:].shape[-1]
                            curr_padded_x = torch.from_numpy(np.reshape(high_level_features["dir_spec"][0,:,:,:], (num_regions*num_regions*num_features)))
                        else:
                            curr_padded_x = torch.flatten(torch.from_numpy(flatten_directed_spectrum_features(high_level_features["dir_spec"][0,:,:,:])))
                        
                        padded_x_versions.append(curr_padded_x.to(torch.float32))
                    
                    y = torch.from_numpy(curr_data[self.Y_LABEL_IND]).to(torch.float32)
                    if len(y.size()) == 3:
                        assert y.size()[1] == x.size()[self.TEMPORAL_DIM]
                        assert y.size()[2] == x.size()[self.CHANNEL_DIM]
                    elif len(y.size()) != 2 and label_each_time_step: # copy y label for every time point in x
                        assert len(y.size()) == 1, y.size()
                        y = y.repeat(x.size()[self.TEMPORAL_DIM],1).T
                    elif len(y.size()) == 1 and y.size()[-1] != x.size()[self.TEMPORAL_DIM] and label_each_time_step: # multiple labels present that haven't been assigned to time points
                        raise NotImplementedError("NormalizedSyntheticWVARDataset.__init__: CAN'T CURRENTLY HANDLE y.size() == "+str(y.size())+" when x.size() == "+str(x.size()))
                    y = y[:,self.max_num_features_per_series:self.max_num_features_per_series+self.pad_X[0]] # only keep the labels immediately following padded windows
                    
                    return padded_x_versions, y.to(torch.float32)

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
                    print("NormalizedSyntheticWVARDataset.__getitem__: len(self.freq_bins) == ", len(self.freq_bins))
                    print("NormalizedSyntheticWVARDataset.__getitem__: self.freq_bins == ", self.freq_bins)
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

            y = torch.from_numpy(curr_data[self.Y_LABEL_IND]).to(torch.float32)
            
            if len(y.size()) == 3:
                assert y.size()[1] == x.size()[self.TEMPORAL_DIM]
                assert y.size()[2] == x.size()[self.CHANNEL_DIM]
            elif len(y.size()) != 2 and label_each_time_step: # copy y label for every time point in x
                assert len(y.size()) == 1, y.size()
                y = y.repeat(x.size()[self.TEMPORAL_DIM],1).T
            elif len(y.size()) == 1 and y.size()[-1] != x.size()[self.TEMPORAL_DIM] and label_each_time_step: # multiple labels present that haven't been assigned to time points
                raise NotImplementedError("NormalizedSyntheticWVARDataset.__init__: CAN'T CURRENTLY HANDLE y.size() == "+str(y.size())+" when x.size() == "+str(x.size()))
        
        return x.to(torch.float32), y.to(torch.float32)


def load_normalized_synthetic_wVAR_data(data_path, batch_size, signal_format="original", shuffle=True, shuffle_seed=0, dirspec_params=None, grid_search=True):
    data_loader = None
    if data_path is not None:
        data_set = NormalizedSyntheticWVARDataset(data_path, signal_format=signal_format, shuffle=shuffle, shuffle_seed=shuffle_seed, dirspec_params=dirspec_params, grid_search=grid_search)
        data_loader = DataLoader(data_set, batch_size=batch_size) # see https://blog.paperspace.com/dataloaders-abstractions-pytorch/
    return data_loader


def load_normalized_synthetic_wVAR_data_train_test_split(data_root_path, batch_size, signal_format="original", shuffle=True, shuffle_seed=0, train_portion=0.8, dirspec_params=None, grid_search=True):
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
    
    train_set = NormalizedSyntheticWVARDataset(train_data_path, signal_format=signal_format, shuffle=shuffle, shuffle_seed=shuffle_seed, dirspec_params=dirspec_params, grid_search=grid_search)
    train_loader = DataLoader(train_set, batch_size=batch_size) # see https://blog.paperspace.com/dataloaders-abstractions-pytorch/
    val_set = NormalizedSyntheticWVARDataset(val_data_path, signal_format=signal_format, shuffle=shuffle, shuffle_seed=shuffle_seed, dirspec_params=dirspec_params, grid_search=grid_search)
    val_loader = DataLoader(val_set, batch_size=batch_size) # see https://blog.paperspace.com/dataloaders-abstractions-pytorch/
    return train_loader, val_loader
    

def load_normalized_synthetic_wVAR_data_train_test_split_as_matrices(data_root_path, signal_format="original flattened", shuffle=True, shuffle_seed=0, 
                                                                     max_num_features_per_series=25, train_portion=0.8, dirspec_params=None, grid_search=True, 
                                                                     average_label_over_time_steps=True, pad_X=None):
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
    
    train_set = NormalizedSyntheticWVARDataset(
        train_data_path, 
        signal_format=signal_format, 
        shuffle=shuffle, 
        shuffle_seed=shuffle_seed, 
        max_num_features_per_series=max_num_features_per_series, 
        dirspec_params=dirspec_params, 
        grid_search=grid_search, 
        pad_X=pad_X
    )
    print("data.synthetic_datasets.load_normalized_synthetic_wVAR_data_train_test_split_as_matrices: len(train_set) == ", len(train_set))
    X_train = None
    Y_train = None
    if pad_X is not None:
        X_train = []
        Y_train = []
        for i, (padded_xs, y) in enumerate(train_set):
            y = y.squeeze()
            assert len(y.size()) == 2
            assert y.size()[1] == len(padded_xs)
            assert not average_label_over_time_steps
            for p, x in enumerate(padded_xs):
                assert len(x.size()) == 1
                if i==0:
                    if p==0:
                        print("data.synthetic_datasets.load_normalized_synthetic_wVAR_data_train_test_split_as_matrices: train x.size() == ", x.size())
                        print("data.synthetic_datasets.load_normalized_synthetic_wVAR_data_train_test_split_as_matrices: train y.size() == ", y.size())
                    curr_X_train = torch.zeros((len(train_set), len(x)))
                    curr_Y_train = torch.zeros((len(train_set), y.size()[0]))
                    X_train.append(curr_X_train)
                    Y_train.append(curr_Y_train)
                X_train[p][i,:] = X_train[p][i,:] + x
                Y_train[p][i,:] = Y_train[p][i,:] + y[:,p]
    else:
        for i in range(len(train_set)):
            x, y = train_set[i]
            y = y.squeeze()
            if len(y.size()) == 2 and average_label_over_time_steps: 
                y = torch.mean(y, 1)
                
            if X_train is None:
                print("data.synthetic_datasets.load_normalized_synthetic_wVAR_data_train_test_split_as_matrices: train x.size() == ", x.size())
                print("data.synthetic_datasets.load_normalized_synthetic_wVAR_data_train_test_split_as_matrices: train y.size() == ", y.size())
                assert Y_train is None
                assert len(x.size()) == 1
                X_train = torch.zeros((len(train_set), len(x)))
                if len(y.size()) < 1:
                    Y_train = torch.zeros((len(train_set), 1))
                else:
                    Y_train = torch.zeros((len(train_set), len(y.flatten())))
            X_train[i,:] = X_train[i,:] + x
            if len(y.size()) >= 1:
                Y_train[i,:] = Y_train[i,:] + y.flatten()
    
    val_set = NormalizedSyntheticWVARDataset(
        val_data_path, 
        signal_format=signal_format, 
        shuffle=shuffle, 
        shuffle_seed=shuffle_seed, 
        max_num_features_per_series=max_num_features_per_series, 
        dirspec_params=dirspec_params, 
        grid_search=grid_search, 
        pad_X=pad_X
    )
    print("data.synthetic_datasets.load_normalized_synthetic_wVAR_data_train_test_split_as_matrices: len(val_set) == ", len(val_set))
    X_val = None
    Y_val = None
    if pad_X is not None:
        X_val = []
        Y_val = []
        for i, (padded_xs, y) in enumerate(val_set):
            y = y.squeeze()
            assert len(y.size()) == 2
            assert y.size()[1] == len(padded_xs)
            assert not average_label_over_time_steps
            for p, x in enumerate(padded_xs):
                assert len(x.size()) == 1
                if i==0:
                    if p==0:
                        print("data.synthetic_datasets.load_normalized_synthetic_wVAR_data_val_test_split_as_matrices: val x.size() == ", x.size())
                        print("data.synthetic_datasets.load_normalized_synthetic_wVAR_data_val_test_split_as_matrices: val y.size() == ", y.size())
                    curr_X_val = torch.zeros((len(val_set), len(x)))
                    curr_Y_val = torch.zeros((len(val_set), y.size()[0]))
                    X_val.append(curr_X_val)
                    Y_val.append(curr_Y_val)
                X_val[p][i,:] = X_val[p][i,:] + x
                Y_val[p][i,:] = Y_val[p][i,:] + y[:,p]
    else:
        for i in range(len(val_set)):
            x, y = val_set[i]
            y = y.squeeze()
            if len(y.size()) == 2 and average_label_over_time_steps: 
                y = torch.mean(y, 1)
                
            if X_val is None:
                print("data.synthetic_datasets.load_normalized_synthetic_wVAR_data_train_test_split_as_matrices: val x.size() == ", x.size())
                print("data.synthetic_datasets.load_normalized_synthetic_wVAR_data_train_test_split_as_matrices: val y.size() == ", y.size())
                assert Y_val is None
                assert len(x.size()) == 1
                X_val = torch.zeros((len(val_set), len(x)))
                if len(y.size()) < 1:
                    Y_val = torch.zeros((len(val_set), 1))
                else:
                    Y_val = torch.zeros((len(val_set), len(y.flatten())))
            X_val[i,:] = X_val[i,:] + x
            if len(y.size()) >= 1:
                Y_val[i,:] = Y_val[i,:] + y.flatten()
    
    if pad_X is not None:
        return [Xt.numpy() for Xt in X_train], [Yt.numpy() for Yt in Y_train], [Xv.numpy() for Xv in X_val], [Yv.numpy() for Yv in Y_val]
    return X_train.numpy(), Y_train.numpy(), X_val.numpy(), Y_val.numpy()