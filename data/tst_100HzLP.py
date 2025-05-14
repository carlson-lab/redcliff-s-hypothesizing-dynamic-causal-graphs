import numpy as np
import scipy.io as scio
import os
import random
import pickle as pkl
import argparse
import json

from general_utils.time_series import *

# fix random seed(s) to 1337 -- see https://discuss.pytorch.org/t/how-could-i-fix-the-random-seed-absolutely/45515
np.random.seed(1337)
random.seed(1337)




def load_lfp_data_matrix(raw_data_path, raw_file_name, keys_of_interest, num_channels_in_samples, sample_freq=1000, 
                         cutoff=LOW_PASS_CUTOFF, lowcut=LOWCUT, highcut=HIGHCUT, mad_threshold=DEFAULT_MAD_TRESHOLD, 
                         q=Q, order=ORDER, apply_notch_filters=True, filter_type="lowpass"):
    # get raw-data, replacing outlying values with np.nan
    raw_data = scio.loadmat(os.path.join(raw_data_path, raw_file_name))
    raw_data = {key:raw_data[key].reshape(-1).astype(float) for key in keys_of_interest}
    raw_data = mark_outliers(
        raw_data, 
        sample_freq, 
        cutoff=cutoff, 
        lowcut=lowcut, 
        highcut=highcut, 
        mad_threshold=mad_threshold, 
        filter_type=filter_type
    )

    # combine raw data into a single matrix
    raw_data_combined = filter_signal(
        raw_data[keys_of_interest[0]], 
        sample_freq, 
        cutoff=cutoff, 
        lowcut=lowcut, 
        highcut=highcut, 
        q=q, 
        order=order, 
        apply_notch_filters=apply_notch_filters, 
        filter_type=filter_type
    ).reshape(1,-1)
    for key in keys_of_interest[1:]:
        raw_data_combined = np.vstack([
            raw_data_combined, 
            filter_signal(
                raw_data[key], 
                sample_freq, 
                cutoff=cutoff, 
                lowcut=lowcut, 
                highcut=highcut, 
                q=q, 
                order=order, 
                apply_notch_filters=apply_notch_filters, 
                filter_type=filter_type
            ).reshape(1,-1)
        ])

    assert raw_data_combined.shape[0] == num_channels_in_samples # sanity check
    return raw_data_combined


def determine_keys_of_interest(files_to_process, raw_data_path):
    keys_of_interest = set()
    keys_to_remove = set()
    for i, raw_file_name in enumerate(files_to_process):
        raw_data = scio.loadmat(os.path.join(raw_data_path, raw_file_name))
        raw_data_useful_keys = [x for x in raw_data.keys() if "__" not in x]
        if i == 0:
            keys_of_interest = set(raw_data_useful_keys)
        else:
            for key in keys_of_interest:
                if key not in raw_data_useful_keys:
                    keys_to_remove.add(key)
    for key in keys_to_remove:
        keys_of_interest.remove(key)
    return sorted(list(keys_of_interest))


def preprocess_tst_raw_lfps_for_windowed_training(lfp_data_path, label_data_path, preprocessed_data_save_path, post_processing_sample_freq, 
                                                  num_processed_samples=10000, sample_temp_window_size=1000, 
                                                  max_num_samps_per_preprocessed_file=100, sample_freq=1000, max_num_samp_attempts=10, 
                                                  cutoff=LOW_PASS_CUTOFF, lowcut=LOWCUT, highcut=HIGHCUT, mad_threshold=DEFAULT_MAD_TRESHOLD, 
                                                  q=Q, order=ORDER, apply_notch_filters=True, filter_type="lowpass"):
    print("tst.preprocess_tst_raw_lfps_for_windowed_training:  START")
    assert sample_freq > post_processing_sample_freq
    downsampling_step_size = sample_freq // post_processing_sample_freq
    print("tst.preprocess_tst_raw_lfps_for_windowed_training:  downsampling_step_size == ", downsampling_step_size, flush=True)
    

    raw_lfp_files_to_load = sorted([x for x in os.listdir(lfp_data_path) if "_LFP" in x and ".mat" in x])
    behavioral_label_files_to_load = sorted([x for x in os.listdir(label_data_path) if "_TIME" in x and ".mat" in x])
    print("tst.preprocess_tst_raw_lfps_for_windowed_training:  len(raw_lfp_files_to_load) == ", len(raw_lfp_files_to_load))
    print("tst.preprocess_tst_raw_lfps_for_windowed_training:  raw_lfp_files_to_load == ", raw_lfp_files_to_load)
    print("tst.preprocess_tst_raw_lfps_for_windowed_training:  len(behavioral_label_files_to_load) == ", len(behavioral_label_files_to_load))
    print("tst.preprocess_tst_raw_lfps_for_windowed_training:  behavioral_label_files_to_load == ", behavioral_label_files_to_load, flush=True)

    unique_mice_names = list(set([x.split('_')[0] for x in raw_lfp_files_to_load]))
    print("tst.preprocess_tst_raw_lfps_for_windowed_training:  len(unique_mice_names) == ", len(unique_mice_names))
    print("tst.preprocess_tst_raw_lfps_for_windowed_training:  unique_mice_names == ", unique_mice_names, flush=True)
    random.shuffle(unique_mice_names)
    num_samps_per_mouse = num_processed_samples // len(unique_mice_names)
    num_samples_per_label_type = num_samps_per_mouse//3
    print("tst.preprocess_tst_raw_lfps_for_windowed_training:  num_samps_per_mouse == ", num_samps_per_mouse)
    print("tst.preprocess_tst_raw_lfps_for_windowed_training:  num_samples_per_label_type == ", num_samples_per_label_type, flush=True)

    keys_of_interest = determine_keys_of_interest(raw_lfp_files_to_load, lfp_data_path)
    keys_of_interest.remove('TailSuspension')
    num_channels_in_samples = len(keys_of_interest)
    print("tst.preprocess_tst_raw_lfps_for_windowed_training: num_channels_in_samples == ", num_channels_in_samples)

    print("\ntst.preprocess_tst_raw_lfps_for_windowed_training: PROCESSING RAW FILES AND SAVING CLEANED SAMPLES\n", flush=True)
    for i, mouse_name in enumerate(unique_mice_names):
        print("\nprocessing mouse samples: now processing files for mouse i==", i, " of n==", len(unique_mice_names), " with mouse_name == ", mouse_name, flush=True)
        mouse_lfp_files = [x for x in raw_lfp_files_to_load if mouse_name in x]
        mouse_label_files = [x for x in behavioral_label_files_to_load if mouse_name in x]
        if len(mouse_lfp_files) == len(mouse_label_files):
            hC_samples = []
            oF_samples = []
            tS_samples = []
            subset_counter = 0
            for (curr_lfp_file_name, curr_intTime_file_name) in zip(mouse_lfp_files, mouse_label_files):
                assert curr_lfp_file_name[:23] == curr_intTime_file_name[:23] # ensure files are properly aligned/matched
                print("tst.preprocess_tst_raw_lfps_for_windowed_training: pairing curr_lfp_file_name == ", curr_lfp_file_name, " with curr_intTime_file_name == ", curr_intTime_file_name, flush=True)

                # read in data
                int_time_vec = scio.loadmat(os.path.join(label_data_path, curr_intTime_file_name))['INT_TIME'].reshape(-1) # int time vector is structured as [openField_start_sec, openField_durration_by_sec, tailSuspension_start_sec, tailSuspension_durration_by_sec]
                raw_data_combined = load_lfp_data_matrix(
                    lfp_data_path, 
                    curr_lfp_file_name, 
                    keys_of_interest, 
                    num_channels_in_samples, 
                    sample_freq=sample_freq, 
                    cutoff=cutoff, 
                    lowcut=lowcut, 
                    highcut=highcut, 
                    mad_threshold=mad_threshold, 
                    q=q, 
                    order=order, 
                    apply_notch_filters=apply_notch_filters, 
                    filter_type=filter_type
                )

                # split data by labels
                homeCage_start = 0
                homeCage_stop = 300*sample_freq # homecage recording was 5min = 5*60sec = 300sec
                openField_start = int_time_vec[0]*sample_freq
                openField_stop = (int_time_vec[0]+int_time_vec[1])*sample_freq
                tailSuspension_start = int_time_vec[2]*sample_freq
                tailSuspension_stop = (int_time_vec[2]+int_time_vec[3])*sample_freq
                
                labels_by_time_step = np.zeros(len(raw_data_combined)) - 1
                labels_by_time_step[homeCage_start:homeCage_stop] = labels_by_time_step[homeCage_start:homeCage_stop] + 1
                labels_by_time_step[openField_start:openField_stop] = labels_by_time_step[openField_start:openField_stop] + 2
                labels_by_time_step[tailSuspension_start:tailSuspension_stop] = labels_by_time_step[tailSuspension_start:tailSuspension_stop] + 3

                hC_nan_locations = [l for l in range(homeCage_start, homeCage_stop) if np.isnan(np.sum(raw_data_combined[:, l]))]
                oF_nan_locations = [l for l in range(openField_start, openField_stop) if np.isnan(np.sum(raw_data_combined[:, l]))]
                tS_nan_locations = [l for l in range(tailSuspension_start, tailSuspension_stop) if np.isnan(np.sum(raw_data_combined[:, l]))]
                homeCage_sample_start_inds = draw_timesteps_to_sample_from(
                    homeCage_start, 
                    homeCage_stop, 
                    sample_temp_window_size, 
                    num_samples_per_label_type, 
                    hC_nan_locations, 
                    max_num_draws=10
                )
                openField_sample_start_inds = draw_timesteps_to_sample_from(
                    openField_start, 
                    openField_stop, 
                    sample_temp_window_size, 
                    num_samples_per_label_type, 
                    oF_nan_locations, 
                    max_num_draws=10
                )
                tailSuspension_sample_start_inds = draw_timesteps_to_sample_from(
                    tailSuspension_start, 
                    tailSuspension_stop, 
                    sample_temp_window_size, 
                    num_samples_per_label_type, 
                    tS_nan_locations, 
                    max_num_draws=10
                )

                print("tst.preprocess_tst_raw_lfps_for_windowed_training: DRAWING RANDOM SAMPLES", flush=True)
                for j in range(num_samples_per_label_type):
                    print("tst.preprocess_tst_raw_lfps_for_windowed_training: sample j == ", j, " of ", num_samples_per_label_type)
                    # draw random sample from raw_data_combined
                    if j < len(homeCage_sample_start_inds):
                        curr_hC_samp = raw_data_combined[:,homeCage_sample_start_inds[j]:homeCage_sample_start_inds[j]+sample_temp_window_size]
                        curr_hC_samp = np.transpose(curr_hC_samp, axes=(1,0)) # ensure all samples are of shape (num_time_steps, num_channels) - see https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
                        if np.isnan(np.sum(curr_hC_samp)):
                            print("tst.preprocess_tst_raw_lfps_for_windowed_training: \t NAN VALUE DETECTED IN SAMPLED SIGNAL")
                            print("tst.preprocess_tst_raw_lfps_for_windowed_training: \t homeCage_sample_start_inds[j] == ", homeCage_sample_start_inds[j])
                            print("tst.preprocess_tst_raw_lfps_for_windowed_training: \t homeCage_sample_start_inds[j]+sample_temp_window_size == ", homeCage_sample_start_inds[j]+sample_temp_window_size)
                            print("tst.preprocess_tst_raw_lfps_for_windowed_training: \t [x for x in hC_nan_locations if x>=homeCage_sample_start_inds[j] and x<=homeCage_sample_start_inds[j]+sample_temp_window_size] == ", [x for x in hC_nan_locations if x>=homeCage_sample_start_inds[j] and x<=homeCage_sample_start_inds[j]+sample_temp_window_size])
                            print("tst.preprocess_tst_raw_lfps_for_windowed_training: \t ENDING SAMPLE COLLECTION ASSOCIATED WITH CURRENT FILE DUE TO NAN DETECTION - curr_lfp_file_name == ", curr_lfp_file_name, flush=True)
                            break # prevent more nan-samples from being drawn
                        curr_hC_label = np.array([1.,0.,0.])
                        if downsampling_step_size > 1:
                            curr_hC_samp = curr_hC_samp[::downsampling_step_size, :]
                        hC_samples.append([curr_hC_samp, curr_hC_label])

                    if j < len(openField_sample_start_inds):
                        curr_oF_samp = raw_data_combined[:,openField_sample_start_inds[j]:openField_sample_start_inds[j]+sample_temp_window_size]
                        curr_oF_samp = np.transpose(curr_oF_samp, axes=(1,0)) # ensure all samples are of shape (num_time_steps, num_channels) - see https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
                        if np.isnan(np.sum(curr_oF_samp)):
                            print("tst.preprocess_tst_raw_lfps_for_windowed_training: \t NAN VALUE DETECTED IN SAMPLED SIGNAL")
                            print("tst.preprocess_tst_raw_lfps_for_windowed_training: \t openField_sample_start_inds[j] == ", openField_sample_start_inds[j])
                            print("tst.preprocess_tst_raw_lfps_for_windowed_training: \t openField_sample_start_inds[j]+sample_temp_window_size == ", openField_sample_start_inds[j]+sample_temp_window_size)
                            print("tst.preprocess_tst_raw_lfps_for_windowed_training: \t [x for x in oF_nan_locations if x>=openField_sample_start_inds[j] and x<=openField_sample_start_inds[j]+sample_temp_window_size] == ", [x for x in oF_nan_locations if x>=openField_sample_start_inds[j] and x<=openField_sample_start_inds[j]+sample_temp_window_size])
                            print("tst.preprocess_tst_raw_lfps_for_windowed_training: \t ENDING SAMPLE COLLECTION ASSOCIATED WITH CURRENT FILE DUE TO NAN DETECTION - curr_lfp_file_name == ", curr_lfp_file_name, flush=True)
                            break # prevent more nan-samples from being drawn
                        curr_oF_label = np.array([0.,1.,0.])
                        if downsampling_step_size > 1:
                            curr_oF_samp = curr_oF_samp[::downsampling_step_size, :]
                        oF_samples.append([curr_oF_samp, curr_oF_label])

                    if j < len(tailSuspension_sample_start_inds):
                        curr_tS_samp = raw_data_combined[:,tailSuspension_sample_start_inds[j]:tailSuspension_sample_start_inds[j]+sample_temp_window_size]
                        curr_tS_samp = np.transpose(curr_tS_samp, axes=(1,0)) # ensure all samples are of shape (num_time_steps, num_channels) - see https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
                        if np.isnan(np.sum(curr_tS_samp)):
                            print("tst.preprocess_tst_raw_lfps_for_windowed_training: \t NAN VALUE DETECTED IN SAMPLED SIGNAL")
                            print("tst.preprocess_tst_raw_lfps_for_windowed_training: \t tailSuspension_sample_start_inds[j] == ", tailSuspension_sample_start_inds[j])
                            print("tst.preprocess_tst_raw_lfps_for_windowed_training: \t tailSuspension_sample_start_inds[j]+sample_temp_window_size == ", tailSuspension_sample_start_inds[j]+sample_temp_window_size)
                            print("tst.preprocess_tst_raw_lfps_for_windowed_training: \t [x for x in tS_nan_locations if x>=tailSuspension_sample_start_inds[j] and x<=tailSuspension_sample_start_inds[j]+sample_temp_window_size] == ", [x for x in tS_nan_locations if x>=tailSuspension_sample_start_inds[j] and x<=tailSuspension_sample_start_inds[j]+sample_temp_window_size])
                            print("tst.preprocess_tst_raw_lfps_for_windowed_training: \t ENDING SAMPLE COLLECTION ASSOCIATED WITH CURRENT FILE DUE TO NAN DETECTION - curr_lfp_file_name == ", curr_lfp_file_name, flush=True)
                            break # prevent more nan-samples from being drawn
                        curr_tS_label = np.array([0.,0.,1.])
                        if downsampling_step_size > 1:
                            curr_tS_samp = curr_tS_samp[::downsampling_step_size, :]
                        tS_samples.append([curr_tS_samp, curr_tS_label])
                    
                    # periodically save sample sets to external files so that full dataset need not be loaded all at once
                    if j+1 == max_num_samps_per_preprocessed_file or j == num_samples_per_label_type-1:
                        print("tst.preprocess_tst_raw_lfps_for_windowed_training: \t now saving subset_counter==", subset_counter)
                        curr_hC_set_name = mouse_name+"_homeCage_processed_data_subset_"+str(subset_counter)+".pkl"
                        curr_oF_set_name = mouse_name+"_openField_processed_data_subset_"+str(subset_counter)+".pkl"
                        curr_tS_set_name = mouse_name+"_tailSuspension_processed_data_subset_"+str(subset_counter)+".pkl"

                        with open(os.path.join(preprocessed_data_save_path, curr_hC_set_name), 'wb') as outfile:
                            pkl.dump(hC_samples, outfile)
                        with open(os.path.join(preprocessed_data_save_path, curr_oF_set_name), 'wb') as outfile:
                            pkl.dump(oF_samples, outfile)
                        with open(os.path.join(preprocessed_data_save_path, curr_tS_set_name), 'wb') as outfile:
                            pkl.dump(tS_samples, outfile)

                        del hC_samples
                        del oF_samples
                        del tS_samples
                        hC_samples = []
                        oF_samples = []
                        tS_samples = []
                        subset_counter += 1
                        pass # end periodic save if statement
                    pass # end loop for saving random samples
                
                del raw_data_combined
                del labels_by_time_step
                del hC_nan_locations
                del oF_nan_locations
                del tS_nan_locations
                del homeCage_sample_start_inds
                del openField_sample_start_inds
                del tailSuspension_sample_start_inds
                pass # end loop over files corresponding to mice id
        else:
            print("tst.preprocess_tst_raw_lfps_for_windowed_training: IGNORING ANY FILES ASSOCIATED WITH mouse_name == ", mouse_name, " DUE TO MISMATCH IN CONTENTS OF mouse_lfp_files == ", mouse_lfp_files, " AND mouse_label_files == ", mouse_label_files, flush=True)
        pass # end loop over mice ids

    print("tst.preprocess_tst_raw_lfps_for_windowed_training:  STOP", flush=True)
    pass



if __name__ == '__main__':
    parse=argparse.ArgumentParser(description='preprocess tst data')
    parse.add_argument(
        "-cached_args_file",
        default="tst_100HzLP_cached_args.txt", # WARNING: edit this line when running new experiments on SLURM server
        help="/path/to/cached_args.txt should contain a json object with updated arg values",
    )
    args = parse.parse_args()

    print("__MAIN__: LOADING VALUES OF ARGUMENTS FOUND IN ", args.cached_args_file, flush=True)
    with open(args.cached_args_file, 'r') as infile:
        new_args_dict = json.load(infile)
        args.original_lfp_data_path = new_args_dict["original_lfp_data_path"]
        args.original_behavioral_label_data_path = new_args_dict["original_behavioral_label_data_path"]
        args.save_path = new_args_dict["save_path"]
        args.num_processed_samples = int(new_args_dict["num_processed_samples"])
        args.sample_temp_window_size = int(new_args_dict["sample_temp_window_size"]) # the number of time steps to capture in each time series recording window in final preprocessed dataset
        args.max_num_samps_per_preprocessed_file = int(new_args_dict["max_num_samps_per_preprocessed_file"])
        args.sample_freq = int(new_args_dict["sample_freq"]) # this represents the frequency in Hz that recordings were made at

        args.post_processing_sample_freq = int(new_args_dict["post_processing_sample_freq"]) # this represents the frequency (in Hz) that the final, preprocessed samples will be in
        args.max_num_samp_attempts = int(new_args_dict["max_num_samp_attempts"])
        args.cutoff = None if new_args_dict["cutoff"]=="None" else float(new_args_dict["cutoff"])
        args.lowcut = None if new_args_dict["lowcut"]=="None" else float(new_args_dict["lowcut"]) 
        args.highcut = None if new_args_dict["highcut"]=="None" else float(new_args_dict["highcut"]) 
        args.mad_threshold = float(new_args_dict["mad_threshold"]) 
        args.q = float(new_args_dict["q"]) 
        args.order = int(new_args_dict["order"]) 
        args.apply_notch_filters = True if new_args_dict["apply_notch_filters"]=="True" else False
        args.filter_type = new_args_dict["filter_type"]
    
    preprocess_tst_raw_lfps_for_windowed_training(
        args.original_lfp_data_path, 
        args.original_behavioral_label_data_path, 
        args.save_path, 
        args.post_processing_sample_freq, 
        num_processed_samples=args.num_processed_samples, 
        sample_temp_window_size=args.sample_temp_window_size, 
        max_num_samps_per_preprocessed_file=args.max_num_samps_per_preprocessed_file, 
        sample_freq=args.sample_freq, 
        max_num_samp_attempts=args.max_num_samp_attempts, 
        cutoff=args.cutoff, 
        lowcut=args.lowcut, 
        highcut=args.highcut, 
        mad_threshold=args.mad_threshold, 
        q=args.q, 
        order=args.order, 
        apply_notch_filters=args.apply_notch_filters, 
        filter_type=args.filter_type
    )
    print("__MAIN__: FINISHED !!!!!!", flush=True)
    pass
###