import os
import json
import argparse
import numpy as np
import pickle as pkl


if __name__=="__main__":
    parse=argparse.ArgumentParser(description='Default .pkl/.npy data generation')
    parse.add_argument(
        "-cached_args_file",
        default="aggregate_synthetic_systems_datasets_cached_args.txt", # WARNING: edit this line when running new experiments on SLURM server
        help="/path/to/cached_args.txt should contain a json object with updated arg values",
    )
    args = parse.parse_args()
    
    print("__MAIN__: BEGINNING DATA AGGREGATION\n\n")
    with open(args.cached_args_file, 'r') as infile:
        args_dict = json.load(infile)
        assert "data_save_path" in args_dict.keys()
        assert "orig_data_root_path" in args_dict.keys()
    
    orig_system_folders = [
        "numF2_numSF2_numN6_numE2_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF", 
        "numF2_numSF2_numN6_numE4_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF", 
        "numF2_numSF2_numN12_numE11_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF", 
        "numF5_numSF5_numN12_numE11_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF", 
    ]
    repeat_folder_names = [
        "fold_0", 
        "fold_1", 
        "fold_2", 
        "fold_3", 
        "fold_4", 
    ]
    
    for o_sys_folder in orig_system_folders:
        print("__MAIN__: o_sys_folder == ", o_sys_folder)
        # make aggregated dir 
        os.mkdir(args_dict["data_save_path"]+os.sep+o_sys_folder)
        
        for rep_folder in repeat_folder_names:
            print("__MAIN__: \t rep_folder == ", rep_folder)
            # list all files in o_sys_folder + rep_folder + train
            curr_train_data_path = args_dict["orig_data_root_path"]+os.sep+o_sys_folder+os.sep+rep_folder+os.sep+"train"
            subset_files = [curr_train_data_path+os.sep+f for f in os.listdir(curr_train_data_path) if "subset" in f and ".pkl" in f]
            
            aggregated_data = []
            for i, subset_file in enumerate(subset_files):
                if i%10==0:
                    print("__MAIN__: \t\t subset i==", i)
                # open file
                samps = pkl.load(open(subset_file, "rb"))
                # add file samps to aggregate
                aggregated_data = aggregated_data + samps
                
            # save aggregated data
            print("__MAIN__: \t\t final len(aggregated_data) == ", len(aggregated_data))
            save_path = args_dict["data_save_path"]+os.sep+o_sys_folder+os.sep+"aggregated_rep"+rep_folder[-1]+".pkl"
            pkl.dump(aggregated_data, open(save_path, 'wb'))
            print("__MAIN__: \t\t AGGREGATED DATA SAVED TO save_path == ", save_path, flush=True)
    
    print("\n\n__MAIN__: DONE !!!")