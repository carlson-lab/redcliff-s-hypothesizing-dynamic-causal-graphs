import os
import shutil
import json
import argparse



if __name__=="__main__":
    parse=argparse.ArgumentParser(description='Default .pkl/.npy data generation')
    parse.add_argument(
        "-cached_args_file",
        default="clean_sVARwInnovativeContinuousGaussianNoise_data_etNL_cached_args.txt", # WARNING: edit this line when running new experiments on SLURM server
        help="/path/to/cached_args.txt should contain a json object with updated arg values",
    )
    args = parse.parse_args()

    print("<<< CLEANING START !!! >>>")
    with open(args.cached_args_file, 'r') as infile:
        args_dict = json.load(infile)
        assert "data_root_path" in args_dict.keys()
        assert "args_save_path" in args_dict.keys()
        assert "num_folds_in_each_cv" in args_dict.keys()
    
    print("args_dict['data_root_path'] == ", args_dict["data_root_path"])
    print("len(os.listdir(args_dict['data_root_path'])) == ", len(os.listdir(args_dict["data_root_path"])))
    
    deleted_cvs = []
    partial_cvs = []
    detected_cached_args_file_names = []
    for cvr_id, cross_val_root_folder_name in enumerate(os.listdir(args_dict["data_root_path"])):
        if os.path.isdir(args_dict["data_root_path"]+os.sep+cross_val_root_folder_name):
            curr_cv_folder = args_dict["data_root_path"] + os.sep + cross_val_root_folder_name
            IS_COMPLETE = 5
            print("CHECKING FOR  data cached args file IN cv folder ", curr_cv_folder)
            
            for cv_fold_folder_name in os.listdir(curr_cv_folder):
                if os.path.isdir(curr_cv_folder+os.sep+cv_fold_folder_name):
                    curr_fold_folder = curr_cv_folder + os.sep + cv_fold_folder_name
                    data_cached_arg_files_for_curr_fold = [x for x in os.listdir(curr_fold_folder) if os.path.exists(curr_fold_folder+os.sep+x) and "data_fold" in x and "_cached_args_sensitive.txt" in x]
                    if len(data_cached_arg_files_for_curr_fold) == 0:
                        IS_COMPLETE -= 1
                        print("\t Removing fold due to missing cached args file; folder was originally saved at curr_fold_folder == ", curr_fold_folder) # for debugging
                        shutil.rmtree(curr_fold_folder) # for debugging
                    elif len(data_cached_arg_files_for_curr_fold) > 1:
                        print("\t *** WARNING: MORE CACHED ARG FILES THAN EXPECTED ENCOUNTERED IN curr_fold_folder == ", curr_fold_folder, flush=True)
                        raise ValueError()
                    else:
                        args_file_name = "_".join([curr_cv_folder.split(os.sep)[-1],data_cached_arg_files_for_curr_fold[0]])
                        detected_cached_args_file_names.append(args_file_name)
                        # see https://www.geeksforgeeks.org/copy-files-and-rename-in-python/
                        shutil.copy(curr_fold_folder+os.sep+data_cached_arg_files_for_curr_fold[0], args_dict["args_save_path"]) # for debugging
                        shutil.move(args_dict["args_save_path"]+os.sep+data_cached_arg_files_for_curr_fold[0], args_dict["args_save_path"]+os.sep+args_file_name) # for debugging
            if IS_COMPLETE == 0:
                deleted_cvs.append(curr_cv_folder)
                print(" ! Removing cv root folder due to no completed folds; cv was originally saved at curr_cv_folder == ", curr_cv_folder) # for debugging
                shutil.rmtree(curr_cv_folder) # for debugging
            elif IS_COMPLETE < int(args_dict["num_folds_in_each_cv"]):
                partial_cvs.append(curr_cv_folder)
    
    print("#########")
    
    print("deleted_cvs = ", deleted_cvs, flush=True)
    print("partial_cvs = ", partial_cvs, flush=True)
    
    detected_cached_args_file_names_SET = set(detected_cached_args_file_names)
    duplicate_filtered_detected_cached_args_file_names = list(detected_cached_args_file_names_SET)
    assert len(detected_cached_args_file_names) == len(duplicate_filtered_detected_cached_args_file_names)
    print("detected_cached_args_file_names = ", detected_cached_args_file_names, flush=True)
    
    print("<<< CLEANING COMPLETED !!! >>>")
    pass