import numpy as np
import os
import pickle as pkl
import argparse
import json
import copy

from general_utils.input_argument_utils import read_in_data_args
from evaluate.eval_utils import load_model_for_eval, get_model_gc_estimates, compute_OptimalF1_stats_betw_two_gc_graphs
from general_utils.misc import mask_diag_elements_of_square_numpy_array, normalize_numpy_array
from general_utils.plotting import plot_gc_est_comparisson, make_scatter_and_stdErrOfMean_plot_overlay_vis



if __name__=="__main__":
    parse=argparse.ArgumentParser(description='Default parameter system-level evaluation')
    parse.add_argument(
        "-cached_args_file",
        default="eval_sysOptF1_crossAlg_d4IC_HSNR_bCgsParsim_REDCSmovOGcMLP_cached_args.txt", # WARNING: edit this line when running new experiments on SLURM server
        help="/path/to/cached_args.txt should contain a json object with updated arg values",
    )
    args = parse.parse_args()
    
    print("__MAIN__: LOADING ARGS", flush=True)
    root_paths_to_trained_models = []
    with open(args.cached_args_file, 'r') as infile:
        args_dict = json.load(infile)
        save_root_path = args_dict["save_root_path"]
        for key in args_dict.keys():
            if "_trained_models_root_dir" in key:
                root_paths_to_trained_models.append(args_dict[key])
    
    # define cross-val (unfolded) dataset names
    CV_DATA_SETS = [
        "dream4_insilicoCombo_size10_HSNR", 
    ]
    # define the number of folds expected in each dataset
    NUM_FOLDS_PER_CV_DATASET = 5
    DELTACON0_EPSILON = 0.1
    
    # load ground-truth causal graphs for each fold of each cv-dataset
    cached_args_folder_name = "cached_dataset_args/"
    files_of_cached_data_args = [
        "cached_dataset_args/dream4_insilicoCombo_size10_HSNR_fold0_cached_args.txt",
        "cached_dataset_args/dream4_insilicoCombo_size10_HSNR_fold1_cached_args.txt",
        "cached_dataset_args/dream4_insilicoCombo_size10_HSNR_fold2_cached_args.txt",
        "cached_dataset_args/dream4_insilicoCombo_size10_HSNR_fold3_cached_args.txt",
        "cached_dataset_args/dream4_insilicoCombo_size10_HSNR_fold4_cached_args.txt",
    ]
    
    true_causal_graphs = {cv_dataset:{fold_id: None for fold_id in range(NUM_FOLDS_PER_CV_DATASET)} for cv_dataset in CV_DATA_SETS}
    
    data_vis_root_save_path = save_root_path+os.sep+"data_set_vis"
    if not os.path.exists(data_vis_root_save_path):
        os.mkdir(data_vis_root_save_path)
    for cv_dataset in true_causal_graphs.keys():
        cv_data_vis_save_path = data_vis_root_save_path+os.sep+cv_dataset
        if not os.path.exists(cv_data_vis_save_path):
            os.mkdir(cv_data_vis_save_path)
        for fold_id in range(NUM_FOLDS_PER_CV_DATASET):
            fold_data_vis_save_path = cv_data_vis_save_path+os.sep+"fold_"+str(fold_id)
            if not os.path.exists(fold_data_vis_save_path):
                os.mkdir(fold_data_vis_save_path)
            args_dict = dict()
            args_dict["model_type"] = "REDCLIFF_S_CMLP" # this setting will simply be used to read in the most generic format of the true causal graphs
            args_dict["save_root_path"] = fold_data_vis_save_path
            args_dict["data_set_name"] = cv_dataset + "_fold" + str(fold_id)
            args_dict["data_cached_args_file"] = cached_args_folder_name + args_dict["data_set_name"] + "_cached_args.txt"
            assert args_dict["data_cached_args_file"] in files_of_cached_data_args # sanity check
            args_dict["true_GC_factors"] = None # need to reset args_dict["true_GC_factors"] to ensure it is not carried over from previous evaluation
            args_dict = read_in_data_args(args_dict, include_gc_views_for_eval=False, read_in_gc_factors_for_eval=True)
            true_causal_graphs[cv_dataset][fold_id] = args_dict["true_GC_factors"]
    
    # define all possible algorithms supported
    ALL_POSSIBLE_ALGORITHMS = [
        "REDCLIFF_S_CMLP", 
        "REDCLIFF_S_CLSTM", 
        "REDCLIFF_S_DGCNN", 
        "CMLP", 
        "CLSTM", 
        "DGCNN", 
        "DCSFA", 
        "DYNOTEARS_Stochastic", 
        "DYNOTEARS_Vanilla", 
        "NAVAR_CLSTM", 
        "NAVAR_CMLP", 
    ]
    
    # grab all algorithm names relevant to this script
    ALGORITHMS_TO_EVALUATE = []
    for x in ALL_POSSIBLE_ALGORITHMS:
        for trained_m_path in root_paths_to_trained_models:
            if x in trained_m_path and x not in ALGORITHMS_TO_EVALUATE:
                ALGORITHMS_TO_EVALUATE.append(x)
    
    # analyze experiment results
    full_comparrisson_summary = dict()
    for cv_id, cv_dset_name in enumerate(CV_DATA_SETS):
        print("__MAIN__: cv_id == ", cv_id, " --------------------------------------------------------------")
        print("__MAIN__: cv_dset_name == ", cv_dset_name)
        cv_level_stats = dict()
        curr_cv_save_path = save_root_path + os.sep + "cv"+str(cv_id)+"_"+cv_dset_name
        if not os.path.exists(curr_cv_save_path):
            os.mkdir(curr_cv_save_path)
        CURR_NUM_FACTORS_IN_CV_DATA_SETS = None
        
        for f_num in range(NUM_FOLDS_PER_CV_DATASET):
            print("__MAIN__: \t fold == ", f_num)
            fold_level_stats = dict()
            curr_fold_save_path = curr_cv_save_path + os.sep + "fold_"+str(f_num)
            if not os.path.exists(curr_fold_save_path):
                os.mkdir(curr_fold_save_path)
                
            true_gcs = true_causal_graphs[cv_dset_name][f_num]
            if f_num == 0:
                CURR_NUM_FACTORS_IN_CV_DATA_SETS = len(true_gcs)
            
            for alg_id, alg_name in enumerate(ALGORITHMS_TO_EVALUATE):
                print("__MAIN__: \t \t alg ", alg_id, "/", len(ALGORITHMS_TO_EVALUATE), " <<< ", alg_name, " >>> ", flush=True)
                alg_level_stats = dict()
                curr_alg_save_path = curr_fold_save_path + os.sep + alg_name
                if not os.path.exists(curr_alg_save_path):
                    os.mkdir(curr_alg_save_path)
                
                # load the trained model corresponding to current iteration --------------------------------
                alg_name_alias = alg_name
                curr_model_root_save_dir = None
                if alg_name in ["CMLP", "CLSTM", "DGCNN"]: # handle edge case from earlier naming convention(s)
                    if alg_name in ["CMLP", "CLSTM"]:
                        alg_name_alias = "c" + alg_name[1:]
                    curr_model_root_save_dir = [x for x in root_paths_to_trained_models if alg_name in x and "REDCLIFF" not in x and "NAVAR" not in x]
                else:
                    curr_model_root_save_dir = [x for x in root_paths_to_trained_models if alg_name in x]
                    
                if len(curr_model_root_save_dir) != 1: 
                    print("__MAIN__: len of curr_model_root_save_dir list not as expected; curr_model_root_save_dir == ", curr_model_root_save_dir)
                    print("__MAIN__: \t alg_name == ", alg_name)
                    print("__MAIN__: \t alg_name_alias == ", alg_name_alias)
                    print("__MAIN__: \t root_paths_to_trained_models == ", root_paths_to_trained_models)
                    raise ValueError()
                curr_model_root_save_dir = curr_model_root_save_dir[0]
                
                model_training_dir = [
                    curr_model_root_save_dir+os.sep+x for x in os.listdir(curr_model_root_save_dir) if os.path.isdir(curr_model_root_save_dir+os.sep+x) and cv_dset_name in x and "fold"+str(f_num) in x
                ]
                
                if len(model_training_dir) != 1: 
                    print("__MAIN__: len of model_training_dir list not as expected; model_training_dir == ", model_training_dir)
                    print("__MAIN__: \t alg_name == ", alg_name)
                    print("__MAIN__: \t alg_name_alias == ", alg_name_alias)
                    print("__MAIN__: \t curr_model_root_save_dir == ", curr_model_root_save_dir)
                    raise ValueError()
                model_training_dir = model_training_dir[0]
                
                trained_model_path = None
                if alg_name != "DCSFA":
                    trained_model_path = [model_training_dir+os.sep+x for x in os.listdir(model_training_dir) if "final_best_model" in x]
                else:
                    trained_model_path = [model_training_dir+os.sep+x for x in os.listdir(model_training_dir) if "dCSFA-NMF-best-model.pt" in x]
                
                if len(trained_model_path) != 1: 
                    print("__MAIN__: len of trained_model_path list not as expected; trained_model_path == ", trained_model_path)
                    print("__MAIN__: \t alg_name == ", alg_name)
                    print("__MAIN__: \t alg_name_alias == ", alg_name_alias)
                    print("__MAIN__: \t model_training_dir == ", model_training_dir)
                    raise ValueError()
                trained_model_path = trained_model_path[0]
                
                trained_model = load_model_for_eval(alg_name_alias, trained_model_path, dynamic_eval=False, d4IC=True) 
                
                # grab granger-causal graphs (both true and estimated) -------------------------------------
                if "REDCLIFF" in alg_name_alias:
                    if "conditional" in trained_model.primary_gc_est_mode: # this case isn't currently handled on next line, for which we pass 'None' in as X
                        print("__MAIN__: WARNING!! OVERWRITING trained_model.primary_gc_est_mode TO fixed_factor_exclusive SETTING FOR SYS-LEVEL INTERPRETATION")
                        trained_model.primary_gc_est_mode = "fixed_factor_exclusive"
                estimated_gcs = get_model_gc_estimates(trained_model, alg_name_alias, len(true_gcs), X=None) 
                
                # compute factor-level stats and create corresponding visualization(s) ---------------------
                for factor_id, (est_fact_gc, true_fact_gc) in enumerate(zip(estimated_gcs, true_gcs)):
                    # prep granger-causal graphs for comparisson
                    if len(est_fact_gc.shape) == 3:
                        est_fact_gc = np.sum(est_fact_gc, axis=2)
                    offDiag_est_fact_gc = mask_diag_elements_of_square_numpy_array(est_fact_gc)
                    normalized_est_fact_gc = normalize_numpy_array(est_fact_gc)
                    normalized_offDiag_est_fact_gc = normalize_numpy_array(offDiag_est_fact_gc)
                    
                    true_fact_gc = np.sum(true_fact_gc, axis=2)
                    offDiag_true_fact_gc = mask_diag_elements_of_square_numpy_array(true_fact_gc)
                    normalized_true_fact_gc = normalize_numpy_array(true_fact_gc)
                    normalized_offDiag_true_fact_gc = normalize_numpy_array(offDiag_true_fact_gc)
                    
                    # compute key statistics of interest between true and estimated causal graphs
                    key_stats_estGC_norm_vs_trueGC_norm = compute_OptimalF1_stats_betw_two_gc_graphs(normalized_est_fact_gc, normalized_true_fact_gc, )
                    key_stats_estGC_normOffDiag_vs_trueGC_normOffDiag = compute_OptimalF1_stats_betw_two_gc_graphs(normalized_offDiag_est_fact_gc, normalized_offDiag_true_fact_gc, )
                    key_stats_estGC_normOffDiagTransposed_vs_trueGC_normOffDiag = compute_OptimalF1_stats_betw_two_gc_graphs(normalized_offDiag_est_fact_gc.T, normalized_offDiag_true_fact_gc, )
                    
                    # record factor-level stats
                    alg_level_stats["factor_"+str(factor_id)] = {
                        "key_stats_estGC_norm_vs_trueGC_norm": key_stats_estGC_norm_vs_trueGC_norm, 
                        "key_stats_estGC_normOffDiag_vs_trueGC_normOffDiag": key_stats_estGC_normOffDiag_vs_trueGC_normOffDiag, 
                        "key_stats_estGC_normOffDiagTransposed_vs_trueGC_normOffDiag": key_stats_estGC_normOffDiagTransposed_vs_trueGC_normOffDiag, 
                    }
                    
                    # plot adj graph comparisson between est./true graphs
                    plot_gc_est_comparisson(
                        normalized_true_fact_gc, normalized_est_fact_gc, curr_alg_save_path+os.sep+"estGC_norm_vs_trueGC_norm_factor_"+str(factor_id)+"_vis.png", 
                        include_lags=False
                    )
                    plot_gc_est_comparisson(
                        normalized_offDiag_true_fact_gc, normalized_offDiag_est_fact_gc, 
                        curr_alg_save_path+os.sep+"estGC_normOffDiag_vs_trueGC_normOffDiag_factor_"+str(factor_id)+"_vis.png", include_lags=False
                    )
                    plot_gc_est_comparisson(
                        normalized_offDiag_true_fact_gc, normalized_offDiag_est_fact_gc.T, 
                        curr_alg_save_path+os.sep+"estGC_normOffDiagTransposed_vs_trueGC_normOffDiag_factor_"+str(factor_id)+"_vis.png", include_lags=False
                    )
                    
                # compute cross-factor means, medians, std. devs, and std. error of the mean for each factor-level stat.
                curr_alg_cross_factor_summaries = dict()
                for f_key_id, f_key in enumerate(alg_level_stats.keys()):
                    assert "factor_" in f_key # sanity check
                    for stat_paradigm_key in alg_level_stats[f_key].keys():
                        if f_key_id == 0:
                            curr_alg_cross_factor_summaries[stat_paradigm_key] = dict()
                        for stat_key in alg_level_stats[f_key][stat_paradigm_key].keys():
                            if f_key_id == 0:
                                curr_alg_cross_factor_summaries[stat_paradigm_key][stat_key] = []
                            curr_alg_cross_factor_summaries[stat_paradigm_key][stat_key].append(alg_level_stats[f_key][stat_paradigm_key][stat_key])
                for stat_paradigm_key in curr_alg_cross_factor_summaries.keys():
                    assert stat_paradigm_key not in alg_level_stats.keys() # sanity check
                    alg_level_stats[stat_paradigm_key] = dict()
                    for stat_key in curr_alg_cross_factor_summaries[stat_paradigm_key].keys():
                        alg_level_stats[stat_paradigm_key][stat_key+"_vals_across_factors"] = copy.deepcopy(curr_alg_cross_factor_summaries[stat_paradigm_key][stat_key])
                        alg_level_stats[stat_paradigm_key][stat_key+"_mean_across_factors"] = np.mean(curr_alg_cross_factor_summaries[stat_paradigm_key][stat_key])
                        alg_level_stats[stat_paradigm_key][stat_key+"_median_across_factors"] = np.median(curr_alg_cross_factor_summaries[stat_paradigm_key][stat_key])
                        alg_level_stats[stat_paradigm_key][stat_key+"_std_dev_across_factors"] = np.std(curr_alg_cross_factor_summaries[stat_paradigm_key][stat_key])
                        alg_level_stats[stat_paradigm_key][stat_key+"_mean_std_err_across_factors"] = alg_level_stats[stat_paradigm_key][stat_key+"_std_dev_across_factors"] / np.sqrt(1.*len(alg_level_stats[stat_paradigm_key][stat_key+"_vals_across_factors"]))
                
                # record cross-factor stats for curr alg in dictionary
                fold_level_stats[alg_name] = alg_level_stats
            
            # make bar/whisker plot of cross-factor stats 
            assert len(fold_level_stats.keys()) == len(ALGORITHMS_TO_EVALUATE) # sanity check
            assert sorted(list(fold_level_stats.keys())) == sorted(ALGORITHMS_TO_EVALUATE) # sanity check
            stat_paradigms = None
            for temp_alg_name_key in fold_level_stats.keys():
                stat_paradigms = [x for x in fold_level_stats[temp_alg_name_key].keys() if "factor" not in x]
                break # we only need the paradigm names from the first alg key, because they should all be shared across algs
            stat_val_keys = [x for x in fold_level_stats[ALGORITHMS_TO_EVALUATE[0]][stat_paradigms[0]].keys() if "_vals_across_factors" in x]
            
            for stat_paradigm in stat_paradigms:
                for stat_val_key in stat_val_keys:
                    curr_results_by_alg = {a_name: fold_level_stats[a_name][stat_paradigm][stat_val_key] for a_name in fold_level_stats.keys()}
                    print("__MAIN__: \t f_num==", f_num, " stat_paradigm==", stat_paradigm," stat_val_key==", stat_val_key," curr_results_by_alg==", curr_results_by_alg, flush=True)
                    make_scatter_and_stdErrOfMean_plot_overlay_vis(
                        curr_results_by_alg, curr_fold_save_path+os.sep+"factor_level_"+stat_paradigm+"_"+stat_val_key+"_by_algorithm.png", 
                        "Comparing Factor-Level "+stat_val_key[:-1*len("_vals_across_factors")]+" Between Algorithms", "Algorithm", stat_val_key, alpha=0.5
                    )
                        
            # record cross-fold stats in dictionary
            cv_level_stats["fold_"+str(f_num)+"_details"] = fold_level_stats
            for sp_ind, stat_paradigm in enumerate(stat_paradigms):
                if f_num == 0:
                    assert stat_paradigm not in cv_level_stats.keys()
                    cv_level_stats[stat_paradigm] = dict()
                for alg_id, alg_name in enumerate(ALGORITHMS_TO_EVALUATE):
                    if f_num == 0:
                        cv_level_stats[stat_paradigm][alg_name] = dict()
                    for stat_val_key in stat_val_keys:
                        if f_num == 0:
                            cv_level_stats[stat_paradigm][alg_name][stat_val_key] = []
                        cv_level_stats[stat_paradigm][alg_name][stat_val_key] = cv_level_stats[stat_paradigm][alg_name][stat_val_key] + fold_level_stats[alg_name][stat_paradigm][stat_val_key]
    
        # compute cross-fold means, medians, std. devs, and std. error of the mean for each cross-factor stat.
        stat_paradigms_in_cv = []
        stat_val_keys_in_cv = set()
        cv_level_stats_update = dict()
        for key in cv_level_stats.keys():
            if "_vs_" in key: # the hallmark of stat_paradigm keys
                stat_paradigms_in_cv.append(key)
                cv_level_stats_update[key] = dict()
                for alg_name in cv_level_stats[key].keys():
                    cv_level_stats_update[key][alg_name] = dict()
                    for stat_val_key in cv_level_stats[key][alg_name].keys(): 
                        stat_val_keys_in_cv.add(stat_val_key)
                        stat_key = stat_val_key[:-1*len("_vals_across_factors")]
                        assert len(cv_level_stats[key][alg_name][stat_val_key]) == CURR_NUM_FACTORS_IN_CV_DATA_SETS*NUM_FOLDS_PER_CV_DATASET # sanity check
                        cv_level_stats_update[key][alg_name][stat_key+"_mean_across_factors"] = np.mean(cv_level_stats[key][alg_name][stat_val_key])
                        cv_level_stats_update[key][alg_name][stat_key+"_median_across_factors"] = np.median(cv_level_stats[key][alg_name][stat_val_key])
                        cv_level_stats_update[key][alg_name][stat_key+"_std_dev_across_factors"] = np.std(cv_level_stats[key][alg_name][stat_val_key])
                        cv_level_stats_update[key][alg_name][stat_key+"_mean_std_err_across_factors"] = cv_level_stats_update[key][alg_name][stat_key+"_std_dev_across_factors"] / np.sqrt(1.*len(cv_level_stats[key][alg_name][stat_val_key]))
        for key in cv_level_stats_update:
            for alg_name in cv_level_stats_update[key]:
                for stat_val_key in stat_val_keys_in_cv: 
                    stat_key = stat_val_key[:-1*len("_vals_across_factors")]
                    cv_level_stats[key][alg_name][stat_key+"_mean_across_factors"] = cv_level_stats_update[key][alg_name][stat_key+"_mean_across_factors"]
                    cv_level_stats[key][alg_name][stat_key+"_median_across_factors"] = cv_level_stats_update[key][alg_name][stat_key+"_median_across_factors"]
                    cv_level_stats[key][alg_name][stat_key+"_std_dev_across_factors"] = cv_level_stats_update[key][alg_name][stat_key+"_std_dev_across_factors"]
                    cv_level_stats[key][alg_name][stat_key+"_mean_std_err_across_factors"] = cv_level_stats_update[key][alg_name][stat_key+"_mean_std_err_across_factors"]
        
        # make bar/whisker plot of cross-fold stats
        for s_para in stat_paradigms_in_cv:
            for s_val_key in stat_val_keys_in_cv:
                curr_full_cv_results_by_alg = {alg: cv_level_stats[s_para][alg][s_val_key] for alg in ALGORITHMS_TO_EVALUATE}
                print("__MAIN__: stat_paradigm==", s_para," stat_val_key==", s_val_key," curr_results_by_alg==", curr_full_cv_results_by_alg, flush=True)
                make_scatter_and_stdErrOfMean_plot_overlay_vis(
                    curr_full_cv_results_by_alg, curr_cv_save_path+os.sep+"factor_level_"+s_para+"_"+s_val_key+"_by_algorithm.png", 
                    "Comparing Factor-Level "+s_val_key[:-1*len("_vals_across_factors")]+" Between Algorithms", "Algorithm", s_val_key, alpha=0.5
                )        
        print("\n\n\n")
        
        # save cv-specific results/analyses
        with open(curr_cv_save_path+os.sep+"results_summary.pkl", 'wb') as outfile:
            pkl.dump(cv_level_stats, outfile)
        
        full_comparrisson_summary[cv_dset_name] = cv_level_stats
    
    # save stats. dict. to save_root_path
    with open(save_root_path+os.sep+"full_comparrisson_summary.pkl", 'wb') as outfile:
        pkl.dump(full_comparrisson_summary, outfile)
    
    print("__MAIN__: DONE !!!")
    pass