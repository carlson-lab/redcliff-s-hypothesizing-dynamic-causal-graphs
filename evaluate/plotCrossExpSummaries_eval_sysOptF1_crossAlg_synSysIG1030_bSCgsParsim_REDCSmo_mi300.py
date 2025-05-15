import shutil
import os
import joblib
import argparse
import json
import pickle as pkl
import numpy as np

from general_utils.plotting import plot_cross_experiment_summary



def get_alg_name_alias(orig_name):
    if orig_name == 'REDCLIFF_S_CMLP_WithSmoothing':
        return 'REDCLIFF-S (cMLP)'#'Smoothed REDCLIFF-S (cMLP)'
    elif orig_name == 'CMLP':
        return 'cMLP'
    elif orig_name == 'CLSTM':
        return 'cLSTM'
    elif orig_name == 'DCSFA':
        return 'dCSFA-NMF'
    elif orig_name == 'DYNOTEARS_Vanilla':
        return 'DYNOTEARS'#'DYNOTEARS (Vanilla)'
    elif orig_name == 'NAVAR_CMLP':
        return 'NAVAR-P'#'NAVAR (cMLP)'
    elif orig_name == 'NAVAR_CLSTM':
        return 'NAVAR-R'#'NAVAR (cLSTM)'
    return orig_name

def get_system_complexity_from_name(sys_name, C_func):
    """ assumes sys_name == 'nN<int>_nE<int>_nF<int>' """
    split_name = sys_name.split("_")
    nc = int(split_name[0][2:])
    ne = int(split_name[1][2:])
    nk = int(split_name[2][2:])
    c_score = C_func((nc, ne, nk))
    return c_score


if __name__=="__main__":
    parse=argparse.ArgumentParser(description='Default parameters for evaluation')
    parse.add_argument(
        "-cached_args_file",
        default="plotCrossExpSummaries_eval_sysOptF1_crossAlg_synSysIG1030_bSCgsParsim_REDCSmo_mi300_v01202025_cached_args_sensitive.txt", # WARNING: edit this line when running new experiments on SLURM server
        help="/path/to/cached_args.txt should contain a json object with updated arg values",
    )
    args = parse.parse_args()
    args_dict = None
    
    # read in sensitive args
    print("__MAIN__: LOADING ARGS", flush=True)
    eval_files_root_path = None
    save_root_path = None
    with open(args.cached_args_file, 'r') as infile:
        args_dict = json.load(infile)
        eval_files_root_path = args_dict["eval_files_root_path"]
        save_root_path = args_dict["save_root_path"]
    
    alg_names = []
    mean_colors = ["darkorange", "darkred", "mediumvioletred", "darkslateblue", "indigo"]
    sem_colors = ["orangered", "tomato", "lightcoral", "slategrey", "mediumpurple"]

    Complexity = lambda x: (x[1]/(x[0]**2 - x[0]))**(-1)# INVERSE SPARSITY, OLD COMPLEXITY WAS FORMERLY: x[1]/(x[0]**2 - x[0])*(x[1]/x[2])
    moderateC_lower_bound = 7.
    moderateC_upper_bound = 13.
    system_details = {
        "numF2_numSF2_numN12_numE11_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN12_nE11_nF2", "dataset_complexity": None, },  
        "numF2_numSF2_numN12_numE12_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN12_nE12_nF2", "dataset_complexity": None, },  
        "numF2_numSF2_numN12_numE22_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN12_nE22_nF2", "dataset_complexity": None, },  
        "numF2_numSF2_numN12_numE33_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN12_nE33_nF2", "dataset_complexity": None, },  
        "numF2_numSF2_numN12_numE44_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN12_nE44_nF2", "dataset_complexity": None, },  
        "numF2_numSF2_numN12_numE55_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN12_nE55_nF2", "dataset_complexity": None, },  
        "numF2_numSF2_numN3_numE1_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN3_nE1_nF2", "dataset_complexity": None, },  
        "numF2_numSF2_numN3_numE2_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN3_nE2_nF2", "dataset_complexity": None, },  
        "numF2_numSF2_numN6_numE10_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN6_nE10_nF2", "dataset_complexity": None, },  
        "numF2_numSF2_numN6_numE12_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN6_nE12_nF2", "dataset_complexity": None, },  
        "numF2_numSF2_numN6_numE2_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN6_nE2_nF2", "dataset_complexity": None, },  
        "numF2_numSF2_numN6_numE4_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN6_nE4_nF2", "dataset_complexity": None, },  
        "numF2_numSF2_numN6_numE6_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN6_nE6_nF2", "dataset_complexity": None, },  
        "numF2_numSF2_numN6_numE8_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN6_nE8_nF2", "dataset_complexity": None, },  
        "numF3_numSF3_numN12_numE11_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN12_nE11_nF3", "dataset_complexity": None, },  
        "numF3_numSF3_numN12_numE12_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN12_nE12_nF3", "dataset_complexity": None, },  
        "numF3_numSF3_numN12_numE22_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN12_nE22_nF3", "dataset_complexity": None, },  
        "numF3_numSF3_numN12_numE33_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN12_nE33_nF3", "dataset_complexity": None, },  
        "numF3_numSF3_numN3_numE1_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN3_nE1_nF3", "dataset_complexity": None, },  
        "numF3_numSF3_numN6_numE2_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN6_nE2_nF3", "dataset_complexity": None, },  
        "numF3_numSF3_numN6_numE4_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN6_nE4_nF3", "dataset_complexity": None, },  
        "numF3_numSF3_numN6_numE6_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN6_nE6_nF3", "dataset_complexity": None, },  
        "numF3_numSF3_numN6_numE8_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN6_nE8_nF3", "dataset_complexity": None, },  
        "numF4_numSF4_numN12_numE11_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN12_nE11_nF4", "dataset_complexity": None, },  
        "numF4_numSF4_numN12_numE12_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN12_nE12_nF4", "dataset_complexity": None, },  
        "numF4_numSF4_numN12_numE22_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN12_nE22_nF4", "dataset_complexity": None, },  
        "numF4_numSF4_numN3_numE1_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN3_nE1_nF4", "dataset_complexity": None, },  
        #"numF4_numSF4_numN6_numE2_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN6_nE2_nF4", "dataset_complexity": None, },  
        "numF4_numSF4_numN6_numE4_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN6_nE4_nF4", "dataset_complexity": None, },  
        "numF4_numSF4_numN6_numE6_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN6_nE6_nF4", "dataset_complexity": None, },  
        "numF5_numSF5_numN12_numE11_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN12_nE11_nF5", "dataset_complexity": None, },  
        "numF5_numSF5_numN12_numE12_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN12_nE12_nF5", "dataset_complexity": None, },  
        "numF5_numSF5_numN12_numE22_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN12_nE22_nF5", "dataset_complexity": None, },  
        "numF5_numSF5_numN3_numE1_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN3_nE1_nF5", "dataset_complexity": None, },  
        "numF5_numSF5_numN6_numE2_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN6_nE2_nF5", "dataset_complexity": None, },  
        "numF5_numSF5_numN6_numE4_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN6_nE4_nF5", "dataset_complexity": None, },  
        "numF6_numSF6_numN12_numE11_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN12_nE11_nF6", "dataset_complexity": None, },  
        "numF6_numSF6_numN12_numE12_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN12_nE12_nF6", "dataset_complexity": None, },  
        "numF6_numSF6_numN6_numE2_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN6_nE2_nF6", "dataset_complexity": None, },  
        "numF6_numSF6_numN6_numE4_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN6_nE4_nF6", "dataset_complexity": None, },  
        "numF7_numSF7_numN12_numE11_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN12_nE11_nF7", "dataset_complexity": None, },  
        "numF7_numSF7_numN12_numE12_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN12_nE12_nF7", "dataset_complexity": None, },  
        "numF7_numSF7_numN6_numE2_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN6_nE2_nF7", "dataset_complexity": None, },  
        "numF8_numSF8_numN12_numE11_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN12_nE11_nF8", "dataset_complexity": None, },  
        "numF8_numSF8_numN12_numE12_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN12_nE12_nF8", "dataset_complexity": None, },  
        "numF8_numSF8_numN6_numE2_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN6_nE2_nF8", "dataset_complexity": None, },  
        "numF9_numSF9_numN12_numE11_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN12_nE11_nF9", "dataset_complexity": None, },  
        "numF9_numSF9_numN12_numE12_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN12_nE12_nF9", "dataset_complexity": None, },  
        "numF9_numSF9_numN6_numE2_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN6_nE2_nF9", "dataset_complexity": None, },  
        #"numF10_numSF10_numN12_numE11_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN12_nE11_nF10", "dataset_complexity": None, },  
        #"numF10_numSF10_numN6_numE2_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data": {"dataset_name": "nN6_nE2_nF10", "dataset_complexity": None, },  
    }
    # populate system_details with complexity scores
    low_complexity_datasets = []
    moderate_complexity_datasets = []
    high_complexity_datasets = []
    for sys_key in system_details.keys():
        system_details[sys_key]["dataset_complexity"] = get_system_complexity_from_name(system_details[sys_key]["dataset_name"], Complexity)
        if system_details[sys_key]["dataset_complexity"] <= moderateC_lower_bound:
            low_complexity_datasets.append(system_details[sys_key]["dataset_name"])
        elif system_details[sys_key]["dataset_complexity"] > moderateC_upper_bound:
            high_complexity_datasets.append(system_details[sys_key]["dataset_name"])
        else:
            moderate_complexity_datasets.append(system_details[sys_key]["dataset_name"])
    
    complexity_categories = ["Low", "Moderate", "High"]
    dset_names_by_complexity = [low_complexity_datasets, moderate_complexity_datasets, high_complexity_datasets]
    for cat_ind, (complex_cat, complex_dset_names) in enumerate(zip(complexity_categories, dset_names_by_complexity)):
        print("__MAIN__: complex_cat == ", complex_cat, " :: len(complex_dset_names) == ", len(complex_dset_names))
        alg_performance_means_tOpt = None
        alg_performance_sems_tOpt = None
        alg_vREDC_performance_means_tOpt = None
        alg_vREDC_performance_sems_tOpt = None
        
        for sys_ind, sys_key in enumerate(system_details.keys()):
            print("__MAIN__: \t sys_key == ", sys_key)
            # check current system complexity category
            meets_complexity_criteria = False
            if complex_cat == "Low" and system_details[sys_key]["dataset_complexity"] <= moderateC_lower_bound: # low-complexity_systems
                meets_complexity_criteria = True
            elif complex_cat == "High" and system_details[sys_key]["dataset_complexity"] > moderateC_upper_bound: # high-complexity_systems
                meets_complexity_criteria = True
            elif complex_cat == "Moderate" and system_details[sys_key]["dataset_complexity"] > moderateC_lower_bound and system_details[sys_key]["dataset_complexity"] <= moderateC_upper_bound: # moderate-complexity_systems
                meets_complexity_criteria = True
            
            print("__MAIN__: \t meets_complexity_criteria == ", meets_complexity_criteria)
            if meets_complexity_criteria:
                if sys_key not in os.listdir(eval_files_root_path):
                    print("WARNING: sys_key == ", sys_key)
                    print("WARNING: eval_files_root_path == ", eval_files_root_path)
                assert sys_key in os.listdir(eval_files_root_path)
                curr_alg_performance_means_tOpt = []
                curr_alg_performance_sems_tOpt = []
                curr_alg_vREDC_performance_means_tOpt = []
                curr_alg_vREDC_performance_sems_tOpt = []

                # read in bsOH results
                curr_synth_results = None
                with open(eval_files_root_path+os.sep+sys_key+os.sep+"full_comparrisson_summary.pkl", "rb") as f:
                    curr_synth_results = pkl.load(f)

                assert len(curr_synth_results.keys()) == 1
                cv_key = list(curr_synth_results.keys())[0]
                for i, alg_key in enumerate(curr_synth_results[cv_key]['key_stats_estGC_normOffDiag_vs_trueGC_normOffDiag'].keys()):
                    curr_alg_alias = get_alg_name_alias(alg_key)
                    if curr_alg_alias not in alg_names: 
                        alg_names.append(curr_alg_alias)
                    else:
                        assert get_alg_name_alias(alg_key) == alg_names[i]

                    curr_alg_eval_stats = curr_synth_results[cv_key]['key_stats_estGC_normOffDiag_vs_trueGC_normOffDiag'][alg_key]
                    for stat_key in curr_alg_eval_stats.keys():
                        if 'f1' in stat_key and "mean_across_factors" in stat_key:
                            curr_alg_performance_means_tOpt.append(curr_alg_eval_stats[stat_key])
                        elif 'f1' in stat_key and "mean_std_err_across_factors" in stat_key:
                            curr_alg_performance_sems_tOpt.append(curr_alg_eval_stats[stat_key])
                        elif 'f1' in stat_key and "vals_across_factors" in stat_key:
                            curr_diffs = [x-y for x, y in zip(curr_synth_results[cv_key]['key_stats_estGC_normOffDiag_vs_trueGC_normOffDiag']['REDCLIFF_S_CMLP_WithSmoothing'][stat_key], curr_alg_eval_stats[stat_key])]
                            curr_alg_vREDC_performance_means_tOpt.append(np.mean(curr_diffs))
                            curr_alg_vREDC_performance_sems_tOpt.append(np.std(curr_diffs)/np.sqrt(1.*len(curr_diffs)))

                    assert len(curr_alg_performance_means_tOpt) == i+1
                    assert len(curr_alg_performance_sems_tOpt) == i+1
                    assert len(curr_alg_vREDC_performance_means_tOpt) == i+1
                    assert len(curr_alg_vREDC_performance_sems_tOpt) == i+1

                if alg_performance_means_tOpt is None: #sys_ind == 0:
                    alg_performance_means_tOpt = [None for _ in range(len(complex_dset_names)) for _ in range(len(alg_names))]
                    alg_performance_sems_tOpt = [None for _ in range(len(complex_dset_names)) for _ in range(len(alg_names))]
                    alg_vREDC_performance_means_tOpt = [None for _ in range(len(complex_dset_names)) for _ in range(len(alg_names))]
                    alg_vREDC_performance_sems_tOpt = [None for _ in range(len(complex_dset_names)) for _ in range(len(alg_names))]
                
                d = complex_dset_names.index(system_details[sys_key]["dataset_name"]) # curr_loc_of_dataset_in_complexity_sorted_list
                for i in range(len(alg_names)):
                    if curr_alg_performance_means_tOpt[i] is None or np.isfinite(curr_alg_performance_means_tOpt[i]):
                        alg_performance_means_tOpt[d*len(alg_names) + i] = curr_alg_performance_means_tOpt[i]
                    else:
                        alg_performance_means_tOpt[d*len(alg_names) + i] = np.nan
                    if curr_alg_performance_sems_tOpt[i] is None or np.isfinite(curr_alg_performance_sems_tOpt[i]):
                        alg_performance_sems_tOpt[d*len(alg_names) + i] = curr_alg_performance_sems_tOpt[i]
                    else:
                        alg_performance_sems_tOpt[d*len(alg_names) + i] = np.nan
                    if curr_alg_vREDC_performance_means_tOpt[i] is None or np.isfinite(curr_alg_vREDC_performance_means_tOpt[i]):
                        alg_vREDC_performance_means_tOpt[d*len(alg_names) + i] = curr_alg_vREDC_performance_means_tOpt[i]
                    else:
                        alg_vREDC_performance_means_tOpt[d*len(alg_names) + i] = np.nan
                    if curr_alg_vREDC_performance_sems_tOpt[i] is None or np.isfinite(curr_alg_vREDC_performance_sems_tOpt[i]):
                        alg_vREDC_performance_sems_tOpt[d*len(alg_names) + i] = curr_alg_vREDC_performance_sems_tOpt[i]
                    else:
                        alg_vREDC_performance_sems_tOpt[d*len(alg_names) + i] = np.nan
        
        num_plot_segmentations = (len(complex_dset_names)//7) + 1
        for seg in range(num_plot_segmentations):
            name_seg_start = seg*7
            perf_seg_start = seg*7*len(alg_names)
            name_seg_stop = (seg+1)*7
            perf_seg_stop = (seg+1)*7*len(alg_names)
            
            x_domain_lim = None
            plot_cross_experiment_summary(
                save_root_path+os.sep+complex_cat+"_complexity_cross_synth_edge_prediction_plot"+str(seg)+".png", 
                alg_performance_means_tOpt[perf_seg_start:perf_seg_stop], 
                alg_performance_sems_tOpt[perf_seg_start:perf_seg_stop], 
                alg_names, 
                complex_dset_names[name_seg_start:name_seg_stop], 
                mean_colors, 
                sem_colors, 
                'Synthetic System Edge Prediction: '+complex_cat+" Complexity", 
                "Synthetic System Name ("+r'$n_c$'+"-"+r'$n_e$'+"-"+r'$n_k$'+")", 
                'Avg. Optimal F1-Score '+r'$\pm$'+' Std. Err. of the Mean', 
                bar_width=0.9, 
                fig_width=10, 
                fig_height=14, 
                FONT_SMALL_SIZE=18, 
                FONT_MEDIUM_SIZE=20, 
                FONT_BIGGER_SIZE=22, 
                x_domain_lim=x_domain_lim
            )    
            
            if complex_cat == "High":
                x_domain_lim = None
            plot_cross_experiment_summary(
                save_root_path+os.sep+complex_cat+"_complexity_cross_pairwise_factorLevel_REDCImprovement_synth_edge_prediction_plot"+str(seg)+".png", 
                alg_vREDC_performance_means_tOpt[perf_seg_start:perf_seg_stop], 
                alg_vREDC_performance_sems_tOpt[perf_seg_start:perf_seg_stop], 
                alg_names, 
                complex_dset_names[name_seg_start:name_seg_stop], 
                mean_colors, 
                sem_colors, 
                ' Pair-wise Improvement REDCLIFF-S With Smoothing for \n Synthetic System Edge Prediction: '+complex_cat+" Complexity ", 
                "Synthetic System Name ("+r'$n_c$'+"-"+r'$n_e$'+"-"+r'$n_k$'+")", 
                'Avg. Difference in Optimal F1-Score '+r'$\pm$'+' Std. Err. of the Mean', 
                bar_width=0.9, 
                fig_width=10, 
                fig_height=14, 
                FONT_SMALL_SIZE=18, 
                FONT_MEDIUM_SIZE=20, 
                FONT_BIGGER_SIZE=22, 
                x_domain_lim=x_domain_lim
            )
            
    with open(save_root_path+os.sep+"system_details.pkl", 'wb') as outfile:
        pkl.dump(system_details, outfile)

    print("__MAIN__: DONE!!!")
    pass