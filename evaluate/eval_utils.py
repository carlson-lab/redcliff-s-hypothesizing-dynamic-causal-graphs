import copy
import numpy as np
from sklearn.metrics import roc_auc_score
import os
import pickle as pkl
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from scipy import stats as scistats

from general_utils.metrics import compute_cosine_similarity, compute_mse, deltacon0, deltacon0_with_directed_degrees, deltaffinity, path_length_mse, linregress, spearmanr
from general_utils.metrics import compute_covariance_betw_two_variables, compute_spearman_numerator_cov_of_ranked_variables, convert_variable_to_rank_variable, compute_covariance_betw_two_variables
from general_utils.metrics import compute_positive_likelihood_ratio, compute_negative_likelihood_ratio, compute_sensitivity, compute_specificity, compute_f1, compute_optimal_f1
from general_utils.model_utils import call_model_eval_method, get_data_for_model_training
from general_utils.misc import get_avg_cosine_similarity_between_combos, sort_unsupervised_estimates, obtain_factor_score_weightings_across_recording, obtain_factor_score_classifications_across_recording
from general_utils.plotting import plot_gc_est_comparisson, plot_system_state_score_comparisson, plot_avg_system_state_score_comparisson, plot_estimated_vs_true_curve
from general_utils.input_argument_utils import read_in_data_args

from models.dcsfa_nmf_vanillaDirSpec import FullDCSFAModel as FullDCSFAModel_GCv2




def read_in_true_causal_graphs_for_all_datasets(DATA_SETS, files_of_cached_data_args, data_vis_root_save_path):
    true_causal_graphs = []
    for (dset_name, dset_args) in zip(DATA_SETS, files_of_cached_data_args):
        data_vis_save_path = data_vis_root_save_path+os.sep+dset_name
        if not os.path.exists(data_vis_save_path):
            os.mkdir(data_vis_save_path)
        args_dict = dict()
        args_dict["model_type"] = "REDCLIFF_S_CMLP" # this setting will simply be used to read in the most generic format of the true causal graphs
        args_dict["save_root_path"] = data_vis_save_path
        args_dict["data_set_name"] = dset_name
        args_dict["data_cached_args_file"] = dset_args
        assert args_dict["data_cached_args_file"] in files_of_cached_data_args # sanity check
        args_dict["true_GC_factors"] = None # need to reset args_dict["true_GC_factors"] to ensure it is not carried over from previous evaluation
        args_dict = read_in_data_args(args_dict, include_gc_views_for_eval=False, read_in_gc_factors_for_eval=True)
        true_causal_graphs.append(args_dict["true_GC_factors"])
    return true_causal_graphs


def compute_edgeLockPerformanceV4_stats_betw_two_gc_graphs(stat_paradigm, est_A_hist, true_A_hist, smoothing_window_size=1, plot_save_path=None):
    """
    Compute Various Edge-Level Statistics Between the Two Histories Provided
    """
    for ind, eA in enumerate(est_A_hist):
        try:
            est_A_hist[ind] = eA.detach().numpy()
        except:
            pass
    for ind, tA in enumerate(true_A_hist):
        try:
            true_A_hist[ind] = tA.detach().numpy()
        except:
            pass
    assert est_A_hist[0].shape == true_A_hist[0].shape
    assert len(est_A_hist[0].shape) == 2
    assert est_A_hist[0].shape[0] == est_A_hist[0].shape[1]
    assert len(est_A_hist) == len(true_A_hist)
    num_channels = est_A_hist[0].shape[0]
    orig_gc_shape = est_A_hist[0].shape
    
    # initialize stat_paradigm stat objects to be computed/recorded (both smoothed activation and ranked_smoothAct stats)
    key_stats = dict()

    # smooth edge activations across time
    smoothed_est_A_hist = [np.zeros(orig_gc_shape) for _ in range(len(est_A_hist)-smoothing_window_size)]
    smoothed_true_A_hist = [np.zeros(orig_gc_shape) for _ in range(len(true_A_hist)-smoothing_window_size)]
    for row_ind in range(num_channels):
        for col_ind in range(num_channels):
            curr_est_edge_history = [est_A_hist[t][row_ind, col_ind] for t in range(len(est_A_hist))]
            curr_true_edge_history = [true_A_hist[t][row_ind, col_ind] for t in range(len(true_A_hist))]
            smoothed_est_edge_hist = [np.mean(curr_est_edge_history[t:t+smoothing_window_size]) for t in range(len(curr_est_edge_history)-smoothing_window_size)]
            smoothed_true_edge_hist = [np.mean(curr_true_edge_history[t:t+smoothing_window_size]) for t in range(len(curr_true_edge_history)-smoothing_window_size)]
            for w, (s_est_edge_act, s_true_edge_act) in enumerate(zip(smoothed_est_edge_hist, smoothed_true_edge_hist)):
                smoothed_est_A_hist[w][row_ind, col_ind] = s_est_edge_act
                smoothed_true_A_hist[w][row_ind, col_ind] = s_true_edge_act

    # rank across smoothed edge activations at each window
    ranked_smoothed_true_A_hist = []
    for w, s_true_A in enumerate(smoothed_true_A_hist):
        ranked_smoothed_true_A_hist.append(convert_variable_to_rank_variable(s_true_A, method='dense'))

    # compute stats
    for row_ind in range(num_channels):
        for col_ind in range(num_channels):
            curr_true_smooth_ranks_across_windows = [rstA[row_ind, col_ind] for rstA in ranked_smoothed_true_A_hist]
            curr_true_avg_smooth_rank = np.mean(curr_true_smooth_ranks_across_windows)
            
            #if curr_true_avg_smooth_rank > 1. and row_ind != col_ind: # filter out edges that have no true activation value (rank==1), as the resulting stats would all be nan, inf, or 0.
            curr_true_smooth_activs_across_windows = [stA[row_ind, col_ind] for stA in smoothed_true_A_hist]
            curr_est_smooth_activs_across_windows = [seA[row_ind, col_ind] for seA in smoothed_est_A_hist]

            # compute activation history stats according to stat_paradigm
            curr_paradigm_smooth_activ_hist_stat = None
            if stat_paradigm == "PearsonCorrelation":
                _, _, pearson_r, pearson_p, _ = linregress(curr_est_smooth_activs_across_windows, curr_true_smooth_activs_across_windows)
                curr_paradigm_smooth_activ_hist_stat = {"pearson_r": pearson_r, "pearson_p": pearson_p, }
            else:
                raise NotImplemented()

            key_stats[str(row_ind)+'<-'+str(col_ind)] = {stat_paradigm+'_curr_paradigm_smooth_activ_hist_stat': curr_paradigm_smooth_activ_hist_stat, }
    
    return key_stats


def compute_edgeLockPerformanceV3_stats_betw_two_gc_graphs(stat_paradigm, est_A_hist, true_A_hist, smoothing_window_size=1, plot_save_path=None):
    """
    Compute Various Edge-Level Statistics Between the Two Histories Provided
    """
    for ind, eA in enumerate(est_A_hist):
        try:
            est_A_hist[ind] = eA.detach().numpy()
        except:
            pass
    for ind, tA in enumerate(true_A_hist):
        try:
            true_A_hist[ind] = tA.detach().numpy()
        except:
            pass
    assert est_A_hist[0].shape == true_A_hist[0].shape
    assert len(est_A_hist[0].shape) == 2
    assert est_A_hist[0].shape[0] == est_A_hist[0].shape[1]
    assert len(est_A_hist) == len(true_A_hist)
    num_channels = est_A_hist[0].shape[0]
    orig_gc_shape = est_A_hist[0].shape
    
    # initialize stat_paradigm stat objects to be computed/recorded (both smoothed activation and ranked_smoothAct stats)
    key_stats = dict()

    # smooth edge activations across time
    smoothed_est_A_hist = [np.zeros(orig_gc_shape) for _ in range(len(est_A_hist)-smoothing_window_size)]
    smoothed_true_A_hist = [np.zeros(orig_gc_shape) for _ in range(len(true_A_hist)-smoothing_window_size)]
    for row_ind in range(num_channels):
        for col_ind in range(num_channels):
            curr_est_edge_history = [est_A_hist[t][row_ind, col_ind] for t in range(len(est_A_hist))]
            curr_true_edge_history = [true_A_hist[t][row_ind, col_ind] for t in range(len(true_A_hist))]
            smoothed_est_edge_hist = [np.mean(curr_est_edge_history[t:t+smoothing_window_size]) for t in range(len(curr_est_edge_history)-smoothing_window_size)]
            smoothed_true_edge_hist = [np.mean(curr_true_edge_history[t:t+smoothing_window_size]) for t in range(len(curr_true_edge_history)-smoothing_window_size)]
            for w, (s_est_edge_act, s_true_edge_act) in enumerate(zip(smoothed_est_edge_hist, smoothed_true_edge_hist)):
                smoothed_est_A_hist[w][row_ind, col_ind] = s_est_edge_act
                smoothed_true_A_hist[w][row_ind, col_ind] = s_true_edge_act

    # rank across smoothed edge activations at each window
    ranked_smoothed_true_A_hist = []
    for w, s_true_A in enumerate(smoothed_true_A_hist):
        ranked_smoothed_true_A_hist.append(convert_variable_to_rank_variable(s_true_A, method='dense'))

    # compute stats
    for row_ind in range(num_channels):
        for col_ind in range(num_channels):
            curr_true_smooth_ranks_across_windows = [rstA[row_ind, col_ind] for rstA in ranked_smoothed_true_A_hist]
            curr_true_avg_smooth_rank = np.mean(curr_true_smooth_ranks_across_windows)
            
            if curr_true_avg_smooth_rank > 1. and row_ind != col_ind: # filter out edges that have no true activation value (rank==1), as the resulting stats would all be nan, inf, or 0.
                curr_true_smooth_activs_across_windows = [stA[row_ind, col_ind] for stA in smoothed_true_A_hist]
                curr_est_smooth_activs_across_windows = [seA[row_ind, col_ind] for seA in smoothed_est_A_hist]
                    
                # compute activation history stats according to stat_paradigm
                curr_paradigm_smooth_activ_hist_stat = None
                if stat_paradigm == "PearsonCorrelation":
                    _, _, pearson_r, pearson_p, _ = linregress(curr_est_smooth_activs_across_windows, curr_true_smooth_activs_across_windows)
                    curr_paradigm_smooth_activ_hist_stat = {"pearson_r": pearson_r, "pearson_p": pearson_p, }
                else:
                    raise NotImplemented()
                
                key_stats[str(row_ind)+'<-'+str(col_ind)] = {stat_paradigm+'_curr_paradigm_smooth_activ_hist_stat': curr_paradigm_smooth_activ_hist_stat, }
    
    return key_stats


def compute_edgeRankPerformanceV2_stats_betw_two_gc_graphs(stat_paradigm, est_A_hist, true_A_hist, smoothing_window_size=1, plot_save_path=None):
    """
    Compute Various Edge-Level Statistics Between the Two Histories Provided
    """
    for ind, eA in enumerate(est_A_hist):
        try:
            est_A_hist[ind] = eA.detach().numpy()
        except:
            pass
    for ind, tA in enumerate(true_A_hist):
        try:
            true_A_hist[ind] = tA.detach().numpy()
        except:
            pass
    assert est_A_hist[0].shape == true_A_hist[0].shape
    assert len(est_A_hist[0].shape) == 2
    assert est_A_hist[0].shape[0] == est_A_hist[0].shape[1]
    assert len(est_A_hist) == len(true_A_hist)
    num_channels = est_A_hist[0].shape[0]
    orig_gc_shape = est_A_hist[0].shape
    
    # initialize stat_paradigm stat objects to be computed/recorded (both smoothed activation and ranked_smoothAct stats)
    key_stats = dict()

    # smooth edge activations across time
    smoothed_est_A_hist = [np.zeros(orig_gc_shape) for _ in range(len(est_A_hist)-smoothing_window_size)]
    smoothed_true_A_hist = [np.zeros(orig_gc_shape) for _ in range(len(true_A_hist)-smoothing_window_size)]
    for row_ind in range(num_channels):
        for col_ind in range(num_channels):
            curr_est_edge_history = [est_A_hist[t][row_ind, col_ind] for t in range(len(est_A_hist))]
            curr_true_edge_history = [true_A_hist[t][row_ind, col_ind] for t in range(len(true_A_hist))]

            smoothed_est_edge_hist = [np.mean(curr_est_edge_history[t:t+smoothing_window_size]) for t in range(len(curr_est_edge_history)-smoothing_window_size)]
            smoothed_true_edge_hist = [np.mean(curr_true_edge_history[t:t+smoothing_window_size]) for t in range(len(curr_true_edge_history)-smoothing_window_size)]

            for w, (s_est_edge_act, s_true_edge_act) in enumerate(zip(smoothed_est_edge_hist, smoothed_true_edge_hist)):
                smoothed_est_A_hist[w][row_ind, col_ind] = s_est_edge_act
                smoothed_true_A_hist[w][row_ind, col_ind] = s_true_edge_act

    # rank across smoothed edge activations at each window
    ranked_smoothed_est_A_hist = []
    ranked_smoothed_true_A_hist = []
    for w, (s_est_A, s_true_A) in enumerate(zip(smoothed_est_A_hist, smoothed_true_A_hist)):
        ranked_smoothed_est_A_hist.append(convert_variable_to_rank_variable(s_est_A, method='dense'))
        ranked_smoothed_true_A_hist.append(convert_variable_to_rank_variable(s_true_A, method='dense'))

    # compute stats
    for row_ind in range(num_channels):
        for col_ind in range(num_channels):
            
            curr_true_smooth_ranks_across_windows = [rstA[row_ind, col_ind] for rstA in ranked_smoothed_true_A_hist]
            curr_true_avg_smooth_rank = np.mean(curr_true_smooth_ranks_across_windows)
            
            if curr_true_avg_smooth_rank > 1. and row_ind != col_ind: # filter out edges that have no true activation value (rank==1), as the resulting stats would all be nan, inf, or 0.
                curr_est_smooth_ranks_across_windows = [rseA[row_ind, col_ind] for rseA in ranked_smoothed_est_A_hist]
                curr_est_avg_smooth_rank = np.mean(curr_est_smooth_ranks_across_windows)
                avg_smooth_rank_diff = curr_est_avg_smooth_rank-curr_true_avg_smooth_rank
                smooth_rank_SqErrs_across_windows = [(est-truth)**2. for (est, truth) in zip(curr_est_smooth_ranks_across_windows, curr_true_smooth_ranks_across_windows)]
                smooth_rank_MSE_across_windows = np.mean(smooth_rank_SqErrs_across_windows)
                
                curr_true_smooth_activs_across_windows = [stA[row_ind, col_ind] for stA in smoothed_true_A_hist]
                curr_true_avg_smooth_activ = np.mean(curr_true_smooth_activs_across_windows)
                curr_est_smooth_activs_across_windows = [seA[row_ind, col_ind] for seA in smoothed_est_A_hist]
                curr_est_avg_smooth_activ = np.mean(curr_est_smooth_activs_across_windows)
                smooth_activ_SqErrs_across_windows = [(est-truth)**2. for (est, truth) in zip(curr_est_smooth_activs_across_windows, curr_true_smooth_activs_across_windows)]
                smooth_activ_MSE_across_windows = np.mean(smooth_activ_SqErrs_across_windows)
                
                # compute ranked history stats according to stat_paradigm
                curr_paradigm_ranked_smooth_hist_stat = None
                if stat_paradigm == "PearsonCorrelation":
                    _, _, pearson_r, pearson_p, _ = linregress(curr_est_smooth_ranks_across_windows, curr_true_smooth_ranks_across_windows)
                    curr_paradigm_ranked_smooth_hist_stat = {"pearson_r": pearson_r, "pearson_p": pearson_p, }
                else:
                    raise NotImplemented()
                    
                # compute activation history stats according to stat_paradigm
                curr_paradigm_smooth_activ_hist_stat = None
                if stat_paradigm == "PearsonCorrelation":
                    _, _, pearson_r, pearson_p, _ = linregress(curr_est_smooth_activs_across_windows, curr_true_smooth_activs_across_windows)
                    curr_paradigm_smooth_activ_hist_stat = {"pearson_r": pearson_r, "pearson_p": pearson_p, }
                else:
                    raise NotImplemented()
                
                key_stats[str(row_ind)+'<-'+str(col_ind)] = {
                    'smooth_rank_MSE_across_windows': smooth_rank_MSE_across_windows, 
                    'smooth_activ_MSE_across_windows': smooth_activ_MSE_across_windows, 
                    stat_paradigm+'_curr_paradigm_ranked_smooth_hist_stat': curr_paradigm_ranked_smooth_hist_stat, 
                    stat_paradigm+'_curr_paradigm_smooth_activ_hist_stat': curr_paradigm_smooth_activ_hist_stat, 
                }
                if curr_true_avg_smooth_rank not in key_stats.keys():
                    key_stats[curr_true_avg_smooth_rank] = {
                        'smooth_rank_MSE_across_windows': [smooth_rank_MSE_across_windows], 
                        'smooth_activ_MSE_across_windows': [smooth_activ_MSE_across_windows], 
                        stat_paradigm+'_curr_paradigm_ranked_smooth_hist_stat': [curr_paradigm_ranked_smooth_hist_stat], 
                        stat_paradigm+'_curr_paradigm_smooth_activ_hist_stat': [curr_paradigm_smooth_activ_hist_stat], 
                    }
                else:
                    key_stats[curr_true_avg_smooth_rank]['smooth_rank_MSE_across_windows'].append(smooth_rank_MSE_across_windows)
                    key_stats[curr_true_avg_smooth_rank]['smooth_activ_MSE_across_windows'].append(smooth_activ_MSE_across_windows)
                    key_stats[curr_true_avg_smooth_rank][stat_paradigm+'_curr_paradigm_ranked_smooth_hist_stat'].append(curr_paradigm_ranked_smooth_hist_stat)
                    key_stats[curr_true_avg_smooth_rank][stat_paradigm+'_curr_paradigm_smooth_activ_hist_stat'].append(curr_paradigm_smooth_activ_hist_stat)
    
    return key_stats


def compute_edgeRankPerformance_stats_betw_two_gc_graphs(stat_paradigm, est_A_hist, true_A_hist, smoothing_window_size=1, plot_save_path=None):
    """
    Compute Various Edge-Level Statistics Between the Two Histories Provided
    """
    for ind, eA in enumerate(est_A_hist):
        try:
            est_A_hist[ind] = eA.detach().numpy()
        except:
            pass
    for ind, tA in enumerate(true_A_hist):
        try:
            true_A_hist[ind] = tA.detach().numpy()
        except:
            pass
    assert est_A_hist[0].shape == true_A_hist[0].shape
    assert len(est_A_hist[0].shape) == 2
    assert est_A_hist[0].shape[0] == est_A_hist[0].shape[1]
    assert len(est_A_hist) == len(true_A_hist)
    num_channels = est_A_hist[0].shape[0]
    orig_gc_shape = est_A_hist[0].shape
    
    # initialize stat_paradigm stat objects to be computed/recorded (both smoothed activation and ranked_smoothAct stats)
    key_stats = dict()

    # smooth edge activations across time
    smoothed_est_A_hist = [np.zeros(orig_gc_shape) for _ in range(len(est_A_hist)-smoothing_window_size)]
    smoothed_true_A_hist = [np.zeros(orig_gc_shape) for _ in range(len(true_A_hist)-smoothing_window_size)]
    for row_ind in range(num_channels):
        for col_ind in range(num_channels):
            curr_est_edge_history = [est_A_hist[t][row_ind, col_ind] for t in range(len(est_A_hist))]
            curr_true_edge_history = [true_A_hist[t][row_ind, col_ind] for t in range(len(true_A_hist))]

            smoothed_est_edge_hist = [np.mean(curr_est_edge_history[t:t+smoothing_window_size]) for t in range(len(curr_est_edge_history)-smoothing_window_size)]
            smoothed_true_edge_hist = [np.mean(curr_true_edge_history[t:t+smoothing_window_size]) for t in range(len(curr_true_edge_history)-smoothing_window_size)]

            for w, (s_est_edge_act, s_true_edge_act) in enumerate(zip(smoothed_est_edge_hist, smoothed_true_edge_hist)):
                smoothed_est_A_hist[w][row_ind, col_ind] = s_est_edge_act
                smoothed_true_A_hist[w][row_ind, col_ind] = s_true_edge_act

    # rank across smoothed edge activations at each window
    ranked_smoothed_est_A_hist = []
    ranked_smoothed_true_A_hist = []
    for w, (s_est_A, s_true_A) in enumerate(zip(smoothed_est_A_hist, smoothed_true_A_hist)):
        ranked_smoothed_est_A_hist.append(convert_variable_to_rank_variable(s_est_A, method='dense'))
        ranked_smoothed_true_A_hist.append(convert_variable_to_rank_variable(s_true_A, method='dense'))

    # compute stats
    for row_ind in range(num_channels):
        for col_ind in range(num_channels):
            
            curr_true_smooth_ranks_across_windows = [rstA[row_ind, col_ind] for rstA in ranked_smoothed_true_A_hist]
            curr_true_avg_smooth_rank = np.mean(curr_true_smooth_ranks_across_windows)
            
            if curr_true_avg_smooth_rank > 1. and row_ind != col_ind: # filter out edges that have no true activation value (rank==1), as the resulting stats would all be nan, inf, or 0.
                curr_est_smooth_ranks_across_windows = [rseA[row_ind, col_ind] for rseA in ranked_smoothed_est_A_hist]
                curr_est_avg_smooth_rank = np.mean(curr_est_smooth_ranks_across_windows)
                avg_smooth_rank_diff = curr_est_avg_smooth_rank-curr_true_avg_smooth_rank
                smooth_rank_diffs_across_windows = [est-truth for (est, truth) in zip(curr_est_smooth_ranks_across_windows, curr_true_smooth_ranks_across_windows)]
                avg_of_smooth_rank_diffs_across_windows = np.mean(smooth_rank_diffs_across_windows)
                max_of_smooth_rank_diffs_across_windows = np.max(smooth_rank_diffs_across_windows)
                min_of_smooth_rank_diffs_across_windows = np.min(smooth_rank_diffs_across_windows)
                med_of_smooth_rank_diffs_across_windows = np.median(smooth_rank_diffs_across_windows)
                
                curr_true_smooth_activs_across_windows = [stA[row_ind, col_ind] for stA in smoothed_true_A_hist]
                curr_true_avg_smooth_activ = np.mean(curr_true_smooth_activs_across_windows)
                curr_est_smooth_activs_across_windows = [seA[row_ind, col_ind] for seA in smoothed_est_A_hist]
                curr_est_avg_smooth_activ = np.mean(curr_est_smooth_activs_across_windows)
                avg_smooth_activ_diff = curr_est_avg_smooth_activ-curr_true_avg_smooth_activ
                smooth_activ_diffs_across_windows = [est-truth for (est, truth) in zip(curr_est_smooth_activs_across_windows, curr_true_smooth_activs_across_windows)]
                avg_of_smooth_activ_diffs_across_windows = np.mean(smooth_activ_diffs_across_windows)
                max_of_smooth_activ_diffs_across_windows = np.max(smooth_activ_diffs_across_windows)
                min_of_smooth_activ_diffs_across_windows = np.min(smooth_activ_diffs_across_windows)
                med_of_smooth_activ_diffs_across_windows = np.median(smooth_activ_diffs_across_windows)
                
                # compute ranked history stats according to stat_paradigm
                curr_paradigm_ranked_smooth_hist_stat = None
                if stat_paradigm == "PearsonCorrelation":
                    _, _, pearson_r, pearson_p, _ = linregress(curr_est_smooth_ranks_across_windows, curr_true_smooth_ranks_across_windows)
                    curr_paradigm_ranked_smooth_hist_stat = {"pearson_r": pearson_r, "pearson_p": pearson_p, }
                elif stat_paradigm == "SpearmanCorrelation":
                    spearman_r, spearman_p = spearmanr(curr_est_smooth_ranks_across_windows, b=curr_true_smooth_ranks_across_windows)
                    curr_paradigm_ranked_smooth_hist_stat = {"spearman_r": spearman_r, "spearman_p": spearman_p, }
                elif stat_paradigm == "ROC_AUC":
                    try:
                        curr_paradigm_ranked_smooth_hist_stat = roc_auc_score(curr_true_smooth_ranks_across_windows, curr_est_smooth_ranks_across_windows)
                    except: 
                        curr_paradigm_ranked_smooth_hist_stat = None
                else:
                    raise NotImplemented()
                    
                # compute activation history stats according to stat_paradigm
                curr_paradigm_smooth_activ_hist_stat = None
                if stat_paradigm == "PearsonCorrelation":
                    _, _, pearson_r, pearson_p, _ = linregress(curr_est_smooth_activs_across_windows, curr_true_smooth_activs_across_windows)
                    curr_paradigm_smooth_activ_hist_stat = {"pearson_r": pearson_r, "pearson_p": pearson_p, }
                elif stat_paradigm == "SpearmanCorrelation":
                    spearman_r, spearman_p = spearmanr(curr_est_smooth_activs_across_windows, b=curr_true_smooth_activs_across_windows)
                    curr_paradigm_smooth_activ_hist_stat = {"spearman_r": spearman_r, "spearman_p": spearman_p, }
                elif stat_paradigm == "ROC_AUC":
                    curr_paradigm_smooth_activ_hist_stat = None
                else:
                    raise NotImplemented()
                
                key_stats[str(row_ind)+'<-'+str(col_ind)] = {
                    'avg_smooth_rank_diff': avg_smooth_rank_diff, 
                    'avg_of_smooth_rank_diffs_across_windows': avg_of_smooth_rank_diffs_across_windows, 
                    'avg_smooth_activ_diff': avg_smooth_activ_diff, 
                    'avg_of_smooth_activ_diffs_across_windows': avg_of_smooth_activ_diffs_across_windows, 
                    stat_paradigm+'_curr_paradigm_ranked_smooth_hist_stat': curr_paradigm_ranked_smooth_hist_stat, 
                    stat_paradigm+'_curr_paradigm_smooth_activ_hist_stat': curr_paradigm_smooth_activ_hist_stat, 
                }
                if curr_true_avg_smooth_rank not in key_stats.keys():
                    key_stats[curr_true_avg_smooth_rank] = {
                        'avg_smooth_rank_diff': [avg_smooth_rank_diff], 
                        'avg_of_smooth_rank_diffs_across_windows': [avg_of_smooth_rank_diffs_across_windows], 
                        'avg_smooth_activ_diff': [avg_smooth_activ_diff], 
                        'avg_of_smooth_activ_diffs_across_windows': [avg_of_smooth_activ_diffs_across_windows], 
                        stat_paradigm+'_curr_paradigm_ranked_smooth_hist_stat': [curr_paradigm_ranked_smooth_hist_stat], 
                        stat_paradigm+'_curr_paradigm_smooth_activ_hist_stat': [curr_paradigm_smooth_activ_hist_stat], 
                    }
                else:
                    key_stats[curr_true_avg_smooth_rank]['avg_smooth_rank_diff'].append(avg_smooth_rank_diff)
                    key_stats[curr_true_avg_smooth_rank]['avg_of_smooth_rank_diffs_across_windows'].append(avg_of_smooth_rank_diffs_across_windows)
                    key_stats[curr_true_avg_smooth_rank]['avg_smooth_activ_diff'].append(avg_smooth_activ_diff)
                    key_stats[curr_true_avg_smooth_rank]['avg_of_smooth_activ_diffs_across_windows'].append(avg_of_smooth_activ_diffs_across_windows)
                    key_stats[curr_true_avg_smooth_rank][stat_paradigm+'_curr_paradigm_ranked_smooth_hist_stat'].append(curr_paradigm_ranked_smooth_hist_stat)
                    key_stats[curr_true_avg_smooth_rank][stat_paradigm+'_curr_paradigm_smooth_activ_hist_stat'].append(curr_paradigm_smooth_activ_hist_stat)
    
    return key_stats


def compute_smoothed_edge_crossEdgeRank_covariance_stats_betw_two_gc_graphs(est_A_hist, true_A_hist, smoothing_window_sizes=[1], plot_save_path=None):
    """
    Computes Avg. Rank-Covariance Between SMOOTHED Estimated and True Ranked-Edge Histories
    """
    for ind, eA in enumerate(est_A_hist):
        try:
            est_A_hist[ind] = eA.detach().numpy()
        except:
            pass
    for ind, tA in enumerate(true_A_hist):
        try:
            true_A_hist[ind] = tA.detach().numpy()
        except:
            pass
    assert est_A_hist[0].shape == true_A_hist[0].shape
    assert len(est_A_hist[0].shape) == 2
    assert len(est_A_hist) == len(true_A_hist)
    
    key_stats = dict()
    for sw_size in smoothing_window_sizes:
        
        # smooth edge activations across time
        smoothed_est_A_hist = [np.zeros(est_A_hist[0].shape) for _ in range(len(est_A_hist)-sw_size)]
        smoothed_true_A_hist = [np.zeros(true_A_hist[0].shape) for _ in range(len(true_A_hist)-sw_size)]
        for row_ind in range(est_A_hist[0].shape[0]):
            for col_ind in range(est_A_hist[0].shape[1]):
                curr_est_edge_history = [est_A_hist[t][row_ind, col_ind] for t in range(len(est_A_hist))]
                curr_true_edge_history = [true_A_hist[t][row_ind, col_ind] for t in range(len(true_A_hist))]
                
                smoothed_est_edge_hist = [np.mean(curr_est_edge_history[t:t+sw_size]) for t in range(len(curr_est_edge_history)-sw_size)]
                smoothed_true_edge_hist = [np.mean(curr_true_edge_history[t:t+sw_size]) for t in range(len(curr_true_edge_history)-sw_size)]
                
                for w, (s_est_edge_act, s_true_edge_act) in enumerate(zip(smoothed_est_edge_hist, smoothed_true_edge_hist)):
                    smoothed_est_A_hist[w][row_ind, col_ind] = s_est_edge_act
                    smoothed_true_A_hist[w][row_ind, col_ind] = s_true_edge_act
                
        # rank across smoothed edge activations at each window
        ranked_smoothed_est_A_hist = []
        ranked_smoothed_true_A_hist = []
        for w, (s_est_A, s_true_A) in enumerate(zip(smoothed_est_A_hist, smoothed_true_A_hist)):
            ranked_smoothed_est_A_hist.append(convert_variable_to_rank_variable(s_est_A))
            ranked_smoothed_true_A_hist.append(convert_variable_to_rank_variable(s_true_A))
        
        # compute covariance across ranked windows
        curr_rank_covs = []
        for row_ind in range(ranked_smoothed_est_A_hist[0].shape[0]):
            for col_ind in range(ranked_smoothed_est_A_hist[0].shape[1]):
                curr_ranked_smooth_est_edge_hist = [ranked_smoothed_est_A_hist[w][row_ind, col_ind] for w in range(len(ranked_smoothed_est_A_hist))]
                curr_ranked_smooth_true_edge_hist = [ranked_smoothed_true_A_hist[w][row_ind, col_ind] for w in range(len(ranked_smoothed_true_A_hist))]
                rank_cov = compute_spearman_numerator_cov_of_ranked_variables(
                    curr_ranked_smooth_est_edge_hist, 
                    curr_ranked_smooth_true_edge_hist
                )
                curr_rank_covs.append(rank_cov)
                
                if plot_save_path is not None:
                    plot_estimated_vs_true_curve(
                        plot_save_path+os.sep+"smoothedWindowSize"+str(sw_size)+"_edge_i"+str(row_ind)+"_j"+str(col_ind)+"_strength_estimate_eval_visual.png", 
                        curr_ranked_smooth_est_edge_hist, curr_ranked_smooth_true_edge_hist, "Ranked Dynamic Edge Strength (Smoothed Across "+str(sw_size)+" Steps)", 
                        "Window in Time", "Ranked Edge Strength"
                    )
        key_stats["smoothWindow"+str(sw_size)+"_avg_edge_rank_cov"] = np.mean(curr_rank_covs)
    return key_stats


def compute_smoothed_edge_rank_covariance_stats_betw_two_gc_graphs(est_A_hist, true_A_hist, smoothing_window_sizes=[1], plot_save_path=None):
    """
    Computes Avg. Rank-Covariance Between SMOOTHED Estimated and True Edges Across Their Histories
    """
    for ind, eA in enumerate(est_A_hist):
        try:
            est_A_hist[ind] = eA.detach().numpy()
        except:
            pass
    for ind, tA in enumerate(true_A_hist):
        try:
            true_A_hist[ind] = tA.detach().numpy()
        except:
            pass
    assert est_A_hist[0].shape == true_A_hist[0].shape
    assert len(est_A_hist[0].shape) == 2
    assert len(est_A_hist) == len(true_A_hist)
    
    key_stats = dict()
    for sw_size in smoothing_window_sizes:
        curr_rank_covs = []
        
        for row_ind in range(est_A_hist[0].shape[0]):
            for col_ind in range(est_A_hist[0].shape[1]):
                curr_est_edge_history = [est_A_hist[t][row_ind, col_ind] for t in range(len(est_A_hist))]
                curr_true_edge_history = [true_A_hist[t][row_ind, col_ind] for t in range(len(true_A_hist))]
                
                smoothed_est_edge_hist = [np.mean(curr_est_edge_history[t:t+sw_size]) for t in range(len(curr_est_edge_history)-sw_size)]
                smoothed_true_edge_hist = [np.mean(curr_true_edge_history[t:t+sw_size]) for t in range(len(curr_true_edge_history)-sw_size)]
                
                rank_cov = compute_spearman_numerator_cov_of_ranked_variables(smoothed_est_edge_hist, smoothed_true_edge_hist)
                curr_rank_covs.append(rank_cov)
                
                if plot_save_path is not None:
                    plot_estimated_vs_true_curve(
                        plot_save_path+os.sep+"smoothedWindowSize"+str(sw_size)+"_edge_i"+str(row_ind)+"_j"+str(col_ind)+"_strength_estimate_eval_visual.png", 
                        smoothed_est_edge_hist, smoothed_true_edge_hist, "Dynamic Edge Strength (Smoothed Across "+str(sw_size)+" Steps)", 
                        "Window in Time", "Edge Strength"
                    )
        key_stats["smoothWindow"+str(sw_size)+"_avg_edge_rank_cov"] = np.mean(curr_rank_covs)
    return key_stats


def compute_key_edge_covariance_stats_betw_two_gc_graphs(est_A_hist, true_A_hist):
    """
    Computes Avg. Covariance and Rank-Covariance Between Estimated and True Edge Histories
    """
    for ind, eA in enumerate(est_A_hist):
        try:
            est_A_hist[ind] = eA.detach().numpy()
        except:
            pass
    for ind, tA in enumerate(true_A_hist):
        try:
            true_A_hist[ind] = tA.detach().numpy()
        except:
            pass
    assert est_A_hist[0].shape == true_A_hist[0].shape
    assert len(est_A_hist[0].shape) == 2
    assert len(est_A_hist) == len(true_A_hist)
    
    covs = []
    rank_covs = []
    for row_ind in range(est_A_hist[0].shape[0]):
        for col_ind in range(est_A_hist[0].shape[1]):
            curr_est_edge_history = [est_A_hist[t][row_ind, col_ind] for t in range(len(est_A_hist))]
            curr_true_edge_history = [true_A_hist[t][row_ind, col_ind] for t in range(len(true_A_hist))]
            cov = compute_covariance_betw_two_variables(curr_est_edge_history, curr_true_edge_history)
            rank_cov = compute_spearman_numerator_cov_of_ranked_variables(curr_est_edge_history, curr_true_edge_history)
            covs.append(cov)
            rank_covs.append(rank_cov)
            
    key_stats = {"avg_edge_cov": np.mean(covs), "avg_edge_rank_cov": np.mean(rank_covs), }
    return key_stats


def compute_key_covariance_stats_betw_two_gc_score_histories(est_h, true_h):
    """
    Computes Covariance and Rank-Covariance Between est_h And true_h
    """
    try:
        est_h = est_h.detach().numpy()
    except:
        pass
    try:
        true_h = true_h.detach().numpy()
    except:
        pass
    cov = compute_covariance_betw_two_variables(est_h, true_h)
    rank_cov = compute_spearman_numerator_cov_of_ranked_variables(est_h, true_h)
    key_stats = {"cov": cov, "rank_cov": rank_cov, }
    return key_stats


def compute_key_edge_correlation_stats_betw_two_gc_graphs(est_A_hist, true_A_hist):
    """
    Computes Avg. Pearson and Spearman Correlation Statistics Between Estimated and True Edge Histories
    """
    for ind, eA in enumerate(est_A_hist):
        try:
            est_A_hist[ind] = eA.detach().numpy()
        except:
            pass
    for ind, tA in enumerate(true_A_hist):
        try:
            true_A_hist[ind] = tA.detach().numpy()
        except:
            pass
    assert est_A_hist[0].shape == true_A_hist[0].shape
    assert len(est_A_hist[0].shape) == 2
    assert len(est_A_hist) == len(true_A_hist)
    
    pearson_rs = []
    pearson_ps = []
    spearman_rs = []
    spearman_ps = []
    for row_ind in range(est_A_hist[0].shape[0]):
        for col_ind in range(est_A_hist[0].shape[1]):
            curr_est_edge_history = [est_A_hist[t][row_ind, col_ind] for t in range(len(est_A_hist))]
            curr_true_edge_history = [true_A_hist[t][row_ind, col_ind] for t in range(len(true_A_hist))]
            _, _, pearson_r, pearson_p, _ = linregress(curr_est_edge_history, curr_true_edge_history) # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
            spearman_r, spearman_p = spearmanr(curr_est_edge_history, b=curr_true_edge_history) # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
            pearson_rs.append(pearson_r)
            pearson_ps.append(pearson_p)
            spearman_rs.append(spearman_r)
            spearman_ps.append(spearman_p)
    key_stats = {
        "avg_edge_pearson_r": np.mean(pearson_rs), 
        "avg_edge_pearson_p": np.mean(pearson_ps), 
        "avg_edge_spearman_r": np.mean(spearman_rs), 
        "avg_edge_spearman_p": np.mean(spearman_ps), 
    }
    return key_stats


def compute_key_spearman_correlation_stats_betw_two_gc_score_histories(est_h, true_h):
    """
    Computes Spearman Correlation Statistics Between est_h And true_h
    """
    try:
        est_h = est_h.detach().numpy()
    except:
        pass
    try:
        true_h = true_h.detach().numpy()
    except:
        pass
    sr, sp = spearmanr(est_h, b=true_h) # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
    key_stats = {"sr": sr, "sp": sp, }
    return key_stats


def compute_key_correlation_stats_betw_two_gc_score_histories(est_h, true_h):
    """
    Computes Pearson Correlation Statistics Between est_h And true_h
    """
    try:
        est_h = est_h.detach().numpy()
    except:
        pass
    try:
        true_h = true_h.detach().numpy()
    except:
        pass
    _, _, r, p, _ = linregress(est_h, true_h) # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
    key_stats = {"r": r, "p": p, }
    return key_stats


def compute_key_stats_betw_two_gc_score_vecs(est_v, true_v):
    try:
        est_v = est_v.detach().numpy()
    except:
        pass
    try:
        true_v = true_v.detach().numpy()
    except:
        pass
    key_stats = {"cosine_similarity": compute_cosine_similarity(est_v, true_v), "mse": compute_mse(est_v, true_v), }
    return key_stats


def compute_OptimalF1_stats_betw_two_gc_graphs(est_A, true_A):
    roc_auc_labels = None
    try:
        roc_auc_labels = [int(l) for l in true_A.flatten()]
    except: 
        roc_auc_labels = None
        
    key_stats = dict()
    if not np.isfinite(np.sum(est_A)):
        print("evaluate.eval_utils.compute_OptimalF1_stats_betw_two_gc_graphs: WARNING - NON-FINITE VALUE ENCOUNTERED IN est_A == ", est_A)
    elif np.min(est_A) == np.max(est_A): 
        print("evaluate.eval_utils.compute_OptimalF1_stats_betw_two_gc_graphs: WARNING - HOMOGENOUS VALUES DETECTED IN  est_A == ", est_A)
    elif not np.isfinite(np.sum(true_A)):
        print("evaluate.eval_utils.compute_OptimalF1_stats_betw_two_gc_graphs: WARNING - NON-FINITE VALUE ENCOUNTERED IN roc_auc_labels == ", roc_auc_labels)
    elif np.min(roc_auc_labels) == np.max(roc_auc_labels): 
        print("evaluate.eval_utils.compute_OptimalF1_stats_betw_two_gc_graphs: WARNING - HOMOGENOUS VALUES DETECTED IN  roc_auc_labels == ", roc_auc_labels)
    else: 
        decision_threshold = None
        optimal_f1 = None
        decision_threshold, optimal_f1 = compute_optimal_f1(roc_auc_labels, est_A.flatten())
        key_stats["f1"] = optimal_f1
        key_stats["decision_threshold"] = decision_threshold
    return key_stats


def compute_f1_stats_betw_two_gc_graphs(est_A, true_A, pred_cutoffs=[0.3,0.4,0.5,0.6,0.7,0.8,0.9]):
    roc_auc_labels = None
    try:
        roc_auc_labels = [int(l) for l in true_A.flatten()]
    except: 
        roc_auc_labels = None
        
    key_stats = dict()
    if not np.isfinite(np.sum(est_A)):
        print("evaluate.eval_utils.compute_f1_stats_betw_two_gc_graphs: WARNING - NON-FINITE VALUE ENCOUNTERED IN est_A == ", est_A)
    elif np.min(est_A) == np.max(est_A): 
        print("evaluate.eval_utils.compute_f1_stats_betw_two_gc_graphs: WARNING - HOMOGENOUS VALUES DETECTED IN  est_A == ", est_A)
    elif not np.isfinite(np.sum(true_A)):
        print("evaluate.eval_utils.compute_f1_stats_betw_two_gc_graphs: WARNING - NON-FINITE VALUE ENCOUNTERED IN roc_auc_labels == ", roc_auc_labels)
    elif np.min(roc_auc_labels) == np.max(roc_auc_labels): 
        print("evaluate.eval_utils.compute_f1_stats_betw_two_gc_graphs: WARNING - HOMOGENOUS VALUES DETECTED IN  roc_auc_labels == ", roc_auc_labels)
    else: 
        for pc in pred_cutoffs:
            try:
                key_stats["f1_pc"+str(pc)] = compute_f1(roc_auc_labels, est_A.flatten(), pc)
            except:
                key_stats["f1_pc"+str(pc)] = None
    return key_stats


def compute_key_stats_betw_two_gc_graphs(est_A, true_A, dcon0_eps=0.1, max_mse_path_length=None, make_graphs_undirected_for_dcon0=False, pred_cutoffs=[0.3,0.4,0.5,0.6,0.7,0.8,0.9]):
    roc_auc_labels = None
    try:
        roc_auc_labels = [int(l) for l in true_A.flatten()]
    except: 
        roc_auc_labels = None
    
    # obtain ROC-AUC score, while handling various edge cases
    key_stats = dict()
    if not np.isfinite(np.sum(est_A)):
        print("evaluate.eval_utils.compute_key_stats_betw_two_gc_graphs: WARNING - NON-FINITE VALUE ENCOUNTERED IN est_A == ", est_A)
    elif np.min(est_A) == np.max(est_A): 
        print("evaluate.eval_utils.compute_key_stats_betw_two_gc_graphs: WARNING - HOMOGENOUS VALUES DETECTED IN  est_A == ", est_A)
    elif not np.isfinite(np.sum(true_A)):
        print("evaluate.eval_utils.compute_key_stats_betw_two_gc_graphs: WARNING - NON-FINITE VALUE ENCOUNTERED IN roc_auc_labels == ", roc_auc_labels)
    elif np.min(roc_auc_labels) == np.max(roc_auc_labels): 
        print("evaluate.eval_utils.compute_key_stats_betw_two_gc_graphs: WARNING - HOMOGENOUS VALUES DETECTED IN  roc_auc_labels == ", roc_auc_labels)
    else: 
        try:
            key_stats["roc_auc"] = roc_auc_score(roc_auc_labels, est_A.flatten())
        except:
            key_stats["roc_auc"] = None
        for pc in pred_cutoffs:
            try:
                key_stats["sensitivity_pc"+str(pc)] = compute_sensitivity(roc_auc_labels, est_A.flatten(), pred_cutoff=pc)
            except:
                key_stats["sensitivity_pc"+str(pc)] = None
            try:
                key_stats["specificity_pc"+str(pc)] = compute_specificity(roc_auc_labels, est_A.flatten(), pred_cutoff=pc)
            except:
                key_stats["specificity_pc"+str(pc)] = None
            
            try:
                key_stats["PLR_pc"+str(pc)] = compute_positive_likelihood_ratio(roc_auc_labels, est_A.flatten(), pred_cutoff=pc)
            except:
                key_stats["PLR_pc"+str(pc)] = None
            try:
                key_stats["NLR_pc"+str(pc)] = compute_negative_likelihood_ratio(roc_auc_labels, est_A.flatten(), pred_cutoff=pc)
            except:
                key_stats["NLR_pc"+str(pc)] = None
    return key_stats


def get_path_to_trained_model(alg_name, dset_name, root_paths_to_trained_models, labeling_scheme_to_swap_towards=None):
    curr_model_root_save_dir = None
    if alg_name in ["CMLP", "CLSTM", "DGCNN"]: # handle edge case from earlier naming convention(s)
        curr_model_root_save_dir = [x for x in root_paths_to_trained_models if alg_name in x and "REDCLIFF" not in x]
    elif alg_name in ["REDCLIFF_S_CMLP_WithSmoothing"]:
        curr_model_root_save_dir = [x for x in root_paths_to_trained_models if "REDCLIFF_S_CMLP" in x]
    else:
        curr_model_root_save_dir = [x for x in root_paths_to_trained_models if alg_name in x]
    assert len(curr_model_root_save_dir) == 1
    curr_model_root_save_dir = curr_model_root_save_dir[0]
    
    model_training_dir = []
    for train_path in os.listdir(curr_model_root_save_dir):
        swapped_train_path = ""+train_path
        if labeling_scheme_to_swap_towards is not None:
            if labeling_scheme_to_swap_towards == "OneHot":
                split_path = swapped_train_path.split("Oracle")
                if len(split_path) > 1:
                    swapped_train_path = "OneHot".join(split_path)
                else:
                    train_path = ""
            elif labeling_scheme_to_swap_towards == "Oracle":
                split_path = swapped_train_path.split("OneHot")
                if len(split_path) > 1:
                    swapped_train_path = "Oracle".join(split_path)
                else:
                    train_path = ""
            else:
                raise ValueError(labeling_scheme_to_swap_towards)
        if os.path.isdir(curr_model_root_save_dir+os.sep+train_path) and dset_name in swapped_train_path:
            model_training_dir.append(curr_model_root_save_dir+os.sep+train_path)
            
    if len(model_training_dir) == 1:
        model_training_dir = model_training_dir[0]
        
        trained_model_path = None
        if "DCSFA" in alg_name or "dCSFA" in alg_name:
            trained_model_path = [model_training_dir+os.sep+x for x in os.listdir(model_training_dir) if "best-model" in x]
        else:
            trained_model_path = [model_training_dir+os.sep+x for x in os.listdir(model_training_dir) if "final_best_model" in x]
            
        if len(trained_model_path) == 1:
            trained_model_path = trained_model_path[0]
            return trained_model_path
        
    return None


def load_model_for_eval(model_type, model_path, dynamic_eval=False, d4IC=False, dcsfa_dim_in=None):
    model = None
    if "cMLP" in model_type or "DGCNN" in model_type or "cLSTM" in model_type or "REDCLIFF" in model_type or "NAVAR" in model_type:
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
    elif "DCSFA" in model_type:
        if dynamic_eval:
            model_param_str = model_path.split(os.sep)[-2]
            split_param_str = model_param_str.split("_")
            n_components = int(split_param_str[1][len("numF"):])
            n_sup_networks = int(split_param_str[2][len("numSF"):])
            num_nodes = int(split_param_str[3][len("numN"):])
            if dcsfa_dim_in is None:
                if num_nodes == 6:
                    dcsfa_dim_in = 468 # this is the default dimension which assumes num_nodes == 6
                elif num_nodes == 3:
                    dcsfa_dim_in = 117
                elif num_nodes == 12:
                    dcsfa_dim_in = 1872
                else:
                    raise NotImplementedError("eval_utils.load_model_for_eval: NO IMPLEMENTATION FOR dim_in WHERE num_nodes == "+str(num_nodes))
                    
            print("eval_utils.load_model_for_eval: WARNING - ASSUMING DCSFA MODEL BELONGS TO models.dcsfa_nmf_vanillaDirSpec FullDCSFAModel CLASS")
            print("eval_utils.load_model_for_eval: WARNING (CONTINUED) - ASSUMING THE FOLLOWING ARGUEMENTS ARE ACCURATE FOR DCSFA MODEL INIT: ")
            print("eval_utils.load_model_for_eval: WARNING (CONTINUED) - \t *  num_nodes == ", num_nodes)
            print("eval_utils.load_model_for_eval: WARNING (CONTINUED) - \t *  num_high_level_node_features == ", 13)
            print("eval_utils.load_model_for_eval: WARNING (CONTINUED) - \t *  n_components == ", n_components)
            print("eval_utils.load_model_for_eval: WARNING (CONTINUED) - \t *  n_sup_networks == ", n_sup_networks)
            print("eval_utils.load_model_for_eval: WARNING (CONTINUED) - \t *  h == ", 256)
            print("eval_utils.load_model_for_eval: WARNING (CONTINUED) - \t *  device == ", "auto")
            print("eval_utils.load_model_for_eval: WARNING (CONTINUED) - \t *  n_intercepts == ", 1)
            print("eval_utils.load_model_for_eval: WARNING (CONTINUED) - \t *  use_deep_encoder == ", True)
            print("eval_utils.load_model_for_eval: WARNING (CONTINUED) - \t *  dim_in == ", dcsfa_dim_in)#468)
            
            model = FullDCSFAModel_GCv2(
                num_nodes=num_nodes, 
                num_high_level_node_features=13, 
                n_components=n_components,
                n_sup_networks=n_sup_networks,
                h=256,
                device="auto",
                n_intercepts=1,
                use_deep_encoder=True,
            )
            model._initialize_NMF(dcsfa_dim_in)
            model._initialize(dcsfa_dim_in)
            model.load_state_dict(torch.load(model_path))
        elif d4IC:
            model_param_str = model_path.split(os.sep)[-2]
            split_param_str = model_param_str.split("_")
            n_components = 5
            n_sup_networks = 5
            num_nodes = 10
            print("eval_utils.load_model_for_eval: WARNING - ASSUMING DCSFA MODEL BELONGS TO models.dcsfa_nmf_vanillaDirSpec FullDCSFAModel CLASS")
            print("eval_utils.load_model_for_eval: WARNING (CONTINUED) - ASSUMING THE FOLLOWING ARGUEMENTS ARE ACCURATE FOR DCSFA MODEL INIT: ")
            print("eval_utils.load_model_for_eval: WARNING (CONTINUED) - \t *  num_nodes == ", num_nodes)
            print("eval_utils.load_model_for_eval: WARNING (CONTINUED) - \t *  num_high_level_node_features == ", 5)
            print("eval_utils.load_model_for_eval: WARNING (CONTINUED) - \t *  n_components == ", n_components)
            print("eval_utils.load_model_for_eval: WARNING (CONTINUED) - \t *  n_sup_networks == ", n_sup_networks)
            print("eval_utils.load_model_for_eval: WARNING (CONTINUED) - \t *  h == ", 256)
            print("eval_utils.load_model_for_eval: WARNING (CONTINUED) - \t *  device == ", "auto")
            print("eval_utils.load_model_for_eval: WARNING (CONTINUED) - \t *  n_intercepts == ", 1)
            print("eval_utils.load_model_for_eval: WARNING (CONTINUED) - \t *  use_deep_encoder == ", True)
            print("eval_utils.load_model_for_eval: WARNING (CONTINUED) - \t *  dim_in == ", 500)
            model = FullDCSFAModel_GCv2(
                num_nodes=num_nodes, 
                num_high_level_node_features=5, 
                n_components=n_components,
                n_sup_networks=n_sup_networks,
                h=256,
                device="auto",
                n_intercepts=1,
                use_deep_encoder=True,
            )
            model._initialize_NMF(500)
            model._initialize(500)
            model.load_state_dict(torch.load(model_path))
        else:
            model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
    elif "DYNOTEARS" in model_type:
        model = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        raise NotImplementedError()
    return model


def get_combined_gc_representations_across_factors(estimated_gcs, true_gcs):
    combo_true_gc = np.zeros(true_gcs[0].shape)
    for tgc in true_gcs:
        combo_true_gc = combo_true_gc + tgc
    combo_estimated_gc = np.zeros(estimated_gcs[0].shape)
    for egc in estimated_gcs:
        combo_estimated_gc = combo_estimated_gc + egc
    return combo_estimated_gc, combo_true_gc

def get_model_gc_score_estimates(model, model_type, num_ests_required, X=None):
    gc_scores = None
    if "REDCLIFF" in model_type:
        _, _, _, gc_scores = model(X)
        gc_scores = gc_scores[0].view(num_ests_required).detach().numpy()
    elif "cMLP" in model_type or "DGCNN" in model_type or "cLSTM" in model_type or "DYNOTEARS" in model_type or "NAVAR" in model_type:
        gc_scores = np.ones(num_ests_required)
    elif "DCSFA" in model_type:
        gc_scores, _ = self.predict_proba(X, return_scores=True)
        gc_scores = gc_scores.reshape(num_ests_required)
    else:
        raise NotImplementedError()
    return gc_scores


def get_model_gc_estimates(model, model_type, num_ests_required, X=None):
    gc_ests = None
    
    if "REDCLIFF" in model_type:
        gc_ests_by_sample = model.GC(model.primary_gc_est_mode, X=X, threshold=False, ignore_lag=False, combine_wavelet_representations=True, rank_wavelets=False)
        assert len(gc_ests_by_sample) == 1
        gc_ests = [x.detach().numpy() for x in gc_ests_by_sample[0]]
        if len(gc_ests) < num_ests_required:
            assert len(gc_ests) == 1
            gc_ests = [copy.deepcopy(gc_ests[0]) for _ in range(num_ests_required)]

    elif "cMLP" in model_type or "DGCNN" in model_type or "cLSTM" in model_type or "DYNOTEARS" in model_type or "NAVAR" in model_type:
        generic_est = None

        if "cMLP" in model_type:
            generic_est = [x.detach().numpy() for x in model.GC(threshold=False, ignore_lag=True, combine_wavelet_representations=True, rank_wavelets=False)]
        elif "DGCNN" in model_type:
            generic_est = [model.GC(threshold=False, combine_node_feature_edges=True).detach().numpy()]
        elif "cLSTM" in model_type:
            generic_est = [x.detach().numpy() for x in model.GC(threshold=False, combine_wavelet_representations=True, rank_wavelets=False)]
        elif "DYNOTEARS" in model_type or "NAVAR" in model_type:
            generic_est = [model.GC()]
        else:
            raise ValueError()
        
        assert len(generic_est) == 1
        gc_ests = [copy.deepcopy(generic_est[0]) for _ in range(num_ests_required)]

    elif "DCSFA" in model_type:
        gc_ests = model.GC(threshold=False, ignore_features=True)
    
    elif "NCFM" in model_type:
        if "CMLP" in model_type:
            gc_ests = [x.detach().numpy() for x in model.GC(threshold=False, ignore_lag=True, combine_wavelet_representations=True, rank_wavelets=False)]
        elif "CLSTM":
            gc_ests = [x.detach().numpy() for x in model.GC(threshold=False, combine_wavelet_representations=True, rank_wavelets=False)]
        else:
            raise ValueError("eval_utils.get_model_gc_estimates: UNRECOGNIZED model_type == "+str(model_type))

    else:
        raise NotImplementedError()

    return gc_ests


def evaluate_avg_factor_scoring_across_recordings(model, data_loader, num_timesteps_to_score_per_state, num_timesteps_in_input_history, 
                                                  save_root_path, titles="|     Home Cage        |     Open Field      |    Tail Suspended    |", 
                                                  colors=['indigo','orangered',"grey"], markers=['+','.','x'], labels=['HC', 'OF', 'TS'], num_comparissons_to_perform=100):
    individual_state_id_titles = [x.strip() for x in titles[1:-1].split("|")]
    num_factors_for_iterating = model.num_supervised_factors if model.num_supervised_factors > 0 else model.num_factors_nK
    for state_id in range(num_factors_for_iterating):
        # obtain sample recording that corresponds to current state_id
        curr_state_id_label_traces = []
        curr_state_id_factor_weightings = []
        curr_state_id_factor_classifications = []
        for batch_num, (X, Y) in enumerate(data_loader):
            Y = Y.numpy()
            if len(Y.shape) == 3 and Y.shape[2] > 1: # implementation for newer synthetic datasets
                assert Y.shape[0] == 1
                label_over_time = np.argmax(Y[0,:,:], axis=0)
                window_label, _ = scistats.mode(label_over_time)
                if window_label == state_id:
                    curr_weighting_scores = obtain_factor_score_weightings_across_recording(model, X, num_factors_for_iterating, num_timesteps_to_score_per_state, num_timesteps_in_input_history)
                    curr_classification_scores = None
                    if model.num_supervised_factors > 0:
                        curr_classification_scores = obtain_factor_score_classifications_across_recording(model, X, num_factors_for_iterating, num_timesteps_to_score_per_state, num_timesteps_in_input_history)
                    else:
                        curr_classification_scores = 0.*curr_weighting_scores
                    curr_label_trace = Y[0,:,:curr_classification_scores.shape[-1]]
                    
                    curr_state_id_label_traces.append(curr_label_trace)
                    curr_state_id_factor_weightings.append(curr_weighting_scores)
                    curr_state_id_factor_classifications.append(curr_classification_scores)
            else: # implementation for DREAM4 and TST100hz datasets
                while len(Y.shape) > 2:
                    assert Y.shape[-1] == 1
                    Y = np.squeeze(Y, axis=-1)
                assert len(Y.shape) == 2
                if np.argmax(Y[0,:]) == state_id:
                    # track model scoring functionality over extent of curr_state_id_recording
                    curr_weighting_scores = obtain_factor_score_weightings_across_recording(model, X, num_factors_for_iterating, num_timesteps_to_score_per_state, num_timesteps_in_input_history)
                    curr_classification_scores = None
                    if model.num_supervised_factors > 0:
                        curr_classification_scores = obtain_factor_score_classifications_across_recording(model, X, num_factors_for_iterating, num_timesteps_to_score_per_state, num_timesteps_in_input_history)
                    else:
                        curr_classification_scores = 0.*curr_weighting_scores
                    curr_label_trace = np.zeros(curr_classification_scores.shape)
                    curr_label_trace = Y[0,:].reshape(-1,1) + curr_label_trace # here we assume that the same label applies to the entire window, in keeping with TST and DREAM4 premises
                    
                    curr_state_id_label_traces.append(curr_label_trace)
                    curr_state_id_factor_weightings.append(curr_weighting_scores)
                    curr_state_id_factor_classifications.append(curr_classification_scores)
            if len(curr_state_id_factor_classifications) == num_comparissons_to_perform:
                break

        plot_avg_system_state_score_comparisson(
            save_root_path+"_avg_weighting_score_comparisson_across_recordings_state"+str(state_id)+".png", 
            curr_state_id_factor_weightings, 
            curr_state_id_label_traces, 
            individual_state_id_titles[state_id], 
            colors, 
            markers, 
            labels, 
        )
        plot_avg_system_state_score_comparisson(
            save_root_path+"_avg_classification_score_comparisson_across_recordings_state"+str(state_id)+".png", 
            curr_state_id_factor_classifications, 
            curr_state_id_label_traces, 
            individual_state_id_titles[state_id], 
            colors, 
            markers, 
            labels
        )
    pass


def evaluate_factor_scoring_across_recordings(model, data_loader, num_timesteps_to_score_per_state, num_timesteps_in_input_history, 
                                              save_root_path, title="|     Home Cage        |     Open Field      |    Tail Suspended    |", 
                                              colors=['indigo','orangered',"grey"], markers=['+','.','x'], labels=['HC', 'OF', 'TS'], num_comparissons_to_perform=10):
    for c in range(num_comparissons_to_perform):
        weighting_scores = None
        classification_scores = None
        if model.num_supervised_factors > 0:
            weighting_scores = np.zeros((model.num_supervised_factors, model.num_supervised_factors*num_timesteps_to_score_per_state))
            classification_scores = np.zeros((model.num_supervised_factors, model.num_supervised_factors*num_timesteps_to_score_per_state))
        else:
            weighting_scores = np.zeros((model.num_factors_nK, model.num_factors_nK*num_timesteps_to_score_per_state))
            classification_scores = np.zeros((model.num_factors_nK, model.num_factors_nK*num_timesteps_to_score_per_state))
        
        num_factors_for_iterating = model.num_supervised_factors if model.num_supervised_factors > 0 else model.num_factors_nK
        for state_id in range(num_factors_for_iterating):
            candidate_index = 0
            # obtain sample recording that corresponds to current state_id
            curr_state_id_recording = None
            for batch_num, (X, Y) in enumerate(data_loader):
                Y = Y.numpy()
                if len(Y.shape) == 3 and Y.shape[2] > 1: # implementation for newer synthetic datasets
                    assert Y.shape[0] == 1
                    label_over_time = np.argmax(Y[0,:,:], axis=0)
                    window_label, _ = scistats.mode(label_over_time)
                    if window_label == state_id:
                        if candidate_index == c:
                            curr_state_id_recording = X
                            break
                        else:
                            assert candidate_index < c # sanity check
                            candidate_index += 1
                else: # implementation for DREAM4 and TST100hz datasets
                    while len(Y.shape) > 2:
                        assert Y.shape[-1] == 1
                        Y = np.squeeze(Y, axis=-1)
                    assert len(Y.shape) == 2
                    if np.argmax(Y[0,:]) == state_id:
                        if candidate_index == c:
                            curr_state_id_recording = X
                            break
                        else:
                            assert candidate_index < c # sanity check
                            candidate_index += 1
            assert curr_state_id_recording is not None

            # track model scoring functionality over extent of curr_state_id_recording
            curr_classification_scores = None
            curr_weighting_scores = obtain_factor_score_weightings_across_recording(
                model, curr_state_id_recording, num_factors_for_iterating, num_timesteps_to_score_per_state, num_timesteps_in_input_history
            )
            if model.num_supervised_factors > 0:
                curr_classification_scores = obtain_factor_score_classifications_across_recording(
                    model, curr_state_id_recording, num_factors_for_iterating, num_timesteps_to_score_per_state, num_timesteps_in_input_history
                )
            else:
                curr_classification_scores = 0.*curr_weighting_scores

            # concatenate recordings across different system states
            weighting_scores[:,state_id*num_timesteps_to_score_per_state:(state_id+1)*num_timesteps_to_score_per_state] = weighting_scores[:,state_id*num_timesteps_to_score_per_state:(state_id+1)*num_timesteps_to_score_per_state] + curr_weighting_scores
            classification_scores[:,state_id*num_timesteps_to_score_per_state:(state_id+1)*num_timesteps_to_score_per_state] = classification_scores[:,state_id*num_timesteps_to_score_per_state:(state_id+1)*num_timesteps_to_score_per_state] + curr_classification_scores

        plot_system_state_score_comparisson(save_root_path+"_weighting_score_comparisson_across_recordings_v"+str(c)+".png", weighting_scores, title, colors, markers, labels)
        plot_system_state_score_comparisson(save_root_path+"_classification_score_comparisson_across_recordings_v"+str(c)+".png", classification_scores, title, colors, markers, labels)
    pass





def perform_system_level_estimation_evaluation_of_cv_model(model_type, save_dir, trained_models_root_path, possible_dataset_cv_splits, trained_model_file_name="final_best_model.bin", 
                                                           eps=0.1, in_degree_coeff=1., out_degree_coeff=1., max_path_length=None, X=None, args_dict=None, POSSIBLE_DATA_SETS=None, 
                                                           files_of_cached_data_args=None, sort_unsupervised_ests=False, cost_criteria="CosineSimilarity", unsupervised_start_index=0,
                                                           average_estimated_graphs_together=False, ablation_folder_tag=None, exclude_self_connections=False, evaluate_identity_baseline=False):
    print("\n\n eval_utils.perform_system_level_estimation_evaluation_of_cv_model: START ----------------------------------------------------------------------------------\n", flush=True)
    print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: exclude_self_connections == ", exclude_self_connections)
    print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: evaluate_identity_baseline == ", evaluate_identity_baseline, flush=True)
    
    for exp_id, cv_split_name in enumerate(possible_dataset_cv_splits):
        print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: NOW PROCESSING CV SPLIT WITH exp_id == ", exp_id, flush=True)
        cv_split_folders = sorted([trained_models_root_path+os.sep+x for x in os.listdir(trained_models_root_path) if cv_split_name in x and "." not in x and "gsTrue_param_training_results" not in x])
        print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: cv_split_folders == ", cv_split_folders, flush=True)
        cv_split_models = []
        if ablation_folder_tag is None:
            for folder in cv_split_folders:
                cv_split_models = cv_split_models + [folder+os.sep+x for x in os.listdir(folder) if trained_model_file_name in x]
        else:
            for folder in cv_split_folders:
                cv_split_models = cv_split_models + [folder+os.sep+x for x in os.listdir(folder) if trained_model_file_name in x and ablation_folder_tag in folder]
        NUM_FOLDS_IN_CV_SPLIT = len(cv_split_models)
        CURR_CANDIDATE_DATASETS = sorted([x for x in POSSIBLE_DATA_SETS if cv_split_name in x])
        CURR_CANDIDATE_DATASET_ARGS = sorted([x for x in files_of_cached_data_args if cv_split_name in x])
        print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: cv_split_models == ", cv_split_models, flush=True)
        print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: CURR_CANDIDATE_DATASETS == ", CURR_CANDIDATE_DATASETS, flush=True)
        print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: CURR_CANDIDATE_DATASET_ARGS == ", CURR_CANDIDATE_DATASET_ARGS, flush=True)
        print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: NUM_FOLDS_IN_CV_SPLIT == ", NUM_FOLDS_IN_CV_SPLIT, flush=True)
        
        if NUM_FOLDS_IN_CV_SPLIT != len(CURR_CANDIDATE_DATASETS):
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: SKIPPING CURRENT CV SPLIT EVAL DUE TO MISSING DATA-MODEL PAIRS", flush=True)
        else:

            # initialize records of cv-level stats
            curr_cv_model_cos_sim_scores_by_fold = {i:None for i in range(NUM_FOLDS_IN_CV_SPLIT)}
            curr_cv_model_cos_sim_score_avgs = []
            curr_cv_model_cos_sim_score_std_devs = []
            curr_cv_model_mse_by_fold = {i:None for i in range(NUM_FOLDS_IN_CV_SPLIT)}
            curr_cv_model_mse_avgs = []
            curr_cv_model_mse_std_devs = []
            curr_cv_model_dir_deltacon0_scores_by_fold = {i:None for i in range(NUM_FOLDS_IN_CV_SPLIT)}
            curr_cv_model_dir_deltacon0_score_avgs = []
            curr_cv_model_dir_deltacon0_score_std_devs = []
            curr_cv_model_undir_deltacon0_scores_by_fold = {i:None for i in range(NUM_FOLDS_IN_CV_SPLIT)}
            curr_cv_model_undir_deltacon0_score_avgs = []
            curr_cv_model_undir_deltacon0_score_std_devs = []
            curr_cv_model_deltacon0_wDD_scores_by_fold = {i:None for i in range(NUM_FOLDS_IN_CV_SPLIT)}
            curr_cv_model_deltacon0_wDD_score_avgs = []
            curr_cv_model_deltacon0_wDD_score_std_devs = []
            curr_cv_model_deltaffinity_scores_by_fold = {i:None for i in range(NUM_FOLDS_IN_CV_SPLIT)}
            curr_cv_model_deltaffinity_score_avgs = []
            curr_cv_model_deltaffinity_score_std_devs = []
            curr_cv_model_roc_auc_scores_by_fold = {i:None for i in range(NUM_FOLDS_IN_CV_SPLIT)}
            curr_cv_model_roc_auc_score_avgs = []
            curr_cv_model_roc_auc_score_std_devs = []

            curr_cv_model_T_cos_sim_scores_by_fold = {i:None for i in range(NUM_FOLDS_IN_CV_SPLIT)}
            curr_cv_model_T_cos_sim_score_avgs = []
            curr_cv_model_T_cos_sim_score_std_devs = []
            curr_cv_model_T_mse_by_fold = {i:None for i in range(NUM_FOLDS_IN_CV_SPLIT)}
            curr_cv_model_T_mse_avgs = []
            curr_cv_model_T_mse_std_devs = []
            curr_cv_model_T_dir_deltacon0_scores_by_fold = {i:None for i in range(NUM_FOLDS_IN_CV_SPLIT)}
            curr_cv_model_T_dir_deltacon0_score_avgs = []
            curr_cv_model_T_dir_deltacon0_score_std_devs = []
            curr_cv_model_T_undir_deltacon0_scores_by_fold = {i:None for i in range(NUM_FOLDS_IN_CV_SPLIT)}
            curr_cv_model_T_undir_deltacon0_score_avgs = []
            curr_cv_model_T_undir_deltacon0_score_std_devs = []
            curr_cv_model_T_deltacon0_wDD_scores_by_fold = {i:None for i in range(NUM_FOLDS_IN_CV_SPLIT)}
            curr_cv_model_T_deltacon0_wDD_score_avgs = []
            curr_cv_model_T_deltacon0_wDD_score_std_devs = []
            curr_cv_model_T_deltaffinity_scores_by_fold = {i:None for i in range(NUM_FOLDS_IN_CV_SPLIT)}
            curr_cv_model_T_deltaffinity_score_avgs = []
            curr_cv_model_T_deltaffinity_score_std_devs = []
            curr_cv_model_T_roc_auc_scores_by_fold = {i:None for i in range(NUM_FOLDS_IN_CV_SPLIT)}
            curr_cv_model_T_roc_auc_score_avgs = []
            curr_cv_model_T_roc_auc_score_std_devs = []

            for fold_num, model_file_path in enumerate(cv_split_models):
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t fold_num == ", fold_num, flush=True)
                data_args_filename = None
                data_set_name = None
                for candidate_fold, candidate_dataset_name in enumerate(CURR_CANDIDATE_DATASETS):
                    if candidate_dataset_name in model_file_path:
                        print("\t\t CURR_CANDIDATE_DATASETS == ", CURR_CANDIDATE_DATASETS)
                        print("\t\t cv_split_models == ", cv_split_models)
                        print("\t\t candidate_fold == ", candidate_fold)
                        print("\t\t fold_num == ", fold_num)
                        print("\t\t candidate_dataset_name == ", candidate_dataset_name)
                        print("\t\t model_file_path == ", model_file_path)
                        assert candidate_fold == fold_num
                        assert candidate_dataset_name in CURR_CANDIDATE_DATASET_ARGS[candidate_fold]
                        data_args_filename = CURR_CANDIDATE_DATASET_ARGS[candidate_fold]
                        data_set_name = candidate_dataset_name
                        break
                args_dict["data_set_name"] = data_set_name
                args_dict["data_cached_args_file"] = data_args_filename
                args_dict["true_GC_factors"] = None # need to reset args_dict["true_GC_factors"] to ensure it is not carried over from previous evaluation
                args_dict = read_in_data_args(args_dict, include_gc_views_for_eval=False, read_in_gc_factors_for_eval=True)
                true_gc_factors = args_dict["true_GC_factors"]
                if exclude_self_connections:
                    print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t EXCLUDING SELF-CONNECTIONS FROM EVALUATION", flush=True)
                    for i in range(len(true_gc_factors)):
                        true_gc_factors[i] = (1.-np.expand_dims(np.eye(true_gc_factors[i].shape[0]), 2))*true_gc_factors[i] # zero-out diagonal elements of gc graphs
                    
                NUM_TRUE_FACTORS_IN_DATASET = len(true_gc_factors)

                # initialize records of fold-level stats
                curr_fold_model_cos_sim_scores_by_factor = {i:0. for i in range(NUM_TRUE_FACTORS_IN_DATASET)}
                curr_fold_model_cos_sim_score_avg = 0.
                curr_fold_model_cos_sim_score_std_dev = []
                curr_fold_model_mse_by_factor = {i:0. for i in range(NUM_TRUE_FACTORS_IN_DATASET)}
                curr_fold_model_mse_avg = 0.
                curr_fold_model_mse_std_dev = []
                curr_fold_model_dir_deltacon0_scores_by_factor = {i:0. for i in range(NUM_TRUE_FACTORS_IN_DATASET)}
                curr_fold_model_dir_deltacon0_score_avg = 0.
                curr_fold_model_dir_deltacon0_score_std_dev = []
                curr_fold_model_undir_deltacon0_scores_by_factor = {i:0. for i in range(NUM_TRUE_FACTORS_IN_DATASET)}
                curr_fold_model_undir_deltacon0_score_avg = 0.
                curr_fold_model_undir_deltacon0_score_std_dev = []
                curr_fold_model_deltacon0_wDD_scores_by_factor = {i:0. for i in range(NUM_TRUE_FACTORS_IN_DATASET)}
                curr_fold_model_deltacon0_wDD_score_avg = 0.
                curr_fold_model_deltacon0_wDD_score_std_dev = []
                curr_fold_model_deltaffinity_scores_by_factor = {i:0. for i in range(NUM_TRUE_FACTORS_IN_DATASET)}
                curr_fold_model_deltaffinity_score_avg = 0.
                curr_fold_model_deltaffinity_score_std_dev = []
                curr_fold_model_roc_auc_scores_by_factor = {i:0. for i in range(NUM_TRUE_FACTORS_IN_DATASET)}
                curr_fold_model_roc_auc_score_avg = 0.
                curr_fold_model_roc_auc_score_std_dev = []

                curr_fold_model_T_cos_sim_scores_by_factor = {i:0. for i in range(NUM_TRUE_FACTORS_IN_DATASET)}
                curr_fold_model_T_cos_sim_score_avg = 0.
                curr_fold_model_T_cos_sim_score_std_dev = []
                curr_fold_model_T_mse_by_factor = {i:0. for i in range(NUM_TRUE_FACTORS_IN_DATASET)}
                curr_fold_model_T_mse_avg = 0.
                curr_fold_model_T_mse_std_dev = []
                curr_fold_model_T_dir_deltacon0_scores_by_factor = {i:0. for i in range(NUM_TRUE_FACTORS_IN_DATASET)}
                curr_fold_model_T_dir_deltacon0_score_avg = 0.
                curr_fold_model_T_dir_deltacon0_score_std_dev = []
                curr_fold_model_T_undir_deltacon0_scores_by_factor = {i:0. for i in range(NUM_TRUE_FACTORS_IN_DATASET)}
                curr_fold_model_T_undir_deltacon0_score_avg = 0.
                curr_fold_model_T_undir_deltacon0_score_std_dev = []
                curr_fold_model_T_deltacon0_wDD_scores_by_factor = {i:0. for i in range(NUM_TRUE_FACTORS_IN_DATASET)}
                curr_fold_model_T_deltacon0_wDD_score_avg = 0.
                curr_fold_model_T_deltacon0_wDD_score_std_dev = []
                curr_fold_model_T_deltaffinity_scores_by_factor = {i:0. for i in range(NUM_TRUE_FACTORS_IN_DATASET)}
                curr_fold_model_T_deltaffinity_score_avg = 0.
                curr_fold_model_T_deltaffinity_score_std_dev = []
                curr_fold_model_T_roc_auc_scores_by_factor = {i:0. for i in range(NUM_TRUE_FACTORS_IN_DATASET)}
                curr_fold_model_T_roc_auc_score_avg = 0.
                curr_fold_model_T_roc_auc_score_std_dev = []

                # load model
                curr_model = load_model_for_eval(model_type, model_file_path)

                # get predicted GC factors
                curr_gc_factor_ests = get_model_gc_estimates(curr_model, model_type, NUM_TRUE_FACTORS_IN_DATASET, X=X)
                if sort_unsupervised_ests:
                    curr_gc_factor_ests = sort_unsupervised_estimates(curr_gc_factor_ests, true_gc_factors, cost_criteria=cost_criteria, unsupervised_start_index=unsupervised_start_index)
                if evaluate_identity_baseline:
                    print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t OVER-WRITING CAUSAL ESTIMATES TO EVALUATE IDENTITY BASELINE", flush=True)
                    for i in range(len(curr_gc_factor_ests)):
                        curr_gc_factor_ests[i] = np.expand_dims(np.eye(curr_gc_factor_ests[i].shape[0]), 2) + (0.*curr_gc_factor_ests[i]) # over-write gc graphs with identity
                if exclude_self_connections:
                    print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t EXCLUDING SELF-CONNECTIONS FROM EVALUATION")
                    for i in range(len(curr_gc_factor_ests)):
                        curr_gc_factor_ests[i] = (1.-np.expand_dims(np.eye(curr_gc_factor_ests[i].shape[0]), 2))*curr_gc_factor_ests[i] # zero-out diagonal elements of gc graphs
                        
                curr_normalized_gc_factor_ests = [x/np.max(x) for x in curr_gc_factor_ests]
                if evaluate_identity_baseline:
                    curr_normalized_gc_factor_ests = [x for x in curr_gc_factor_ests]

                if average_estimated_graphs_together and len(curr_normalized_gc_factor_ests) > len(true_gc_factors):
                    assert len(true_gc_factors) == 1
                    print("eval_utils.perform_system_level_estimation_evaluation_of_gs: WARNING - averaging together all estimated gc factors since there are more estimates than true gc factors to compare with.")
                    average_normalized_est = np.zeros(curr_normalized_gc_factor_ests[0].shape)
                    for est in curr_normalized_gc_factor_ests:
                        average_normalized_est = average_normalized_est + est
                    average_normalized_est = (1./len(curr_normalized_gc_factor_ests))*average_normalized_est
                    curr_normalized_gc_factor_ests = [average_normalized_est]

                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t FACTOR LEVEL STATS", flush=True)
                for i, (true_gc, gc_est) in enumerate(zip(true_gc_factors, curr_normalized_gc_factor_ests)):
                    print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t\t factor == ", i)
                    print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t\t WARNING - ONLY COMPUTING NON-LAG GC STATS FOR FAIR COMPARISSON BETW. LAGGED AND NON-LAGGED ESTIMATORS")
                    assert len(true_gc.shape) == 3
                    true_gc = true_gc.sum(axis=2) # sum over the true-lag axis
                    if len(gc_est.shape) == 3:
                        gc_est = gc_est.sum(axis=2) # sum over the true-lag axis
                    assert len(true_gc.shape) == len(gc_est.shape)
                    assert np.isfinite(true_gc.sum())
                    assert np.isfinite(gc_est.sum())

                    curr_cos_sim = compute_cosine_similarity(true_gc, gc_est)
                    curr_fold_model_cos_sim_scores_by_factor[i] = curr_cos_sim
                    curr_fold_model_cos_sim_score_avg += curr_cos_sim
                    curr_fold_model_cos_sim_score_std_dev.append(curr_cos_sim)
                    print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t\t curr_cos_sim == ", curr_cos_sim)
                    curr_T_cos_sim = compute_cosine_similarity(true_gc, gc_est.T)
                    curr_fold_model_T_cos_sim_scores_by_factor[i] = curr_T_cos_sim
                    curr_fold_model_T_cos_sim_score_avg += curr_T_cos_sim
                    curr_fold_model_T_cos_sim_score_std_dev.append(curr_T_cos_sim)
                    print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t\t curr_T_cos_sim == ", curr_T_cos_sim)

                    curr_mse = compute_mse(true_gc, gc_est)
                    curr_fold_model_mse_by_factor[i] = curr_mse
                    curr_fold_model_mse_avg += curr_mse
                    curr_fold_model_mse_std_dev.append(curr_mse)
                    print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t\t curr_mse == ", curr_mse)
                    curr_T_mse = compute_mse(true_gc, gc_est.T)
                    curr_fold_model_T_mse_by_factor[i] = curr_T_mse
                    curr_fold_model_T_mse_avg += curr_T_mse
                    curr_fold_model_T_mse_std_dev.append(curr_T_mse)
                    print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t\t curr_T_mse == ", curr_T_mse)

                    curr_dir_deltacon0 = deltacon0(true_gc, gc_est, eps, make_graphs_undirected=False)
                    curr_fold_model_dir_deltacon0_scores_by_factor[i] = curr_dir_deltacon0
                    curr_fold_model_dir_deltacon0_score_avg += curr_dir_deltacon0
                    curr_fold_model_dir_deltacon0_score_std_dev.append(curr_dir_deltacon0)
                    print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t\t curr_dir_deltacon0 == ", curr_dir_deltacon0)
                    curr_T_dir_deltacon0 = deltacon0(true_gc, gc_est.T, eps, make_graphs_undirected=False)
                    curr_fold_model_T_dir_deltacon0_scores_by_factor[i] = curr_T_dir_deltacon0
                    curr_fold_model_T_dir_deltacon0_score_avg += curr_T_dir_deltacon0
                    curr_fold_model_T_dir_deltacon0_score_std_dev.append(curr_T_dir_deltacon0)
                    print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t\t curr_T_dir_deltacon0 == ", curr_T_dir_deltacon0)

                    curr_undir_deltacon0 = deltacon0(true_gc, gc_est, eps, make_graphs_undirected=True)
                    curr_fold_model_undir_deltacon0_scores_by_factor[i] = curr_undir_deltacon0
                    curr_fold_model_undir_deltacon0_score_avg += curr_undir_deltacon0
                    curr_fold_model_undir_deltacon0_score_std_dev.append(curr_undir_deltacon0)
                    print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t\t curr_undir_deltacon0 == ", curr_undir_deltacon0)
                    curr_T_undir_deltacon0 = deltacon0(true_gc, gc_est.T, eps, make_graphs_undirected=True)
                    curr_fold_model_T_undir_deltacon0_scores_by_factor[i] = curr_T_undir_deltacon0
                    curr_fold_model_T_undir_deltacon0_score_avg += curr_T_undir_deltacon0
                    curr_fold_model_T_undir_deltacon0_score_std_dev.append(curr_T_undir_deltacon0)
                    print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t\t curr_T_undir_deltacon0 == ", curr_T_undir_deltacon0)

                    curr_deltacon0_wDD = deltacon0_with_directed_degrees(true_gc, gc_est, eps, in_degree_coeff=in_degree_coeff, out_degree_coeff=out_degree_coeff)
                    curr_fold_model_deltacon0_wDD_scores_by_factor[i] = curr_deltacon0_wDD
                    curr_fold_model_deltacon0_wDD_score_avg += curr_deltacon0_wDD
                    curr_fold_model_deltacon0_wDD_score_std_dev.append(curr_deltacon0_wDD)
                    print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t\t curr_deltacon0_wDD == ", curr_deltacon0_wDD)
                    curr_T_deltacon0_wDD = deltacon0_with_directed_degrees(true_gc, gc_est.T, eps, in_degree_coeff=in_degree_coeff, out_degree_coeff=out_degree_coeff)
                    curr_fold_model_T_deltacon0_wDD_scores_by_factor[i] = curr_T_deltacon0_wDD
                    curr_fold_model_T_deltacon0_wDD_score_avg += curr_T_deltacon0_wDD
                    curr_fold_model_T_deltacon0_wDD_score_std_dev.append(curr_T_deltacon0_wDD)
                    print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t\t curr_T_deltacon0_wDD == ", curr_T_deltacon0_wDD)

                    curr_deltaffinity = deltaffinity(true_gc, gc_est, eps, max_path_length=max_path_length)
                    curr_fold_model_deltaffinity_scores_by_factor[i] = curr_deltaffinity
                    curr_fold_model_deltaffinity_score_avg += curr_deltaffinity
                    curr_fold_model_deltaffinity_score_std_dev.append(curr_deltaffinity)
                    print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t\t curr_deltaffinity == ", curr_deltaffinity)
                    curr_T_deltaffinity = deltaffinity(true_gc, gc_est.T, eps, max_path_length=max_path_length)
                    curr_fold_model_T_deltaffinity_scores_by_factor[i] = curr_T_deltaffinity
                    curr_fold_model_T_deltaffinity_score_avg += curr_T_deltaffinity
                    curr_fold_model_T_deltaffinity_score_std_dev.append(curr_T_deltaffinity)
                    print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t\t curr_T_deltaffinity == ", curr_T_deltaffinity)

                    true_gc_mask = true_gc > 0.
                    true_gc_mask = true_gc_mask * 1.
                    roc_auc_labels = [int(l) for l in true_gc_mask.flatten()]
                    curr_roc_auc = roc_auc_score(roc_auc_labels, gc_est.flatten())
                    curr_fold_model_roc_auc_scores_by_factor[i] = curr_roc_auc
                    curr_fold_model_roc_auc_score_avg += curr_roc_auc
                    curr_fold_model_roc_auc_score_std_dev.append(curr_roc_auc)
                    print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t\t curr_roc_auc == ", curr_roc_auc, flush=True)
                    curr_T_roc_auc = roc_auc_score(roc_auc_labels, gc_est.T.flatten())
                    curr_fold_model_T_roc_auc_scores_by_factor[i] = curr_T_roc_auc
                    curr_fold_model_T_roc_auc_score_avg += curr_T_roc_auc
                    curr_fold_model_T_roc_auc_score_std_dev.append(curr_T_roc_auc)
                    print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t\t curr_T_roc_auc == ", curr_T_roc_auc, flush=True)

                    plot_gc_est_comparisson(true_gc, gc_est, save_dir+os.sep+"cv"+str(exp_id)+"_fold"+str(fold_num)+"_factor"+str(i)+"_gc_comparisson_vis_"+cv_split_name+".png")
                    plot_gc_est_comparisson(true_gc, gc_est.T, save_dir+os.sep+"cv"+str(exp_id)+"_fold"+str(fold_num)+"_factor"+str(i)+"_gc_comparisson_TRANSPOSED_vis_"+cv_split_name+".png")

                curr_fold_model_cos_sim_score_avg /= NUM_TRUE_FACTORS_IN_DATASET
                curr_fold_model_mse_avg /= NUM_TRUE_FACTORS_IN_DATASET
                curr_fold_model_dir_deltacon0_score_avg /= NUM_TRUE_FACTORS_IN_DATASET
                curr_fold_model_undir_deltacon0_score_avg /= NUM_TRUE_FACTORS_IN_DATASET
                curr_fold_model_deltacon0_wDD_score_avg /= NUM_TRUE_FACTORS_IN_DATASET
                curr_fold_model_deltaffinity_score_avg /= NUM_TRUE_FACTORS_IN_DATASET
                curr_fold_model_roc_auc_score_avg /= NUM_TRUE_FACTORS_IN_DATASET

                curr_fold_model_cos_sim_score_std_dev = np.std(curr_fold_model_cos_sim_score_std_dev)
                curr_fold_model_mse_std_dev = np.std(curr_fold_model_mse_std_dev)
                curr_fold_model_dir_deltacon0_score_std_dev = np.std(curr_fold_model_dir_deltacon0_score_std_dev)
                curr_fold_model_undir_deltacon0_score_std_dev = np.std(curr_fold_model_undir_deltacon0_score_std_dev)
                curr_fold_model_deltacon0_wDD_score_std_dev = np.std(curr_fold_model_deltacon0_wDD_score_std_dev)
                curr_fold_model_deltaffinity_score_std_dev = np.std(curr_fold_model_deltaffinity_score_std_dev)
                curr_fold_model_roc_auc_score_std_dev = np.std(curr_fold_model_roc_auc_score_std_dev)

                curr_cv_model_cos_sim_scores_by_fold[fold_num] = curr_fold_model_cos_sim_scores_by_factor
                curr_cv_model_cos_sim_score_avgs.append(curr_fold_model_cos_sim_score_avg)
                curr_cv_model_cos_sim_score_std_devs.append(curr_fold_model_cos_sim_score_std_dev)
                curr_cv_model_mse_by_fold[fold_num] = curr_fold_model_mse_by_factor
                curr_cv_model_mse_avgs.append(curr_fold_model_mse_avg)
                curr_cv_model_mse_std_devs.append(curr_fold_model_mse_std_dev)
                curr_cv_model_dir_deltacon0_scores_by_fold[fold_num] = curr_fold_model_dir_deltacon0_scores_by_factor
                curr_cv_model_dir_deltacon0_score_avgs.append(curr_fold_model_dir_deltacon0_score_avg)
                curr_cv_model_dir_deltacon0_score_std_devs.append(curr_fold_model_dir_deltacon0_score_std_dev)
                curr_cv_model_undir_deltacon0_scores_by_fold[fold_num] = curr_fold_model_undir_deltacon0_scores_by_factor
                curr_cv_model_undir_deltacon0_score_avgs.append(curr_fold_model_undir_deltacon0_score_avg)
                curr_cv_model_undir_deltacon0_score_std_devs.append(curr_fold_model_undir_deltacon0_score_std_dev)
                curr_cv_model_deltacon0_wDD_scores_by_fold[fold_num] = curr_fold_model_deltacon0_wDD_scores_by_factor
                curr_cv_model_deltacon0_wDD_score_avgs.append(curr_fold_model_deltacon0_wDD_score_avg)
                curr_cv_model_deltacon0_wDD_score_std_devs.append(curr_fold_model_deltacon0_wDD_score_std_dev)
                curr_cv_model_deltaffinity_scores_by_fold[fold_num] = curr_fold_model_deltaffinity_scores_by_factor
                curr_cv_model_deltaffinity_score_avgs.append(curr_fold_model_deltaffinity_score_avg)
                curr_cv_model_deltaffinity_score_std_devs.append(curr_fold_model_deltaffinity_score_std_dev)
                curr_cv_model_roc_auc_scores_by_fold[fold_num] = curr_fold_model_roc_auc_scores_by_factor
                curr_cv_model_roc_auc_score_avgs.append(curr_fold_model_roc_auc_score_avg)
                curr_cv_model_roc_auc_score_std_devs.append(curr_fold_model_roc_auc_score_std_dev)

                curr_fold_model_T_cos_sim_score_avg /= NUM_TRUE_FACTORS_IN_DATASET
                curr_fold_model_T_mse_avg /= NUM_TRUE_FACTORS_IN_DATASET
                curr_fold_model_T_dir_deltacon0_score_avg /= NUM_TRUE_FACTORS_IN_DATASET
                curr_fold_model_T_undir_deltacon0_score_avg /= NUM_TRUE_FACTORS_IN_DATASET
                curr_fold_model_T_deltacon0_wDD_score_avg /= NUM_TRUE_FACTORS_IN_DATASET
                curr_fold_model_T_deltaffinity_score_avg /= NUM_TRUE_FACTORS_IN_DATASET
                curr_fold_model_T_roc_auc_score_avg /= NUM_TRUE_FACTORS_IN_DATASET

                curr_fold_model_T_cos_sim_score_std_dev = np.std(curr_fold_model_T_cos_sim_score_std_dev)
                curr_fold_model_T_mse_std_dev = np.std(curr_fold_model_T_mse_std_dev)
                curr_fold_model_T_dir_deltacon0_score_std_dev = np.std(curr_fold_model_T_dir_deltacon0_score_std_dev)
                curr_fold_model_T_undir_deltacon0_score_std_dev = np.std(curr_fold_model_T_undir_deltacon0_score_std_dev)
                curr_fold_model_T_deltacon0_wDD_score_std_dev = np.std(curr_fold_model_T_deltacon0_wDD_score_std_dev)
                curr_fold_model_T_deltaffinity_score_std_dev = np.std(curr_fold_model_T_deltaffinity_score_std_dev)
                curr_fold_model_T_roc_auc_score_std_dev = np.std(curr_fold_model_T_roc_auc_score_std_dev)

                curr_cv_model_T_cos_sim_scores_by_fold[fold_num] = curr_fold_model_T_cos_sim_scores_by_factor
                curr_cv_model_T_cos_sim_score_avgs.append(curr_fold_model_T_cos_sim_score_avg)
                curr_cv_model_T_cos_sim_score_std_devs.append(curr_fold_model_T_cos_sim_score_std_dev)
                curr_cv_model_T_mse_by_fold[fold_num] = curr_fold_model_T_mse_by_factor
                curr_cv_model_T_mse_avgs.append(curr_fold_model_T_mse_avg)
                curr_cv_model_T_mse_std_devs.append(curr_fold_model_T_mse_std_dev)
                curr_cv_model_T_dir_deltacon0_scores_by_fold[fold_num] = curr_fold_model_T_dir_deltacon0_scores_by_factor
                curr_cv_model_T_dir_deltacon0_score_avgs.append(curr_fold_model_T_dir_deltacon0_score_avg)
                curr_cv_model_T_dir_deltacon0_score_std_devs.append(curr_fold_model_T_dir_deltacon0_score_std_dev)
                curr_cv_model_T_undir_deltacon0_scores_by_fold[fold_num] = curr_fold_model_T_undir_deltacon0_scores_by_factor
                curr_cv_model_T_undir_deltacon0_score_avgs.append(curr_fold_model_T_undir_deltacon0_score_avg)
                curr_cv_model_T_undir_deltacon0_score_std_devs.append(curr_fold_model_T_undir_deltacon0_score_std_dev)
                curr_cv_model_T_deltacon0_wDD_scores_by_fold[fold_num] = curr_fold_model_T_deltacon0_wDD_scores_by_factor
                curr_cv_model_T_deltacon0_wDD_score_avgs.append(curr_fold_model_T_deltacon0_wDD_score_avg)
                curr_cv_model_T_deltacon0_wDD_score_std_devs.append(curr_fold_model_T_deltacon0_wDD_score_std_dev)
                curr_cv_model_T_deltaffinity_scores_by_fold[fold_num] = curr_fold_model_T_deltaffinity_scores_by_factor
                curr_cv_model_T_deltaffinity_score_avgs.append(curr_fold_model_T_deltaffinity_score_avg)
                curr_cv_model_T_deltaffinity_score_std_devs.append(curr_fold_model_T_deltaffinity_score_std_dev)
                curr_cv_model_T_roc_auc_scores_by_fold[fold_num] = curr_fold_model_T_roc_auc_scores_by_factor
                curr_cv_model_T_roc_auc_score_avgs.append(curr_fold_model_T_roc_auc_score_avg)
                curr_cv_model_T_roc_auc_score_std_devs.append(curr_fold_model_T_roc_auc_score_std_dev)

                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_cos_sim_scores_by_factor == ", curr_fold_model_cos_sim_scores_by_factor)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_cos_sim_score_avg == ", curr_fold_model_cos_sim_score_avg)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_cos_sim_score_std_dev == ", curr_fold_model_cos_sim_score_std_dev)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_mse_by_factor == ", curr_fold_model_mse_by_factor)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_mse_avg == ", curr_fold_model_mse_avg)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_mse_std_dev == ", curr_fold_model_mse_std_dev)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_dir_deltacon0_scores_by_factor == ", curr_fold_model_dir_deltacon0_scores_by_factor)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_dir_deltacon0_score_avg == ", curr_fold_model_dir_deltacon0_score_avg)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_dir_deltacon0_score_std_dev == ", curr_fold_model_dir_deltacon0_score_std_dev)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_undir_deltacon0_scores_by_factor == ", curr_fold_model_undir_deltacon0_scores_by_factor)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_undir_deltacon0_score_avg == ", curr_fold_model_undir_deltacon0_score_avg)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_undir_deltacon0_score_std_dev == ", curr_fold_model_undir_deltacon0_score_std_dev)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_deltacon0_wDD_scores_by_factor == ", curr_fold_model_deltacon0_wDD_scores_by_factor)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_deltacon0_wDD_score_avg == ", curr_fold_model_deltacon0_wDD_score_avg)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_deltacon0_wDD_score_std_dev == ", curr_fold_model_deltacon0_wDD_score_std_dev)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_deltaffinity_scores_by_factor == ", curr_fold_model_deltaffinity_scores_by_factor)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_deltaffinity_score_avg == ", curr_fold_model_deltaffinity_score_avg)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_deltaffinity_score_std_dev == ", curr_fold_model_deltaffinity_score_std_dev)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_roc_auc_scores_by_factor == ", curr_fold_model_roc_auc_scores_by_factor)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_roc_auc_score_avg == ", curr_fold_model_roc_auc_score_avg)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_roc_auc_score_std_dev == ", curr_fold_model_roc_auc_score_std_dev, flush=True)

                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_T_cos_sim_scores_by_factor == ", curr_fold_model_T_cos_sim_scores_by_factor)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_T_cos_sim_score_avg == ", curr_fold_model_T_cos_sim_score_avg)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_T_cos_sim_score_std_dev == ", curr_fold_model_T_cos_sim_score_std_dev)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_T_mse_by_factor == ", curr_fold_model_T_mse_by_factor)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_T_mse_avg == ", curr_fold_model_T_mse_avg)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_T_mse_std_dev == ", curr_fold_model_T_mse_std_dev)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_T_dir_deltacon0_scores_by_factor == ", curr_fold_model_T_dir_deltacon0_scores_by_factor)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_T_dir_deltacon0_score_avg == ", curr_fold_model_T_dir_deltacon0_score_avg)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_T_dir_deltacon0_score_std_dev == ", curr_fold_model_T_dir_deltacon0_score_std_dev)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_T_undir_deltacon0_scores_by_factor == ", curr_fold_model_T_undir_deltacon0_scores_by_factor)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_T_undir_deltacon0_score_avg == ", curr_fold_model_T_undir_deltacon0_score_avg)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_T_undir_deltacon0_score_std_dev == ", curr_fold_model_T_undir_deltacon0_score_std_dev)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_T_deltacon0_wDD_scores_by_factor == ", curr_fold_model_T_deltacon0_wDD_scores_by_factor)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_T_deltacon0_wDD_score_avg == ", curr_fold_model_T_deltacon0_wDD_score_avg)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_T_deltacon0_wDD_score_std_dev == ", curr_fold_model_T_deltacon0_wDD_score_std_dev)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_T_deltaffinity_scores_by_factor == ", curr_fold_model_T_deltaffinity_scores_by_factor)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_T_deltaffinity_score_avg == ", curr_fold_model_T_deltaffinity_score_avg)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_T_deltaffinity_score_std_dev == ", curr_fold_model_T_deltaffinity_score_std_dev)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_T_roc_auc_scores_by_factor == ", curr_fold_model_T_roc_auc_scores_by_factor)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_T_roc_auc_score_avg == ", curr_fold_model_T_roc_auc_score_avg)
                print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t\t curr_fold_model_T_roc_auc_score_std_dev == ", curr_fold_model_T_roc_auc_score_std_dev, flush=True)

            # compute current split-level stats
            curr_cross_cv_cos_sim_score_avg = 0.
            curr_cross_cv_cos_sim_score_std_dev = []
            curr_cross_cv_mse_avg = 0.
            curr_cross_cv_mse_std_dev = []
            curr_cross_cv_dir_deltacon0_score_avg = 0.
            curr_cross_cv_dir_deltacon0_score_std_dev = []
            curr_cross_cv_undir_deltacon0_score_avg = 0.
            curr_cross_cv_undir_deltacon0_score_std_dev = []
            curr_cross_cv_deltacon0_wDD_score_avg = 0.
            curr_cross_cv_deltacon0_wDD_score_std_dev = []
            curr_cross_cv_deltaffinity_score_avg = 0.
            curr_cross_cv_deltaffinity_score_std_dev = []
            curr_cross_cv_roc_auc_score_avg = 0.
            curr_cross_cv_roc_auc_score_std_dev = []

            curr_cross_cv_T_cos_sim_score_avg = 0.
            curr_cross_cv_T_cos_sim_score_std_dev = []
            curr_cross_cv_T_mse_avg = 0.
            curr_cross_cv_T_mse_std_dev = []
            curr_cross_cv_T_dir_deltacon0_score_avg = 0.
            curr_cross_cv_T_dir_deltacon0_score_std_dev = []
            curr_cross_cv_T_undir_deltacon0_score_avg = 0.
            curr_cross_cv_T_undir_deltacon0_score_std_dev = []
            curr_cross_cv_T_deltacon0_wDD_score_avg = 0.
            curr_cross_cv_T_deltacon0_wDD_score_std_dev = []
            curr_cross_cv_T_deltaffinity_score_avg = 0.
            curr_cross_cv_T_deltaffinity_score_std_dev = []
            curr_cross_cv_T_roc_auc_score_avg = 0.
            curr_cross_cv_T_roc_auc_score_std_dev = []

            for fold_key in curr_cv_model_cos_sim_scores_by_fold.keys():
                for factor_key in curr_cv_model_cos_sim_scores_by_fold[fold_key].keys():
                    curr_cross_cv_cos_sim_score_avg += curr_cv_model_cos_sim_scores_by_fold[fold_key][factor_key]
                    curr_cross_cv_cos_sim_score_std_dev.append(curr_cv_model_cos_sim_scores_by_fold[fold_key][factor_key])
                    curr_cross_cv_mse_avg += curr_cv_model_mse_by_fold[fold_key][factor_key]
                    curr_cross_cv_mse_std_dev.append(curr_cv_model_mse_by_fold[fold_key][factor_key])
                    curr_cross_cv_dir_deltacon0_score_avg += curr_cv_model_dir_deltacon0_scores_by_fold[fold_key][factor_key]
                    curr_cross_cv_dir_deltacon0_score_std_dev.append(curr_cv_model_dir_deltacon0_scores_by_fold[fold_key][factor_key])
                    curr_cross_cv_undir_deltacon0_score_avg += curr_cv_model_undir_deltacon0_scores_by_fold[fold_key][factor_key]
                    curr_cross_cv_undir_deltacon0_score_std_dev.append(curr_cv_model_undir_deltacon0_scores_by_fold[fold_key][factor_key])
                    curr_cross_cv_deltacon0_wDD_score_avg += curr_cv_model_deltacon0_wDD_scores_by_fold[fold_key][factor_key]
                    curr_cross_cv_deltacon0_wDD_score_std_dev.append(curr_cv_model_deltacon0_wDD_scores_by_fold[fold_key][factor_key])
                    curr_cross_cv_deltaffinity_score_avg += curr_cv_model_deltaffinity_scores_by_fold[fold_key][factor_key]
                    curr_cross_cv_deltaffinity_score_std_dev.append(curr_cv_model_deltaffinity_scores_by_fold[fold_key][factor_key])
                    curr_cross_cv_roc_auc_score_avg += curr_cv_model_roc_auc_scores_by_fold[fold_key][factor_key]
                    curr_cross_cv_roc_auc_score_std_dev.append(curr_cv_model_roc_auc_scores_by_fold[fold_key][factor_key])

                    curr_cross_cv_T_cos_sim_score_avg += curr_cv_model_T_cos_sim_scores_by_fold[fold_key][factor_key]
                    curr_cross_cv_T_cos_sim_score_std_dev.append(curr_cv_model_T_cos_sim_scores_by_fold[fold_key][factor_key])
                    curr_cross_cv_T_mse_avg += curr_cv_model_T_mse_by_fold[fold_key][factor_key]
                    curr_cross_cv_T_mse_std_dev.append(curr_cv_model_T_mse_by_fold[fold_key][factor_key])
                    curr_cross_cv_T_dir_deltacon0_score_avg += curr_cv_model_T_dir_deltacon0_scores_by_fold[fold_key][factor_key]
                    curr_cross_cv_T_dir_deltacon0_score_std_dev.append(curr_cv_model_T_dir_deltacon0_scores_by_fold[fold_key][factor_key])
                    curr_cross_cv_T_undir_deltacon0_score_avg += curr_cv_model_T_undir_deltacon0_scores_by_fold[fold_key][factor_key]
                    curr_cross_cv_T_undir_deltacon0_score_std_dev.append(curr_cv_model_T_undir_deltacon0_scores_by_fold[fold_key][factor_key])
                    curr_cross_cv_T_deltacon0_wDD_score_avg += curr_cv_model_T_deltacon0_wDD_scores_by_fold[fold_key][factor_key]
                    curr_cross_cv_T_deltacon0_wDD_score_std_dev.append(curr_cv_model_T_deltacon0_wDD_scores_by_fold[fold_key][factor_key])
                    curr_cross_cv_T_deltaffinity_score_avg += curr_cv_model_T_deltaffinity_scores_by_fold[fold_key][factor_key]
                    curr_cross_cv_T_deltaffinity_score_std_dev.append(curr_cv_model_T_deltaffinity_scores_by_fold[fold_key][factor_key])
                    curr_cross_cv_T_roc_auc_score_avg += curr_cv_model_T_roc_auc_scores_by_fold[fold_key][factor_key]
                    curr_cross_cv_T_roc_auc_score_std_dev.append(curr_cv_model_T_roc_auc_scores_by_fold[fold_key][factor_key])

            num_stat_samples = 1.*len(curr_cross_cv_cos_sim_score_std_dev)

            curr_cross_cv_cos_sim_score_avg /= num_stat_samples
            curr_cross_cv_cos_sim_score_std_dev = np.std(curr_cross_cv_cos_sim_score_std_dev)
            curr_cross_cv_mse_avg /= num_stat_samples
            curr_cross_cv_mse_std_dev = np.std(curr_cross_cv_mse_std_dev)
            curr_cross_cv_dir_deltacon0_score_avg /= num_stat_samples
            curr_cross_cv_dir_deltacon0_score_std_dev = np.std(curr_cross_cv_dir_deltacon0_score_std_dev)
            curr_cross_cv_undir_deltacon0_score_avg /= num_stat_samples
            curr_cross_cv_undir_deltacon0_score_std_dev = np.std(curr_cross_cv_undir_deltacon0_score_std_dev)
            curr_cross_cv_deltacon0_wDD_score_avg /= num_stat_samples
            curr_cross_cv_deltacon0_wDD_score_std_dev = np.std(curr_cross_cv_deltacon0_wDD_score_std_dev)
            curr_cross_cv_deltaffinity_score_avg /= num_stat_samples
            curr_cross_cv_deltaffinity_score_std_dev = np.std(curr_cross_cv_deltaffinity_score_std_dev)
            curr_cross_cv_roc_auc_score_avg /= num_stat_samples
            curr_cross_cv_roc_auc_score_std_dev = np.std(curr_cross_cv_roc_auc_score_std_dev)

            curr_cross_cv_T_cos_sim_score_avg /= num_stat_samples
            curr_cross_cv_T_cos_sim_score_std_dev = np.std(curr_cross_cv_T_cos_sim_score_std_dev)
            curr_cross_cv_T_mse_avg /= num_stat_samples
            curr_cross_cv_T_mse_std_dev = np.std(curr_cross_cv_T_mse_std_dev)
            curr_cross_cv_T_dir_deltacon0_score_avg /= num_stat_samples
            curr_cross_cv_T_dir_deltacon0_score_std_dev = np.std(curr_cross_cv_T_dir_deltacon0_score_std_dev)
            curr_cross_cv_T_undir_deltacon0_score_avg /= num_stat_samples
            curr_cross_cv_T_undir_deltacon0_score_std_dev = np.std(curr_cross_cv_T_undir_deltacon0_score_std_dev)
            curr_cross_cv_T_deltacon0_wDD_score_avg /= num_stat_samples
            curr_cross_cv_T_deltacon0_wDD_score_std_dev = np.std(curr_cross_cv_T_deltacon0_wDD_score_std_dev)
            curr_cross_cv_T_deltaffinity_score_avg /= num_stat_samples
            curr_cross_cv_T_deltaffinity_score_std_dev = np.std(curr_cross_cv_T_deltaffinity_score_std_dev)
            curr_cross_cv_T_roc_auc_score_avg /= num_stat_samples
            curr_cross_cv_T_roc_auc_score_std_dev = np.std(curr_cross_cv_T_roc_auc_score_std_dev)

            # report current split-level stats
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_cos_sim_score_avg == ", curr_cross_cv_cos_sim_score_avg)
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_cos_sim_score_std_dev == ", curr_cross_cv_cos_sim_score_std_dev)
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_mse_avg == ", curr_cross_cv_mse_avg)
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_mse_std_dev == ", curr_cross_cv_mse_std_dev)
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_dir_deltacon0_score_avg == ", curr_cross_cv_dir_deltacon0_score_avg)
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_dir_deltacon0_score_std_dev == ", curr_cross_cv_dir_deltacon0_score_std_dev)
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_undir_deltacon0_score_avg == ", curr_cross_cv_undir_deltacon0_score_avg)
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_undir_deltacon0_score_std_dev == ", curr_cross_cv_undir_deltacon0_score_std_dev)
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_deltacon0_wDD_score_avg == ", curr_cross_cv_deltacon0_wDD_score_avg)
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_deltacon0_wDD_score_std_dev == ", curr_cross_cv_deltacon0_wDD_score_std_dev)
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_deltaffinity_score_avg == ", curr_cross_cv_deltaffinity_score_avg)
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_deltaffinity_score_std_dev == ", curr_cross_cv_deltaffinity_score_std_dev)
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_roc_auc_score_avg == ", curr_cross_cv_roc_auc_score_avg)
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_roc_auc_score_std_dev == ", curr_cross_cv_roc_auc_score_std_dev, flush=True)

            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_T_cos_sim_score_avg == ", curr_cross_cv_T_cos_sim_score_avg)
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_T_cos_sim_score_std_dev == ", curr_cross_cv_T_cos_sim_score_std_dev)
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_T_mse_avg == ", curr_cross_cv_T_mse_avg)
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_T_mse_std_dev == ", curr_cross_cv_T_mse_std_dev)
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_T_dir_deltacon0_score_avg == ", curr_cross_cv_T_dir_deltacon0_score_avg)
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_T_dir_deltacon0_score_std_dev == ", curr_cross_cv_T_dir_deltacon0_score_std_dev)
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_T_undir_deltacon0_score_avg == ", curr_cross_cv_T_undir_deltacon0_score_avg)
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_T_undir_deltacon0_score_std_dev == ", curr_cross_cv_T_undir_deltacon0_score_std_dev)
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_T_deltacon0_wDD_score_avg == ", curr_cross_cv_T_deltacon0_wDD_score_avg)
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_T_deltacon0_wDD_score_std_dev == ", curr_cross_cv_T_deltacon0_wDD_score_std_dev)
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_T_deltaffinity_score_avg == ", curr_cross_cv_T_deltaffinity_score_avg)
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_T_deltaffinity_score_std_dev == ", curr_cross_cv_T_deltaffinity_score_std_dev)
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_T_roc_auc_score_avg == ", curr_cross_cv_T_roc_auc_score_avg)
            print("eval_utils.perform_system_level_estimation_evaluation_of_cv_model: \t curr_cross_cv_T_roc_auc_score_std_dev == ", curr_cross_cv_T_roc_auc_score_std_dev, flush=True)

            # save current cv summary stats
            with open(save_dir+os.sep+"cv"+str(exp_id)+"_"+cv_split_name+"_summary_stats.pkl", 'wb') as outfile:
                pkl.dump({
                    "curr_cv_model_cos_sim_scores_by_fold": curr_cv_model_cos_sim_scores_by_fold, 
                    "curr_cv_model_cos_sim_score_avgs": curr_cv_model_cos_sim_score_avgs, 
                    "curr_cv_model_cos_sim_score_std_devs": curr_cv_model_cos_sim_score_std_devs, 
                    "curr_cv_model_mse_by_fold": curr_cv_model_mse_by_fold, 
                    "curr_cv_model_mse_avgs": curr_cv_model_mse_avgs, 
                    "curr_cv_model_mse_std_devs": curr_cv_model_mse_std_devs, 
                    "curr_cv_model_dir_deltacon0_scores_by_fold": curr_cv_model_dir_deltacon0_scores_by_fold, 
                    "curr_cv_model_dir_deltacon0_score_avgs": curr_cv_model_dir_deltacon0_score_avgs, 
                    "curr_cv_model_dir_deltacon0_score_std_devs": curr_cv_model_dir_deltacon0_score_std_devs, 
                    "curr_cv_model_undir_deltacon0_scores_by_fold": curr_cv_model_undir_deltacon0_scores_by_fold, 
                    "curr_cv_model_undir_deltacon0_score_avgs": curr_cv_model_undir_deltacon0_score_avgs, 
                    "curr_cv_model_undir_deltacon0_score_std_devs": curr_cv_model_undir_deltacon0_score_std_devs, 
                    "curr_cv_model_deltacon0_wDD_scores_by_fold": curr_cv_model_deltacon0_wDD_scores_by_fold, 
                    "curr_cv_model_deltacon0_wDD_score_avgs": curr_cv_model_deltacon0_wDD_score_avgs, 
                    "curr_cv_model_deltacon0_wDD_score_std_devs": curr_cv_model_deltacon0_wDD_score_std_devs, 
                    "curr_cv_model_deltaffinity_scores_by_fold": curr_cv_model_deltaffinity_scores_by_fold, 
                    "curr_cv_model_deltaffinity_score_avgs": curr_cv_model_deltaffinity_score_avgs, 
                    "curr_cv_model_deltaffinity_score_std_devs": curr_cv_model_deltaffinity_score_std_devs, 
                    "curr_cv_model_roc_auc_scores_by_fold": curr_cv_model_roc_auc_scores_by_fold, 
                    "curr_cv_model_roc_auc_score_avgs": curr_cv_model_roc_auc_score_avgs, 
                    "curr_cv_model_roc_auc_score_std_devs": curr_cv_model_roc_auc_score_std_devs, 
                    "curr_cross_cv_cos_sim_score_avg": curr_cross_cv_cos_sim_score_avg, 
                    "curr_cross_cv_cos_sim_score_std_dev": curr_cross_cv_cos_sim_score_std_dev, 
                    "curr_cross_cv_mse_avg": curr_cross_cv_mse_avg, 
                    "curr_cross_cv_mse_std_dev": curr_cross_cv_mse_std_dev, 
                    "curr_cross_cv_dir_deltacon0_score_avg": curr_cross_cv_dir_deltacon0_score_avg, 
                    "curr_cross_cv_dir_deltacon0_score_std_dev": curr_cross_cv_dir_deltacon0_score_std_dev, 
                    "curr_cross_cv_undir_deltacon0_score_avg": curr_cross_cv_undir_deltacon0_score_avg, 
                    "curr_cross_cv_undir_deltacon0_score_std_dev": curr_cross_cv_undir_deltacon0_score_std_dev, 
                    "curr_cross_cv_deltacon0_wDD_score_avg": curr_cross_cv_deltacon0_wDD_score_avg, 
                    "curr_cross_cv_deltacon0_wDD_score_std_dev": curr_cross_cv_deltacon0_wDD_score_std_dev, 
                    "curr_cross_cv_deltaffinity_score_avg": curr_cross_cv_deltaffinity_score_avg, 
                    "curr_cross_cv_deltaffinity_score_std_dev": curr_cross_cv_deltaffinity_score_std_dev, 
                    "curr_cross_cv_roc_auc_score_avg": curr_cross_cv_roc_auc_score_avg, 
                    "curr_cross_cv_roc_auc_score_std_dev": curr_cross_cv_roc_auc_score_std_dev, 
                    "curr_cv_model_T_cos_sim_scores_by_fold": curr_cv_model_T_cos_sim_scores_by_fold, 
                    "curr_cv_model_T_cos_sim_score_avgs": curr_cv_model_T_cos_sim_score_avgs, 
                    "curr_cv_model_T_cos_sim_score_std_devs": curr_cv_model_T_cos_sim_score_std_devs, 
                    "curr_cv_model_T_mse_by_fold": curr_cv_model_T_mse_by_fold, 
                    "curr_cv_model_T_mse_avgs": curr_cv_model_T_mse_avgs, 
                    "curr_cv_model_T_mse_std_devs": curr_cv_model_T_mse_std_devs, 
                    "curr_cv_model_T_dir_deltacon0_scores_by_fold": curr_cv_model_T_dir_deltacon0_scores_by_fold, 
                    "curr_cv_model_T_dir_deltacon0_score_avgs": curr_cv_model_T_dir_deltacon0_score_avgs, 
                    "curr_cv_model_T_dir_deltacon0_score_std_devs": curr_cv_model_T_dir_deltacon0_score_std_devs, 
                    "curr_cv_model_T_undir_deltacon0_scores_by_fold": curr_cv_model_T_undir_deltacon0_scores_by_fold, 
                    "curr_cv_model_T_undir_deltacon0_score_avgs": curr_cv_model_T_undir_deltacon0_score_avgs, 
                    "curr_cv_model_T_undir_deltacon0_score_std_devs": curr_cv_model_T_undir_deltacon0_score_std_devs, 
                    "curr_cv_model_T_deltacon0_wDD_scores_by_fold": curr_cv_model_T_deltacon0_wDD_scores_by_fold, 
                    "curr_cv_model_T_deltacon0_wDD_score_avgs": curr_cv_model_T_deltacon0_wDD_score_avgs, 
                    "curr_cv_model_T_deltacon0_wDD_score_std_devs": curr_cv_model_T_deltacon0_wDD_score_std_devs, 
                    "curr_cv_model_T_deltaffinity_scores_by_fold": curr_cv_model_T_deltaffinity_scores_by_fold, 
                    "curr_cv_model_T_deltaffinity_score_avgs": curr_cv_model_T_deltaffinity_score_avgs, 
                    "curr_cv_model_T_deltaffinity_score_std_devs": curr_cv_model_T_deltaffinity_score_std_devs, 
                    "curr_cv_model_T_roc_auc_scores_by_fold": curr_cv_model_T_roc_auc_scores_by_fold, 
                    "curr_cv_model_T_roc_auc_score_avgs": curr_cv_model_T_roc_auc_score_avgs, 
                    "curr_cv_model_T_roc_auc_score_std_devs": curr_cv_model_T_roc_auc_score_std_devs, 
                    "curr_cross_cv_T_cos_sim_score_avg": curr_cross_cv_T_cos_sim_score_avg, 
                    "curr_cross_cv_T_cos_sim_score_std_dev": curr_cross_cv_T_cos_sim_score_std_dev, 
                    "curr_cross_cv_T_mse_avg": curr_cross_cv_T_mse_avg, 
                    "curr_cross_cv_T_mse_std_dev": curr_cross_cv_T_mse_std_dev, 
                    "curr_cross_cv_T_dir_deltacon0_score_avg": curr_cross_cv_T_dir_deltacon0_score_avg, 
                    "curr_cross_cv_T_dir_deltacon0_score_std_dev": curr_cross_cv_T_dir_deltacon0_score_std_dev, 
                    "curr_cross_cv_T_undir_deltacon0_score_avg": curr_cross_cv_T_undir_deltacon0_score_avg, 
                    "curr_cross_cv_T_undir_deltacon0_score_std_dev": curr_cross_cv_T_undir_deltacon0_score_std_dev, 
                    "curr_cross_cv_T_deltacon0_wDD_score_avg": curr_cross_cv_T_deltacon0_wDD_score_avg, 
                    "curr_cross_cv_T_deltacon0_wDD_score_std_dev": curr_cross_cv_T_deltacon0_wDD_score_std_dev, 
                    "curr_cross_cv_T_deltaffinity_score_avg": curr_cross_cv_T_deltaffinity_score_avg, 
                    "curr_cross_cv_T_deltaffinity_score_std_dev": curr_cross_cv_T_deltaffinity_score_std_dev, 
                    "curr_cross_cv_T_roc_auc_score_avg": curr_cross_cv_T_roc_auc_score_avg, 
                    "curr_cross_cv_T_roc_auc_score_std_dev": curr_cross_cv_T_roc_auc_score_std_dev, 
                }, outfile)

    print("\n\n eval_utils.perform_system_level_estimation_evaluation_of_cv_model: COMPLETED ANALYSIS -------------------------------------------------\n", flush=True)
    pass


def perform_system_level_estimation_evaluation_of_gs(model_type, save_dir, trained_models_root_path, true_gc_factors, trained_model_file_name="final_best_model.bin", 
                                                     eps=0.1, in_degree_coeff=1., out_degree_coeff=1., max_path_length=None, X=None, sort_unsupervised_ests=False, average_estimated_graphs_together=False):
    NUM_TRUE_FACTORS_IN_DATASET = len(true_gc_factors)
    # grab all results folders and associated .pkl summary files
    model_files = {
        x:trained_models_root_path+os.sep+x+os.sep+trained_model_file_name for x in os.listdir(trained_models_root_path) if os.path.isfile(trained_models_root_path+os.sep+x+os.sep+trained_model_file_name)
    }
    print("perform_system_level_estimation_evaluation_of_gs: model_files == ", model_files)

    # report summaries from .pkl files
    model_cos_sim_scores_by_factor = {i:[] for i in range(NUM_TRUE_FACTORS_IN_DATASET)}
    model_cos_sim_score_avgs = []
    model_cos_sim_score_std_devs = []
    model_mse_by_factor = {i:[] for i in range(NUM_TRUE_FACTORS_IN_DATASET)}
    model_mse_avgs = []
    model_mse_std_devs = []
    model_dir_deltacon0_scores_by_factor = {i:[] for i in range(NUM_TRUE_FACTORS_IN_DATASET)}
    model_dir_deltacon0_score_avgs = []
    model_dir_deltacon0_score_std_devs = []
    model_undir_deltacon0_scores_by_factor = {i:[] for i in range(NUM_TRUE_FACTORS_IN_DATASET)}
    model_undir_deltacon0_score_avgs = []
    model_undir_deltacon0_score_std_devs = []
    model_deltacon0_wDD_scores_by_factor = {i:[] for i in range(NUM_TRUE_FACTORS_IN_DATASET)}
    model_deltacon0_wDD_score_avgs = []
    model_deltacon0_wDD_score_std_devs = []
    model_deltaffinity_scores_by_factor = {i:[] for i in range(NUM_TRUE_FACTORS_IN_DATASET)}
    model_deltaffinity_score_avgs = []
    model_deltaffinity_score_std_devs = []
    model_roc_auc_scores_by_factor = {i:[] for i in range(NUM_TRUE_FACTORS_IN_DATASET)}
    model_roc_auc_score_avgs = []
    model_roc_auc_score_std_devs = []

    candidate_model_files = []
    for model_folder_name in model_files.keys():
        # load model
        curr_model = load_model_for_eval(model_type, model_files[model_folder_name])

        # get predicted GC factors
        curr_gc_factor_ests = get_model_gc_estimates(curr_model, model_type, NUM_TRUE_FACTORS_IN_DATASET, X=X)
        if sort_unsupervised_ests:
            curr_gc_factor_ests = sort_unsupervised_estimates(curr_gc_factor_ests, true_gc_factors, cost_criteria=cost_criteria, unsupervised_start_index=unsupervised_start_index)
        curr_normalized_gc_factor_ests = [x/np.max(x) for x in curr_gc_factor_ests]

        if average_estimated_graphs_together and len(curr_normalized_gc_factor_ests) > len(true_gc_factors):
            assert len(true_gc_factors) == 1
            print("eval_utils.perform_system_level_estimation_evaluation_of_gs: WARNING - averaging together all estimated gc factors since there are more estimates than true gc factors to compare with.")
            average_normalized_est = np.zeros(curr_normalized_gc_factor_ests[0].shape)
            for est in curr_normalized_gc_factor_ests:
                average_normalized_est = average_normalized_est + est
            average_normalized_est = (1./len(curr_normalized_gc_factor_ests))*average_normalized_est
            curr_normalized_gc_factor_ests = [average_normalized_est]

        ignore_model = False
        for i, (true_gc, gc_est) in enumerate(zip(true_gc_factors, curr_normalized_gc_factor_ests)):
            if not np.isfinite(np.sum(gc_est)):
                ignore_model = True
        if ignore_model:
            print("perform_system_level_estimation_evaluation_of_gs: INGORING MODEL DUE TO NON-FINITE GC-ESTIMATE BEING DETECTED; THE MODEL CAN BE FOUND AT ", model_folder_name, flush=True)
        else:
            candidate_model_files.append(model_folder_name)
            # compute roc-auc scores and cosine similarity scores associated with each gc factor estimation
            curr_cos_sim_avg = 0.
            curr_cos_sim_std_dev = []
            curr_mse_avg = 0.
            curr_mse_std_dev = []
            curr_dir_deltacon0_avg = 0.
            curr_dir_deltacon0_std_dev = []
            curr_undir_deltacon0_avg = 0.
            curr_undir_deltacon0_std_dev = []
            curr_deltacon0_wDD_avg = 0.
            curr_deltacon0_wDD_std_dev = []
            curr_deltaffinity_avg = 0.
            curr_deltaffinity_std_dev = []
            curr_roc_auc_avg = 0.
            curr_roc_auc_std_dev = []

            for i, (true_gc, gc_est) in enumerate(zip(true_gc_factors, curr_normalized_gc_factor_ests)):

                curr_cos_sim = compute_cosine_similarity(true_gc, gc_est)
                model_cos_sim_scores_by_factor[i].append(curr_cos_sim)
                curr_cos_sim_avg += curr_cos_sim
                curr_cos_sim_std_dev.append(curr_cos_sim)

                curr_mse = compute_mse(true_gc, gc_est)
                model_mse_by_factor[i].append(curr_mse)
                curr_mse_avg += curr_mse
                curr_mse_std_dev.append(curr_mse)

                curr_dir_deltacon0 = deltacon0(true_gc, gc_est, eps, make_graphs_undirected=False)
                model_dir_deltacon0_scores_by_factor[i].append(curr_dir_deltacon0)
                curr_dir_deltacon0_avg += curr_dir_deltacon0
                curr_dir_deltacon0_std_dev.append(curr_dir_deltacon0)

                curr_undir_deltacon0 = deltacon0(true_gc, gc_est, eps, make_graphs_undirected=True)
                model_undir_deltacon0_scores_by_factor[i].append(curr_undir_deltacon0)
                curr_undir_deltacon0_avg += curr_undir_deltacon0
                curr_undir_deltacon0_std_dev.append(curr_undir_deltacon0)

                curr_deltacon0_wDD = deltacon0_with_directed_degrees(true_gc, gc_est, eps, in_degree_coeff=in_degree_coeff, out_degree_coeff=out_degree_coeff)
                model_deltacon0_wDD_scores_by_factor[i].append(curr_deltacon0_wDD)
                curr_deltacon0_wDD_avg += curr_deltacon0_wDD
                curr_deltacon0_wDD_std_dev.append(curr_deltacon0_wDD)

                curr_deltaffinity = deltaffinity(true_gc, gc_est, eps, max_path_length=max_path_length)
                model_deltaffinity_scores_by_factor[i].append(curr_deltaffinity)
                curr_deltaffinity_avg += curr_deltaffinity
                curr_deltaffinity_std_dev.append(curr_deltaffinity)

                true_gc_mask = true_gc > 0.
                true_gc_mask = true_gc_mask * 1.
                roc_auc_labels = [int(l) for l in true_gc_mask.flatten()]
                curr_roc_auc = roc_auc_score(roc_auc_labels, gc_est.flatten())
                model_roc_auc_scores_by_factor[i].append(curr_roc_auc)
                curr_roc_auc_avg += curr_roc_auc
                curr_roc_auc_std_dev.append(curr_roc_auc)

            curr_cos_sim_avg /= NUM_TRUE_FACTORS_IN_DATASET
            curr_mse_avg /= NUM_TRUE_FACTORS_IN_DATASET
            curr_dir_deltacon0_avg /= NUM_TRUE_FACTORS_IN_DATASET
            curr_undir_deltacon0_avg /= NUM_TRUE_FACTORS_IN_DATASET
            curr_deltacon0_wDD_avg /= NUM_TRUE_FACTORS_IN_DATASET
            curr_deltaffinity_avg /= NUM_TRUE_FACTORS_IN_DATASET
            curr_roc_auc_avg /= NUM_TRUE_FACTORS_IN_DATASET
            curr_cos_sim_std_dev = np.std(curr_cos_sim_std_dev)
            curr_mse_std_dev = np.std(curr_mse_std_dev)
            curr_dir_deltacon0_std_dev = np.std(curr_dir_deltacon0_std_dev)
            curr_undir_deltacon0_std_dev = np.std(curr_undir_deltacon0_std_dev)
            curr_deltacon0_wDD_std_dev = np.std(curr_deltacon0_wDD_std_dev)
            curr_deltaffinity_std_dev = np.std(curr_deltaffinity_std_dev)
            curr_roc_auc_std_dev = np.std(curr_roc_auc_std_dev)

            model_cos_sim_score_avgs.append(curr_cos_sim_avg)
            model_cos_sim_score_std_devs.append(curr_cos_sim_std_dev)
            model_mse_avgs.append(curr_mse_avg)
            model_mse_std_devs.append(curr_mse_std_dev)
            model_dir_deltacon0_score_avgs.append(curr_dir_deltacon0_avg)
            model_dir_deltacon0_score_std_devs.append(curr_dir_deltacon0_std_dev)
            model_undir_deltacon0_score_avgs.append(curr_undir_deltacon0_avg)
            model_undir_deltacon0_score_std_devs.append(curr_undir_deltacon0_std_dev)
            model_deltacon0_wDD_score_avgs.append(curr_deltacon0_wDD_avg)
            model_deltacon0_wDD_score_std_devs.append(curr_deltacon0_wDD_std_dev)
            model_deltaffinity_score_avgs.append(curr_deltaffinity_avg)
            model_deltaffinity_score_std_devs.append(curr_deltaffinity_std_dev)
            model_roc_auc_score_avgs.append(curr_roc_auc_avg)
            model_roc_auc_score_std_devs.append(curr_roc_auc_std_dev)
    
    max_cos_sim_score = np.max(model_cos_sim_score_avgs)
    min_mse = np.min(model_mse_avgs)
    max_dir_deltacon0_score = np.max(model_dir_deltacon0_score_avgs)
    max_undir_deltacon0_score = np.max(model_undir_deltacon0_score_avgs)
    max_deltacon0_wDD_score = np.max(model_deltacon0_wDD_score_avgs)
    max_deltaffinity_score = np.max(model_deltaffinity_score_avgs)
    max_roc_auc_score = np.max(model_roc_auc_score_avgs)

    models_with_max_cos_sim_score = [i for i in range(len(candidate_model_files)) if model_cos_sim_score_avgs[i] == max_cos_sim_score]
    models_with_min_mse = [i for i in range(len(candidate_model_files)) if model_mse_avgs[i] == min_mse]
    models_with_max_dir_deltacon0_score = [i for i in range(len(candidate_model_files)) if model_dir_deltacon0_score_avgs[i] == max_dir_deltacon0_score]
    models_with_max_undir_deltacon0_score = [i for i in range(len(candidate_model_files)) if model_undir_deltacon0_score_avgs[i] == max_undir_deltacon0_score]
    models_with_max_deltacon0_wDD_score = [i for i in range(len(candidate_model_files)) if model_deltacon0_wDD_score_avgs[i] == max_deltacon0_wDD_score]
    models_with_max_deltaffinity_score = [i for i in range(len(candidate_model_files)) if model_deltaffinity_score_avgs[i] == max_deltaffinity_score]
    models_with_max_roc_auc_score = [i for i in range(len(candidate_model_files)) if model_roc_auc_score_avgs[i] == max_roc_auc_score]

    print("\n candidate_model_files: ", candidate_model_files)
    
    for i, model_folder_name in enumerate(candidate_model_files):
        print("NOTABLE SYSTEM-LEVEL ESTIMATION CHARACTERISTICS OF MODEL(S) FROM model_folder_name: ", model_folder_name)
        if i in models_with_max_cos_sim_score:
            print("<<< BEST MODEL BY COS-SIM SCORE: MODEL i==", i, " >>> ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("\t model_folder_name == ", model_folder_name)
            print("\t max_cos_sim_score == ", max_cos_sim_score)
            print("\t model_mse_avgs[i] == ", model_mse_avgs[i])
            print("\t model_dir_deltacon0_score_avgs[i] == ", model_dir_deltacon0_score_avgs[i])
            print("\t model_undir_deltacon0_score_avgs[i] == ", model_undir_deltacon0_score_avgs[i])
            print("\t model_deltacon0_wDD_score_avgs[i] == ", model_deltacon0_wDD_score_avgs[i])
            print("\t model_deltaffinity_score_avgs[i] == ", model_deltaffinity_score_avgs[i])
            print("\t model_roc_auc_score_avgs[i] == ", model_roc_auc_score_avgs[i])
            print("\t model_cos_sim_score_std_devs[i] == ", model_cos_sim_score_std_devs[i])
            print("\t model_mse_std_devs[i] == ", model_mse_std_devs[i])
            print("\t model_dir_deltacon0_score_std_devs[i] == ", model_dir_deltacon0_score_std_devs[i])
            print("\t model_undir_deltacon0_score_std_devs[i] == ", model_undir_deltacon0_score_std_devs[i])
            print("\t model_deltacon0_wDD_score_std_devs[i] == ", model_deltacon0_wDD_score_std_devs[i])
            print("\t model_deltaffinity_score_std_devs[i] == ", model_deltaffinity_score_std_devs[i])
            print("\t model_roc_auc_score_std_devs[i] == ", model_roc_auc_score_std_devs[i])
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        if i in models_with_min_mse:
            print("<<< BEST MODEL BY MSE: MODEL i==", i, " >>> EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            print("\t model_folder_name == ", model_folder_name)
            print("\t model_cos_sim_score_avgs[i] == ", model_cos_sim_score_avgs[i])
            print("\t min_mse == ", min_mse)
            print("\t model_dir_deltacon0_score_avgs[i] == ", model_dir_deltacon0_score_avgs[i])
            print("\t model_undir_deltacon0_score_avgs[i] == ", model_undir_deltacon0_score_avgs[i])
            print("\t model_deltacon0_wDD_score_avgs[i] == ", model_deltacon0_wDD_score_avgs[i])
            print("\t model_deltaffinity_score_avgs[i] == ", model_deltaffinity_score_avgs[i])
            print("\t model_roc_auc_score_avgs[i] == ", model_roc_auc_score_avgs[i])
            print("\t model_cos_sim_score_std_devs[i] == ", model_cos_sim_score_std_devs[i])
            print("\t model_mse_std_devs[i] == ", model_mse_std_devs[i])
            print("\t model_dir_deltacon0_score_std_devs[i] == ", model_dir_deltacon0_score_std_devs[i])
            print("\t model_undir_deltacon0_score_std_devs[i] == ", model_undir_deltacon0_score_std_devs[i])
            print("\t model_deltacon0_wDD_score_std_devs[i] == ", model_deltacon0_wDD_score_std_devs[i])
            print("\t model_deltaffinity_score_std_devs[i] == ", model_deltaffinity_score_std_devs[i])
            print("\t model_roc_auc_score_std_devs[i] == ", model_roc_auc_score_std_devs[i])
            print("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
        if i in models_with_max_dir_deltacon0_score:
            print("<<< BEST MODEL BY DIRECTED DELTACON0 SIMILARITY: MODEL i==", i, " >>> d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^")
            print("\t model_folder_name == ", model_folder_name)
            print("\t model_cos_sim_score_avgs[i] == ", model_cos_sim_score_avgs[i])
            print("\t model_mse_avgs[i] == ", model_mse_avgs[i])
            print("\t max_dir_deltacon0_score == ", max_dir_deltacon0_score)
            print("\t model_undir_deltacon0_score_avgs[i] == ", model_undir_deltacon0_score_avgs[i])
            print("\t model_deltacon0_wDD_score_avgs[i] == ", model_deltacon0_wDD_score_avgs[i])
            print("\t model_deltaffinity_score_avgs[i] == ", model_deltaffinity_score_avgs[i])
            print("\t model_roc_auc_score_avgs[i] == ", model_roc_auc_score_avgs[i])
            print("\t model_cos_sim_score_std_devs[i] == ", model_cos_sim_score_std_devs[i])
            print("\t model_mse_std_devs[i] == ", model_mse_std_devs[i])
            print("\t model_dir_deltacon0_score_std_devs[i] == ", model_dir_deltacon0_score_std_devs[i])
            print("\t model_undir_deltacon0_score_std_devs[i] == ", model_undir_deltacon0_score_std_devs[i])
            print("\t model_deltacon0_wDD_score_std_devs[i] == ", model_deltacon0_wDD_score_std_devs[i])
            print("\t model_deltaffinity_score_std_devs[i] == ", model_deltaffinity_score_std_devs[i])
            print("\t model_roc_auc_score_std_devs[i] == ", model_roc_auc_score_std_devs[i])
            print("d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^d^")
        if i in models_with_max_undir_deltacon0_score:
            print("<<< BEST MODEL BY UNDIRECTED DELTACON0 SIMILARITY: MODEL i==", i, " >>> u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^")
            print("\t model_folder_name == ", model_folder_name)
            print("\t model_cos_sim_score_avgs[i] == ", model_cos_sim_score_avgs[i])
            print("\t model_mse_avgs[i] == ", model_mse_avgs[i])
            print("\t model_dir_deltacon0_score_avgs[i] == ", model_dir_deltacon0_score_avgs[i])
            print("\t max_undir_deltacon0_score == ", max_undir_deltacon0_score)
            print("\t model_deltacon0_wDD_score_avgs[i] == ", model_deltacon0_wDD_score_avgs[i])
            print("\t model_deltaffinity_score_avgs[i] == ", model_deltaffinity_score_avgs[i])
            print("\t model_roc_auc_score_avgs[i] == ", model_roc_auc_score_avgs[i])
            print("\t model_cos_sim_score_std_devs[i] == ", model_cos_sim_score_std_devs[i])
            print("\t model_mse_std_devs[i] == ", model_mse_std_devs[i])
            print("\t model_dir_deltacon0_score_std_devs[i] == ", model_dir_deltacon0_score_std_devs[i])
            print("\t model_undir_deltacon0_score_std_devs[i] == ", model_undir_deltacon0_score_std_devs[i])
            print("\t model_deltacon0_wDD_score_std_devs[i] == ", model_deltacon0_wDD_score_std_devs[i])
            print("\t model_deltaffinity_score_std_devs[i] == ", model_deltaffinity_score_std_devs[i])
            print("\t model_roc_auc_score_std_devs[i] == ", model_roc_auc_score_std_devs[i])
            print("u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^u^")
        if i in models_with_max_deltacon0_wDD_score:
            print("<<< BEST MODEL BY DELTACON0 withDirectedDegrees SIMILARITY: MODEL i==", i, " >>> ^w^w^w^w^w^w^w^w^w^w^w^w^w^w")
            print("\t model_folder_name == ", model_folder_name)
            print("\t model_cos_sim_score_avgs[i] == ", model_cos_sim_score_avgs[i])
            print("\t model_mse_avgs[i] == ", model_mse_avgs[i])
            print("\t model_dir_deltacon0_score_avgs[i] == ", model_dir_deltacon0_score_avgs[i])
            print("\t model_undir_deltacon0_score_avgs[i] == ", model_undir_deltacon0_score_avgs[i])
            print("\t max_deltacon0_wDD_score == ", max_deltacon0_wDD_score)
            print("\t model_deltaffinity_score_avgs[i] == ", model_deltaffinity_score_avgs[i])
            print("\t model_roc_auc_score_avgs[i] == ", model_roc_auc_score_avgs[i])
            print("\t model_cos_sim_score_std_devs[i] == ", model_cos_sim_score_std_devs[i])
            print("\t model_mse_std_devs[i] == ", model_mse_std_devs[i])
            print("\t model_dir_deltacon0_score_std_devs[i] == ", model_dir_deltacon0_score_std_devs[i])
            print("\t model_undir_deltacon0_score_std_devs[i] == ", model_undir_deltacon0_score_std_devs[i])
            print("\t model_deltacon0_wDD_score_std_devs[i] == ", model_deltacon0_wDD_score_std_devs[i])
            print("\t model_deltaffinity_score_std_devs[i] == ", model_deltaffinity_score_std_devs[i])
            print("\t model_roc_auc_score_std_devs[i] == ", model_roc_auc_score_std_devs[i])
            print("^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w^w")
        if i in models_with_max_deltaffinity_score:
            print("<<< BEST MODEL BY DELTAFFINITY SIMILARITY: MODEL i==", i, " >>> ^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f")
            print("\t model_folder_name == ", model_folder_name)
            print("\t model_cos_sim_score_avgs[i] == ", model_cos_sim_score_avgs[i])
            print("\t model_mse_avgs[i] == ", model_mse_avgs[i])
            print("\t model_dir_deltacon0_score_avgs[i] == ", model_dir_deltacon0_score_avgs[i])
            print("\t model_undir_deltacon0_score_avgs[i] == ", model_undir_deltacon0_score_avgs[i])
            print("\t model_deltacon0_wDD_score_avgs[i] == ", model_deltacon0_wDD_score_avgs[i])
            print("\t max_deltaffinity_score == ", max_deltaffinity_score)
            print("\t model_roc_auc_score_avgs[i] == ", model_roc_auc_score_avgs[i])
            print("\t model_cos_sim_score_std_devs[i] == ", model_cos_sim_score_std_devs[i])
            print("\t model_mse_std_devs[i] == ", model_mse_std_devs[i])
            print("\t model_dir_deltacon0_score_std_devs[i] == ", model_dir_deltacon0_score_std_devs[i])
            print("\t model_undir_deltacon0_score_std_devs[i] == ", model_undir_deltacon0_score_std_devs[i])
            print("\t model_deltacon0_wDD_score_std_devs[i] == ", model_deltacon0_wDD_score_std_devs[i])
            print("\t model_deltaffinity_score_std_devs[i] == ", model_deltaffinity_score_std_devs[i])
            print("\t model_roc_auc_score_std_devs[i] == ", model_roc_auc_score_std_devs[i])
            print("^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f^f")
        if i in models_with_max_roc_auc_score:
            print("<<< BEST MODEL BY ROC-AUC SCORE: MODEL i==", i, " >>> -----------------------------------------------------")
            print("\t model_folder_name == ", model_folder_name)
            print("\t model_cos_sim_score_avgs[i] == ", model_cos_sim_score_avgs[i])
            print("\t model_mse_avgs[i] == ", model_mse_avgs[i])
            print("\t model_dir_deltacon0_score_avgs[i] == ", model_dir_deltacon0_score_avgs[i])
            print("\t model_undir_deltacon0_score_avgs[i] == ", model_undir_deltacon0_score_avgs[i])
            print("\t model_deltacon0_wDD_score_avgs[i] == ", model_deltacon0_wDD_score_avgs[i])
            print("\t model_deltaffinity_score_avgs[i] == ", model_deltaffinity_score_avgs[i])
            print("\t max_roc_auc_score == ", max_roc_auc_score)
            print("\t model_cos_sim_score_std_devs[i] == ", model_cos_sim_score_std_devs[i])
            print("\t model_mse_std_devs[i] == ", model_mse_std_devs[i])
            print("\t model_dir_deltacon0_score_std_devs[i] == ", model_dir_deltacon0_score_std_devs[i])
            print("\t model_undir_deltacon0_score_std_devs[i] == ", model_undir_deltacon0_score_std_devs[i])
            print("\t model_deltacon0_wDD_score_std_devs[i] == ", model_deltacon0_wDD_score_std_devs[i])
            print("\t model_deltaffinity_score_std_devs[i] == ", model_deltaffinity_score_std_devs[i])
            print("\t model_roc_auc_score_std_devs[i] == ", model_roc_auc_score_std_devs[i])
            print("----------------------------------------------------------------------------------------------------")
        print("\n\n", flush=True)
    pass
