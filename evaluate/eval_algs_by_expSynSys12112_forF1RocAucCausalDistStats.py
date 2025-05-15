import sys
print("IMPORTS: sys.path == ", sys.path, flush=True)

import os
import json
import copy
import argparse
import numpy as np
import pickle as pkl

import gadjid # see https://github.com/CausalDisco/gadjid
from gadjid import ancestor_aid, oset_aid, parent_aid, shd # see https://github.com/CausalDisco/gadjid
from general_utils.metrics import compute_optimal_f1, roc_auc_score

from general_utils.input_argument_utils import read_in_data_args
from evaluate.eval_utils import load_model_for_eval, get_model_gc_estimates
from general_utils.misc import mask_diag_elements_of_square_numpy_array, normalize_numpy_array
from general_utils.plotting import plot_gc_est_comparisson, make_scatter_and_stdErrOfMean_plot_overlay_vis

from tidybench import slarac, qrbs, lasar

# PCMCI Imports
import matplotlib
from matplotlib import pyplot as plt
# plt.style.use('ggplot')
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve

import tigramite
from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.rpcmci import RPCMCI

from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.cmisymb import CMIsymb



def prepare_data_for_rpcmci_modeling(orig_data):
    num_samps = len(orig_data)
    data = None
    labels = None
    T_window_size = None
    N = None
    num_regimes = None
    masks_by_regime_index = None
    for i, samp in enumerate(orig_data):
        x = samp[0]
        y = samp[3]
        if data is None:
            assert labels is None
            T_window_size = x.shape[0]
            N = x.shape[1]
            data = x
            assert data.shape == (T_window_size, N)
            num_regimes = y.shape[0]
            labels = y.T
            assert labels.shape == (T_window_size, num_regimes)
            masks_by_regime_index = {r:np.zeros(data.shape) for r in range(num_regimes)}
        else:
            data = np.concatenate((data, x), axis=0)
            labels = np.concatenate([labels]+[y.T], axis=0)
            for r in masks_by_regime_index.keys():
                masks_by_regime_index[r] = np.concatenate([masks_by_regime_index[r], np.zeros(x.shape)], axis=0)
        for r in masks_by_regime_index.keys():
            assert masks_by_regime_index[r].shape == data.shape
        for j in range(T_window_size):
            curr_dom_regime = np.argmax(y[:,j])
            masks_by_regime_index[curr_dom_regime][(-1*T_window_size)+j,:] = masks_by_regime_index[curr_dom_regime][(-1*T_window_size)+j,:] + 1
    T = T_window_size*num_samps
    assert data.shape == (T, N)
    assert labels.shape == (T, num_regimes)
    assert np.max([np.max(masks_by_regime_index[r]) for r in range(num_regimes)]) == 1
    assert np.min([np.min(masks_by_regime_index[r]) for r in range(num_regimes)]) == 0
    return data, labels, masks_by_regime_index, T_window_size, T, N, num_regimes

def get_standardized_off_diagonal_relation_predictions_for_rpcmci(A_tensor, transpose=False):
    assert len(A_tensor.shape) == 3
    assert A_tensor.shape[0] == A_tensor.shape[1]
    standard_A = np.sum(np.abs(A_tensor), axis=2)
    if transpose: # standard convention is that columns drive rows
        standard_A = standard_A.T
    off_diag_mask = np.ones(standard_A.shape) - np.eye(standard_A.shape[0])
    off_diag_standard_A = standard_A*off_diag_mask
    return off_diag_standard_A


def get_pcmci_edge_preds_from_graph(graph):
    assert len(graph.shape) == 3
    assert graph.shape[0] == graph.shape[1]
    edge_pred_tensor = np.zeros(graph.shape)
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            for k in range(graph.shape[2]):
                if graph[i,j,k] == "-->":
                    edge_pred_tensor[i,j,k] = 1
    return edge_pred_tensor


def run_regime_aware_PCMCI(orig_train_data, pred_source="val_matrix", transpose=True):
    # load data and initialize necessary variables
    train_data, _, masks_by_regime_index, _, T, N, num_regimes = prepare_data_for_rpcmci_modeling(orig_train_data)
    var_names = ["c"+str(i) for i in range(N)]
    datatime = np.arange(T)

    dataframe_plotting_by_regime = dict()
    for r in range(num_regimes):
        dataframe_plotting_by_regime[r] = pp.DataFrame(train_data, mask=masks_by_regime_index[r])
        # tp.plot_timeseries(dataframe_plotting_by_regime[r], figsize=(N,N), grey_masked_samples='data')
        # plt.xlabel("Regime "+str(r)+" Data (Grey)")
        # plt.show()

    # Case where causal regimes are known
    pcmci_by_regime = {r:PCMCI(dataframe=dataframe_plotting_by_regime[r], cond_ind_test=ParCorr(mask_type='y')) for r in range(num_regimes)}
    pcmci_results_by_regime = {r: pcmci_by_regime[r].run_pcmci(tau_min=1, tau_max=2, pc_alpha=0.2, alpha_level=0.01) for r in range(num_regimes)}
    # for r in pcmci_results_by_regime.keys():
    #     tp.plot_graph(
    #         val_matrix=pcmci_results_by_regime[r]['val_matrix'],
    #         graph=pcmci_results_by_regime[r]['graph'],
    #         var_names=var_names,
    #         node_aspect=0.5, node_size=0.5
    #     )
    #     plt.title("PCMCI Results for Regime "+str(r))
    #     plt.show()

    pcmci_edge_preds_by_regime = None
    if pred_source == "graph":
        pcmci_edge_preds_by_regime = {r:get_pcmci_edge_preds_from_graph(pcmci_results_by_regime[r]['graph']) for r in range(num_regimes)}
    elif pred_source == "val_matrix":
        pcmci_edge_preds_by_regime = {r:pcmci_results_by_regime[r]['val_matrix'] for r in range(num_regimes)}
    else:
        raise ValueError()
    preds = [get_standardized_off_diagonal_relation_predictions_for_rpcmci(pcmci_edge_preds_by_regime[r], transpose=transpose) for r in range(num_regimes)]
    # for r in pcmci_standardizedRelationPreds_by_regime.keys():
    #     plt.imshow(pcmci_standardizedRelationPreds_by_regime[r])
    #     plt.colorbar()
    #     plt.title("PCMCI Standardized Inter-Variable Relation Preds for Regime "+str(r))
    #     plt.show()
    return preds


def prepare_data_for_modeling(orig_data):
    num_samps = len(orig_data)
    data = None
    labels = None
    T_window_size = None
    N = None
    num_regimes = None
    masks_by_regime_index = None
    for i, samp in enumerate(orig_data):
        x = samp[0]
        y = samp[3]
        if data is None:
            assert labels is None
            T_window_size = x.shape[0]
            N = x.shape[1]
            data = x
            assert data.shape == (T_window_size, N)
            num_regimes = y.shape[0]
            labels = y.T
            assert labels.shape == (T_window_size, num_regimes)
            masks_by_regime_index = {r:np.zeros(data.shape) for r in range(num_regimes)}
        else:
            data = np.concatenate((data, x), axis=0)
            labels = np.concatenate([labels]+[y.T], axis=0)
            for r in masks_by_regime_index.keys():
                masks_by_regime_index[r] = np.concatenate([masks_by_regime_index[r], np.zeros(x.shape)], axis=0)
        for r in masks_by_regime_index.keys():
            assert masks_by_regime_index[r].shape == data.shape
        for j in range(T_window_size):
            curr_dom_regime = np.argmax(y[:,j])
            masks_by_regime_index[curr_dom_regime][(-1*T_window_size)+j,:] = masks_by_regime_index[curr_dom_regime][(-1*T_window_size)+j,:] + 1
    T = T_window_size*num_samps
    assert data.shape == (T, N)
    assert labels.shape == (T, num_regimes)
    assert np.max([np.max(masks_by_regime_index[r]) for r in range(num_regimes)]) == 1
    assert np.min([np.min(masks_by_regime_index[r]) for r in range(num_regimes)]) == 0
    return data, labels, masks_by_regime_index, T_window_size, T, N, num_regimes

def get_standardized_off_diagonal_relation_predictions(A, transpose=False):
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    standard_A = np.abs(A)
    if transpose:# standard convention is that columns drive rows
        standard_A = standard_A.T
    off_diag_mask = np.ones(standard_A.shape) - np.eye(standard_A.shape[0])
    off_diag_standard_A = standard_A*off_diag_mask
    return off_diag_standard_A



def run_tidybench_experiment(orig_train_data, alg_name):
    # load data and initialize necessary variables
    print("run_tidybench_experiment: loading train data", flush=True)
    train_data, _, masks_by_regime_index, _, _, _, num_regimes = prepare_data_for_modeling(orig_train_data)
    # Case where causal regimes are known
    print("run_tidybench_experiment: obtaining preds", flush=True)
    preds = None
    if alg_name =="slarac":
        slarac_edge_preds = [slarac(train_data*masks_by_regime_index[r],post_standardise=True) for r in range(num_regimes)]
        preds = [get_standardized_off_diagonal_relation_predictions(slarac_edge_preds[r], transpose=False) for r in range(num_regimes)]
    elif alg_name =="qrbs":
        qrbs_edge_preds = [qrbs(train_data*masks_by_regime_index[r],post_standardise=True) for r in range(num_regimes)]
        preds = [get_standardized_off_diagonal_relation_predictions(qrbs_edge_preds[r], transpose=False) for r in range(num_regimes)]
    elif alg_name == "lasar":
        lasar_edge_preds = [lasar(train_data*masks_by_regime_index[r],post_standardise=True) for r in range(num_regimes)]
        preds = [get_standardized_off_diagonal_relation_predictions(lasar_edge_preds[r], transpose=False) for r in range(num_regimes)]
    else:
        raise ValueError()
    return preds


if __name__=="__main__":
    parse=argparse.ArgumentParser(description='Default parameter system-level evaluation')
    parse.add_argument(
        "-cached_args_file",
        default="eval_algs_by_expSynSys12112_forF1RocAucCausalDistStats_cached_args.txt", # WARNING: edit this line when running new experiments on SLURM server
        help="/path/to/cached_args.txt should contain a json object with updated arg values",
    )
    args = parse.parse_args()
    TRANSPOSE_PREDICTIONS_DURING_EVAL = False
    PCMCI_PRED_SOURCE = "val_matrix"
    
    print("__MAIN__: LOADING ARGS", flush=True)
    main_args_dict = None
    with open(args.cached_args_file, 'r') as infile:
        main_args_dict = json.load(infile)
        print("__MAIN__: main_args_dict.keys() == ", main_args_dict.keys())
    
    experiments_to_evaluate = [x for x in main_args_dict.keys() if "experiment_name" in x]
    for i,exp_name_key in enumerate(experiments_to_evaluate):
        print("__MAIN__: i==", i, " -- exp_name_key == ", exp_name_key, flush=True)
        exp_name = main_args_dict[exp_name_key]
        curr_exp_save_path = main_args_dict["save_root_path"]+os.sep+exp_name
        if not os.path.exists(curr_exp_save_path):
            os.mkdir(curr_exp_save_path)
        
        # read in details of dataset related to experiment
        curr_exp_data_name = main_args_dict[exp_name+"_data_name"]
        curr_exp_data_args_path = main_args_dict[exp_name+"_cached_data_args_path"]
        data_args_dict = {
            "model_type": "REDCLIFF_S_CMLP", 
            "save_root_path": curr_exp_save_path, 
            "data_set_name": curr_exp_data_name, 
            "data_cached_args_file": curr_exp_data_args_path, 
            "true_GC_factors": None, 
        }
        data_args_dict = read_in_data_args(data_args_dict, include_gc_views_for_eval=False, read_in_gc_factors_for_eval=True)
        if len(data_args_dict["true_GC_factors"][0].shape) == 3:
            assert data_args_dict["true_GC_factors"][0].shape[0] == data_args_dict["true_GC_factors"][0].shape[1] # sanity check
            data_args_dict["true_GC_factors"] = [np.sum(x, axis=2) for x in data_args_dict["true_GC_factors"]]
            data_args_dict["true_GC_factors"] = [x>0 for x in data_args_dict["true_GC_factors"]]
            data_args_dict["true_GC_factors"] = [x.astype(int) for x in data_args_dict["true_GC_factors"]]
            data_args_dict["true_GC_factors"] = [mask_diag_elements_of_square_numpy_array(x) for x in data_args_dict["true_GC_factors"]]
        assert len(data_args_dict["true_GC_factors"][0].shape) == 2 # sanity check
        assert np.max(data_args_dict["true_GC_factors"]) == 1
        assert np.min(data_args_dict["true_GC_factors"]) == 0
        assert np.median(data_args_dict["true_GC_factors"]) in [0,1]

        print("__MAIN__: \t data_args_dict == ", data_args_dict)
        curr_exp_aggr_data = None
        if exp_name+"_aggregated_data_path" in main_args_dict.keys():
            print("__MAIN__: \t loading aggregated data from aggregated_data_path == ", main_args_dict[exp_name+"_aggregated_data_path"])
            curr_exp_aggr_data = pkl.load(open(main_args_dict[exp_name+"_aggregated_data_path"],'rb'))
        
        # evaluate each algorithm
        curr_exp_alg_name_keys = [x for x in main_args_dict.keys() if exp_name+"_model_name_" in x]
        preds_by_alg_key_dict = dict()
        stats_by_alg_key_dict = dict()
        for alg_name_key in curr_exp_alg_name_keys:
            print("__MAIN__: \t evaluating alg_name_key == ", alg_name_key, flush=True)
            curr_alg_name = main_args_dict[alg_name_key]
            
            # obtain factor granger causal preds from alg
            preds = None
            if curr_alg_name in ["PCMCI", "slarac", "qrbs", "lasar"]:
                if main_args_dict[exp_name+"_"+curr_alg_name+"_path"] != "None":
                    preds = pkl.load(open(main_args_dict[exp_name+"_"+curr_alg_name+"_path"], 'rb'))
                else:
                    assert curr_exp_aggr_data is not None
                    if curr_alg_name == "PCMCI":
                        preds = run_regime_aware_PCMCI(curr_exp_aggr_data, pred_source=PCMCI_PRED_SOURCE, transpose=False)
                    else:
                        preds = run_tidybench_experiment(curr_exp_aggr_data, curr_alg_name)
                    with open(curr_exp_save_path+os.sep+exp_name+"_"+curr_alg_name+"_preds.pkl", 'wb') as outfile:
                        pkl.dump(preds, outfile)
            else:
                trained_alg_path = main_args_dict[exp_name+"_"+curr_alg_name+"_path"]
                trained_model = load_model_for_eval(curr_alg_name, trained_alg_path, dynamic_eval=True, d4IC=False)
                if "REDCLIFF" in curr_alg_name and "conditional" in trained_model.primary_gc_est_mode: # this case isn't handled on next line, where X=None
                    print("__MAIN__: \t\t WARNING!! OVERWRITING trained_model.primary_gc_est_mode TO fixed_factor_exclusive FOR SYS-LEVEL INTERPRETATION")
                    trained_model.primary_gc_est_mode = "fixed_factor_exclusive"
                preds = get_model_gc_estimates(trained_model, curr_alg_name, len(data_args_dict["true_GC_factors"]), X=None)
            
            # standardize the preds
            if len(preds[0].shape) == 3:
                assert preds[0].shape[0] == preds[0].shape[1]
                preds = [np.sum(x, axis=2) for x in preds]
            no_diag_preds = [mask_diag_elements_of_square_numpy_array(x) for x in preds] # remove diagonal from preds
            normalized_no_diag_preds = [normalize_numpy_array(x) for x in no_diag_preds] # normalize preds
            preds_by_alg_key_dict[alg_name_key] = normalized_no_diag_preds
        
            # evaluate preds
            curr_alg_stats_by_regime_factor = {
                "rf_"+str(f): {
                    "optF1_thresh": None, 
                    "optF1_score": None, 
                    "roc_auc": None, "optF1Thresh_roc_auc": None, 
                    "optF1Thresh_ancestor_aid": None, "upper_optF1Thresh_ancestor_aid": None, "lower_optF1Thresh_ancestor_aid": None,
                    "optF1Thresh_oset_aid": None, "upper_optF1Thresh_oset_aid": None, "lower_optF1Thresh_oset_aid": None, 
                    "optF1Thresh_parent_aid": None, "upper_optF1Thresh_parent_aid": None, "lower_optF1Thresh_parent_aid": None, 
                    "optF1Thresh_shd": None, "upper_optF1Thresh_shd": None, "lower_optF1Thresh_shd": None, 
                } for f in range(len(data_args_dict["true_GC_factors"]))
            }
            for rf in range(len(data_args_dict["true_GC_factors"])):
                true_graph = data_args_dict["true_GC_factors"][rf].astype(np.int8)
                true_graph_labels = true_graph.flatten().astype(int)
                rf_pred = preds_by_alg_key_dict[alg_name_key][rf]
                if TRANSPOSE_PREDICTIONS_DURING_EVAL:
                    rf_pred = rf_pred.T
                # opt f1 score and threshold
                curr_alg_stats_by_regime_factor["rf_"+str(rf)]["optF1_thresh"], \
                curr_alg_stats_by_regime_factor["rf_"+str(rf)]["optF1_score"] = compute_optimal_f1(true_graph_labels, rf_pred.flatten())
                rf_pred_opt1Thresh_mask = rf_pred > curr_alg_stats_by_regime_factor["rf_"+str(rf)]["optF1_thresh"]
                rf_pred_opt1Thresh_mask = rf_pred_opt1Thresh_mask*(np.ones(rf_pred_opt1Thresh_mask.shape)-np.eye(rf_pred_opt1Thresh_mask.shape[0]))
                rf_pred_opt1Thresh_mask = rf_pred_opt1Thresh_mask.astype(np.int8)
                upper_true_graph = np.triu(true_graph)
                upper_rf_pred_opt1Thresh_mask = np.triu(rf_pred_opt1Thresh_mask)
                lower_true_graph = np.tril(true_graph)
                lower_rf_pred_opt1Thresh_mask = np.tril(rf_pred_opt1Thresh_mask)

                curr_alg_stats_by_regime_factor["rf_"+str(rf)]["roc_auc"] = roc_auc_score(true_graph_labels, rf_pred.flatten())
                curr_alg_stats_by_regime_factor["rf_"+str(rf)]["optF1Thresh_roc_auc"] = roc_auc_score(true_graph_labels, rf_pred_opt1Thresh_mask.flatten())
                try:
                    curr_alg_stats_by_regime_factor["rf_"+str(rf)]["optF1Thresh_ancestor_aid"] = ancestor_aid(
                        true_graph, rf_pred_opt1Thresh_mask, edge_direction="from column to row"
                    )
                except:
                    print("WARNING - GRAPH AND/OR ESTIMATE WAS INCOMPATIBLE WITH ancestor_aid METRIC")
                    curr_alg_stats_by_regime_factor["rf_"+str(rf)]["optF1Thresh_ancestor_aid"] = np.nan
                try:
                    curr_alg_stats_by_regime_factor["rf_"+str(rf)]["upper_optF1Thresh_ancestor_aid"] = ancestor_aid(
                        upper_true_graph, upper_rf_pred_opt1Thresh_mask, edge_direction="from column to row"
                    )
                except:
                    print("WARNING - UPPER GRAPH AND/OR ESTIMATE WAS INCOMPATIBLE WITH ancestor_aid METRIC")
                    curr_alg_stats_by_regime_factor["rf_"+str(rf)]["upper_optF1Thresh_ancestor_aid"] = np.nan
                try:
                    curr_alg_stats_by_regime_factor["rf_"+str(rf)]["lower_optF1Thresh_ancestor_aid"] = ancestor_aid(
                        lower_true_graph, lower_rf_pred_opt1Thresh_mask, edge_direction="from column to row"
                    )
                except:
                    print("WARNING - LOWER GRAPH AND/OR ESTIMATE WAS INCOMPATIBLE WITH ancestor_aid METRIC")
                    curr_alg_stats_by_regime_factor["rf_"+str(rf)]["lower_optF1Thresh_ancestor_aid"] = np.nan

                try:
                    curr_alg_stats_by_regime_factor["rf_"+str(rf)]["optF1Thresh_oset_aid"] = oset_aid(
                        true_graph, rf_pred_opt1Thresh_mask, edge_direction="from column to row"
                    )
                except:
                    print("WARNING - GRAPH AND/OR ESTIMATE WAS INCOMPATIBLE WITH oset_aid METRIC")
                    curr_alg_stats_by_regime_factor["rf_"+str(rf)]["optF1Thresh_oset_aid"] = np.nan
                try:
                    curr_alg_stats_by_regime_factor["rf_"+str(rf)]["upper_optF1Thresh_oset_aid"] = oset_aid(
                        upper_true_graph, upper_rf_pred_opt1Thresh_mask, edge_direction="from column to row"
                    )
                except:
                    print("WARNING - UPPER GRAPH AND/OR ESTIMATE WAS INCOMPATIBLE WITH oset_aid METRIC")
                    curr_alg_stats_by_regime_factor["rf_"+str(rf)]["upper_optF1Thresh_oset_aid"] = np.nan
                try:
                    curr_alg_stats_by_regime_factor["rf_"+str(rf)]["lower_optF1Thresh_oset_aid"] = oset_aid(
                        lower_true_graph, lower_rf_pred_opt1Thresh_mask, edge_direction="from column to row"
                    )
                except:
                    print("WARNING - LOWER GRAPH AND/OR ESTIMATE WAS INCOMPATIBLE WITH oset_aid METRIC")
                    curr_alg_stats_by_regime_factor["rf_"+str(rf)]["lower_optF1Thresh_oset_aid"] = np.nan

                try:
                    curr_alg_stats_by_regime_factor["rf_"+str(rf)]["optF1Thresh_parent_aid"] = parent_aid(
                        true_graph, rf_pred_opt1Thresh_mask, edge_direction="from column to row"
                    )
                except:
                    print("WARNING - GRAPH AND/OR ESTIMATE WAS INCOMPATIBLE WITH parent_aid METRIC")
                    curr_alg_stats_by_regime_factor["rf_"+str(rf)]["optF1Thresh_parent_aid"] = np.nan
                try:
                    curr_alg_stats_by_regime_factor["rf_"+str(rf)]["upper_optF1Thresh_parent_aid"] = parent_aid(
                        upper_true_graph, upper_rf_pred_opt1Thresh_mask, edge_direction="from column to row"
                    )
                except:
                    print("WARNING - UPPER GRAPH AND/OR ESTIMATE WAS INCOMPATIBLE WITH parent_aid METRIC")
                    curr_alg_stats_by_regime_factor["rf_"+str(rf)]["upper_optF1Thresh_parent_aid"] = np.nan
                try:
                    curr_alg_stats_by_regime_factor["rf_"+str(rf)]["lower_optF1Thresh_parent_aid"] = parent_aid(
                        lower_true_graph, lower_rf_pred_opt1Thresh_mask, edge_direction="from column to row"
                    )
                except:
                    print("WARNING - LOWER GRAPH AND/OR ESTIMATE WAS INCOMPATIBLE WITH parent_aid METRIC")
                    curr_alg_stats_by_regime_factor["rf_"+str(rf)]["lower_optF1Thresh_parent_aid"] = np.nan

                try:
                    curr_alg_stats_by_regime_factor["rf_"+str(rf)]["optF1Thresh_shd"] = shd(true_graph, rf_pred_opt1Thresh_mask)
                except:
                    print("WARNING - GRAPH AND/OR ESTIMATE WAS INCOMPATIBLE WITH shd METRIC")
                    curr_alg_stats_by_regime_factor["rf_"+str(rf)]["optF1Thresh_shd"] = np.nan
                try:
                    curr_alg_stats_by_regime_factor["rf_"+str(rf)]["upper_optF1Thresh_shd"] = shd(upper_true_graph, upper_rf_pred_opt1Thresh_mask)
                except:
                    print("WARNING - UPPER GRAPH AND/OR ESTIMATE WAS INCOMPATIBLE WITH shd METRIC")
                    curr_alg_stats_by_regime_factor["rf_"+str(rf)]["upper_optF1Thresh_shd"] = np.nan
                try:
                    curr_alg_stats_by_regime_factor["rf_"+str(rf)]["lower_optF1Thresh_shd"] = shd(lower_true_graph, lower_rf_pred_opt1Thresh_mask)
                except:
                    print("WARNING - LOWER GRAPH AND/OR ESTIMATE WAS INCOMPATIBLE WITH shd METRIC")
                    curr_alg_stats_by_regime_factor["rf_"+str(rf)]["lower_optF1Thresh_shd"] = np.nan
            
            stats_by_alg_key_dict[alg_name_key] = curr_alg_stats_by_regime_factor
        
        # report (and save?) results for current exp
        print("__MAIN__: \t stats_by_alg_key_dict == ", stats_by_alg_key_dict, flush=True)
        with open(curr_exp_save_path+os.sep+"stats_by_alg_key_dict.pkl", 'wb') as outfile:
            pkl.dump(stats_by_alg_key_dict, outfile)
        print("__MAIN__: \t now saving preds_by_alg_key_dict ", flush=True)
        with open(curr_exp_save_path+os.sep+"preds_by_alg_key_dict.pkl", 'wb') as outfile:
            pkl.dump(preds_by_alg_key_dict, outfile)
        print("__MAIN__: WARNING!! CROSS-EXP STATS-STITCHING IS NOT IMPLEMENTED")
    
    print("__MAIN__: DONE !!!")
    pass
