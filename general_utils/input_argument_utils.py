import torch
import numpy as np
import json
import os
import copy

from general_utils.plotting import plot_gc_est_comparisson, plot_gc_est_comparissons_by_factor


def parse_input_list_of_ints(list_string):
    if list_string == "[]":
        return []
    print("input list_string == ", list_string)
    items = list_string[1:-1]
    items = items.split(",")
    print("items == ", items)
    items = [int(chars) for chars in items]
    return items


def parse_input_list_of_strs(list_string):
    if list_string == "[]":
        return []
    print("input list_string == ", list_string)
    items = list_string[1:-1]
    items = items.split(",")
    print("items == ", items)
    items = [string for string in items]
    return items


def parse_tensor_string_representation(tensor_string):
    tensor_slices = []
    if ",],],]" in tensor_string: # (assumed) case where gamma has only one element
        print("parse_tensor_string_representation: WARNING - ASSUMING THERE IS ONLY ONE ELEMENT IN tensor_string == ", tensor_string, flush=True)
        tensor_slices = [[[float(tensor_string[3:-6])]]]
    else:
        tensor_slices = tensor_string[3:-3].split("]], [[")
        for i, matrix_slice in enumerate(tensor_slices):
            matrix_slice = matrix_slice.split("], [")
            matrix_slice = [[float(x) for x in row.split(",")] for row in matrix_slice]
            tensor_slices[i] = matrix_slice
    tensor = np.array(tensor_slices)
    assert len(tensor.shape) == 3 # ensure returned matrix is 2-dimensional
    if tensor.shape[1] == tensor.shape[2]:
        tensor = np.transpose(tensor, axes=[1,2,0]) # see https://stackoverflow.com/questions/57438392/rearranging-axes-in-numpy
    assert tensor.shape[0] == tensor.shape[1]
    return tensor


def read_in_data_adjacency_matrices(args_dict, cached_args_file_path):
    with open(cached_args_file_path, 'r') as infile:
        data_args_dict = json.load(infile)
        args_dict["true_lagged_GC_tensor"] = None
        args_dict["true_nontemporal_GC_tensor"] = None
        args_dict["true_lagged_GC_tensor_factors"] = [None, None, None, None]
        args_dict["true_nontemporal_GC_tensor_factors"] = [None, None, None, None]
        for key in data_args_dict.keys():
            if "adjacency_tensor" in key:
                curr_lagged_tensor = parse_tensor_string_representation(data_args_dict[key])
                curr_lagged_tensor = curr_lagged_tensor[:,:,::-1].copy() # lagged tensors have been saved in reverse-lag order, which we correct here for later comparissons with gc estimates
                curr_nonTemporal_matrix = np.sum(curr_lagged_tensor, axis=2)
                
                curr_factor_ind = int(key[3])-1 # convention that all tensor keys begin with "net" followed by an integer and "_"
                args_dict["true_lagged_GC_tensor_factors"][curr_factor_ind] = torch.from_numpy(curr_lagged_tensor).cpu().data.numpy()
                args_dict["true_nontemporal_GC_tensor_factors"][curr_factor_ind] = torch.from_numpy(curr_nonTemporal_matrix).cpu().data.numpy()

                if args_dict["true_lagged_GC_tensor"] is None:
                    args_dict["true_lagged_GC_tensor"] = curr_lagged_tensor
                else:
                    args_dict["true_lagged_GC_tensor"] = args_dict["true_lagged_GC_tensor"] + curr_lagged_tensor

                if args_dict["true_nontemporal_GC_tensor"] is None:
                    args_dict["true_nontemporal_GC_tensor"] = curr_nonTemporal_matrix
                else:
                    args_dict["true_nontemporal_GC_tensor"] = args_dict["true_nontemporal_GC_tensor"] + curr_nonTemporal_matrix
                
                curr_lagged_GC = [torch.from_numpy(curr_lagged_tensor).cpu().data.numpy()]
                plot_gc_est_comparissons_by_factor(
                    curr_lagged_GC, 
                    None, 
                    args_dict['save_root_path']+os.sep+key+"_visualization_WITH_LAGS.png", 
                    include_lags=True
                )
                curr_nonTemporal_GC = [torch.from_numpy(curr_nonTemporal_matrix).cpu().data.numpy()]
                plot_gc_est_comparissons_by_factor(
                    curr_nonTemporal_GC, 
                    None, 
                    args_dict['save_root_path']+os.sep+key+"_visualization_WITHOUT_LAGS.png", 
                    include_lags=False
                )
    return args_dict


def read_in_model_args(args_dict):

    print("read_in_model_args: READING IN ARGS FOR args_dict['model_type'] == ", args_dict["model_type"])
    print("read_in_model_args: READING IN ARGS FROM args_dict['model_cached_args_file'] == ", args_dict["model_cached_args_file"])
    
    with open(args_dict["model_cached_args_file"], 'r') as infile:
        new_args_dict = json.load(infile)

        print("read_in_model_args: new_args_dict == ", new_args_dict)

        if "cMLP" in args_dict["model_type"] or ("CMLP" in args_dict["model_type"] and "REDCLIFF" in args_dict["model_type"]):
            args_dict["num_sims"] = int(new_args_dict["num_sims"])
            args_dict["embed_hidden_sizes"] = parse_input_list_of_ints(new_args_dict["embed_hidden_sizes"])
            args_dict['batch_size'] = int(new_args_dict["batch_size"])
            args_dict['gen_eps'] = float(new_args_dict['gen_eps'])
            args_dict['gen_weight_decay'] = float(new_args_dict['gen_weight_decay'])
            args_dict["max_iter"] = int(new_args_dict["max_iter"]) 
            args_dict["lookback"] = int(new_args_dict["lookback"]) 
            args_dict["check_every"] = int(new_args_dict["check_every"]) 
            args_dict["verbose"] = int(new_args_dict["verbose"]) 
            args_dict["output_length"] = int(new_args_dict["output_length"])
            args_dict["wavelet_level"] = None if new_args_dict["wavelet_level"] == "None" else int(new_args_dict["wavelet_level"])
            args_dict["gen_hidden"] = parse_input_list_of_ints(new_args_dict["gen_hidden"])
            args_dict["gen_lr"] = float(new_args_dict["gen_lr"])
            args_dict["input_length"] = int(new_args_dict["gen_lag_and_input_len"])
            args_dict["gen_lag"] = int(new_args_dict["gen_lag_and_input_len"])
            args_dict["coeff_dict"] = {
                "FORECAST_COEFF": float(new_args_dict["FORECAST_COEFF"]), 
                "ADJ_L1_REG_COEFF": float(new_args_dict["ADJ_L1_REG_COEFF"]),
            }
            if args_dict["wavelet_level"] is not None:
                args_dict["signal_format"] = "wavelet_decomp"
            else:
                args_dict["signal_format"] = "original"
            
            if "REDCLIFF" not in args_dict["model_type"]:
                args_dict["coeff_dict"]["DAGNESS_REG_COEFF"] = float(new_args_dict["DAGNESS_REG_COEFF"])
                args_dict["coeff_dict"]["DAGNESS_LAG_COEFF"] = float(new_args_dict["DAGNESS_LAG_COEFF"])
                args_dict["coeff_dict"]["DAGNESS_NODE_COEFF"] = float(new_args_dict["DAGNESS_NODE_COEFF"])
            else:
                args_dict["num_factors"] = int(new_args_dict["num_factors"])
                args_dict["num_supervised_factors"] = int(new_args_dict["num_supervised_factors"])
                if "_S_" in args_dict["model_type"]:
                    args_dict["embed_lag"] = int(new_args_dict["embed_lag"])
                    args_dict["use_sigmoid_restriction"] = bool(int(new_args_dict["use_sigmoid_restriction"]))
                    args_dict["factor_score_embedder_type"] = new_args_dict["factor_score_embedder_type"]
                    if args_dict["factor_score_embedder_type"] == "cEmbedder":
                        args_dict["factor_score_embedder_args"] = [
                            ("sigmoid_eccentricity_coeff", float(new_args_dict["sigmoid_eccentricity_coeff"])), 
                            ("lag", int(new_args_dict["embed_lag"])), 
                            ("hidden", copy.deepcopy(args_dict["embed_hidden_sizes"]))
                        ]
                    elif args_dict["factor_score_embedder_type"] == "DGCNN":
                        args_dict["factor_score_embedder_args"] = [
                            ("num_features_per_node", int(new_args_dict["embed_lag"])), 
                            ("num_graph_conv_layers", int(new_args_dict["embed_num_graph_conv_layers"])), 
                            ("num_hidden_nodes", int(new_args_dict["embed_num_hidden_nodes"])), 
                            ("sigmoid_eccentricity_coeff", float(new_args_dict["sigmoid_eccentricity_coeff"]))
                        ]
                    elif args_dict["factor_score_embedder_type"] == "Vanilla_Embedder":
                        args_dict["factor_score_embedder_args"] = []
                    else:
                        raise ValueError("input_argument_utils.read_in_model_args: UNRECOGNIZED args_dict['factor_score_embedder_type'] == "+str(args_dict["factor_score_embedder_type"]))
                    args_dict["primary_gc_est_mode"] = new_args_dict["primary_gc_est_mode"]
                    args_dict["forward_pass_mode"] = new_args_dict["forward_pass_mode"]
                    
                args_dict["coeff_dict"]["FACTOR_SCORE_COEFF"] = float(new_args_dict["FACTOR_SCORE_COEFF"])
                args_dict["coeff_dict"]["DAGNESS_REG_COEFF"] = float(new_args_dict["DAGNESS_REG_COEFF"])
                args_dict["coeff_dict"]["DAGNESS_LAG_COEFF"] = float(new_args_dict["DAGNESS_LAG_COEFF"])
                args_dict["coeff_dict"]["DAGNESS_NODE_COEFF"] = float(new_args_dict["DAGNESS_NODE_COEFF"])
                if "_S_" in args_dict["model_type"]:
                    args_dict["coeff_dict"]["FACTOR_WEIGHT_L1_COEFF"] = float(new_args_dict["FACTOR_WEIGHT_L1_COEFF"])
                    args_dict["coeff_dict"]["FACTOR_COS_SIM_COEFF"] = float(new_args_dict["FACTOR_COS_SIM_COEFF"])
                    if "FACTOR_WEIGHT_SMOOTHING_PENALTY_COEFF" in new_args_dict.keys():
                        args_dict["coeff_dict"]["FACTOR_WEIGHT_SMOOTHING_PENALTY_COEFF"] = float(new_args_dict["FACTOR_WEIGHT_SMOOTHING_PENALTY_COEFF"])

                print("read_in_model_args: SETTING TRAINING MODE new_args_dict['training_mode'] == ", new_args_dict['training_mode'])
                args_dict["training_mode"] = new_args_dict["training_mode"]
                print("read_in_model_args: args_dict['training_mode'] == ", args_dict['training_mode'])
                args_dict["embed_lr"] = float(new_args_dict["embed_lr"])
                args_dict['embed_eps'] = float(new_args_dict['embed_eps'])
                args_dict['embed_weight_decay'] = float(new_args_dict['embed_weight_decay'])

                args_dict["num_pretrain_epochs"] = int(new_args_dict["num_pretrain_epochs"])
                if "_S_" in args_dict["model_type"]:
                    args_dict["num_acclimation_epochs"] = int(new_args_dict["num_acclimation_epochs"])
                args_dict["prior_factors_path"] = None if new_args_dict["prior_factors_path"] == "None" else new_args_dict["prior_factors_path"]#None, 
                args_dict["cost_criteria"] = new_args_dict["cost_criteria"]#="CosineSimilarity", 
                args_dict["unsupervised_start_index"] = int(new_args_dict["unsupervised_start_index"])#=0, 
                args_dict["max_factor_prior_batches"] = int(new_args_dict["max_factor_prior_batches"])#=10, 
                args_dict["stopping_criteria_forecast_coeff"] = float(new_args_dict["stopping_criteria_forecast_coeff"])#=1., 
                args_dict["stopping_criteria_factor_coeff"] = float(new_args_dict["stopping_criteria_factor_coeff"])#=1., 
                args_dict["stopping_criteria_cosSim_coeff"] = float(new_args_dict["stopping_criteria_cosSim_coeff"])#=1.
                
                args_dict["deltaConEps"] = float(new_args_dict["deltaConEps"])#, #0.1, 
                args_dict["in_degree_coeff"] = float(new_args_dict["in_degree_coeff"])#, #1., 
                args_dict["out_degree_coeff"] = float(new_args_dict["out_degree_coeff"])#, #1., 

        elif "cLSTM" in args_dict["model_type"] or ("CLSTM" in args_dict["model_type"] and "REDCLIFF" in args_dict["model_type"]):
            args_dict["num_sims"] = int(new_args_dict["num_sims"])
            args_dict["embed_hidden_sizes"] = parse_input_list_of_ints(new_args_dict["embed_hidden_sizes"])
            args_dict["coeff_dict"] = {
                "FORECAST_COEFF": float(new_args_dict["FORECAST_COEFF"]), # "RIDGE_REG_COEFF": float(new_args_dict["RIDGE_REG_COEFF"]), 
                "ADJ_L1_REG_COEFF": float(new_args_dict["ADJ_L1_REG_COEFF"]),
                "DAGNESS_REG_COEFF": float(new_args_dict["DAGNESS_REG_COEFF"])
            }
            args_dict['batch_size'] = int(new_args_dict["batch_size"])
            args_dict['gen_eps'] = float(new_args_dict['gen_eps'])
            args_dict['gen_weight_decay'] = float(new_args_dict['gen_weight_decay'])
            args_dict["max_iter"] = int(new_args_dict["max_iter"]) 
            args_dict["lookback"] = int(new_args_dict["lookback"]) 
            args_dict["check_every"] = int(new_args_dict["check_every"]) 
            args_dict["verbose"] = int(new_args_dict["verbose"]) 
            args_dict["wavelet_level"] = None if new_args_dict["wavelet_level"] == "None" else int(new_args_dict["wavelet_level"])
            args_dict["gen_hidden"] = int(new_args_dict["gen_hidden"])
            args_dict["gen_lr"] = float(new_args_dict["gen_lr"])
            args_dict["context"] = int(new_args_dict["context"])
            args_dict["max_input_length"] = int(new_args_dict["max_input_length"])

            if args_dict["wavelet_level"] is not None:
                args_dict["signal_format"] = "wavelet_decomp"
            else:
                args_dict["signal_format"] = "original"

            if "REDCLIFF" in args_dict["model_type"]:
                args_dict["num_factors"] = int(new_args_dict["num_factors"])
                args_dict["num_supervised_factors"] = int(new_args_dict["num_supervised_factors"])
                if "_S_" in args_dict["model_type"]:
                    args_dict["num_in_timesteps"] = int(new_args_dict["embed_lag"])
                    args_dict["use_sigmoid_restriction"] = bool(int(new_args_dict["use_sigmoid_restriction"]))
                    args_dict["factor_score_embedder_type"] = new_args_dict["factor_score_embedder_type"]
                    if args_dict["factor_score_embedder_type"] == "cEmbedder":
                        args_dict["factor_score_embedder_args"] = [
                            ("sigmoid_eccentricity_coeff", float(new_args_dict["sigmoid_eccentricity_coeff"])), 
                            ("lag", int(new_args_dict["embed_lag"])), 
                            ("hidden", copy.deepcopy(args_dict["embed_hidden_sizes"]))
                        ]
                    elif args_dict["factor_score_embedder_type"] == "DGCNN":
                        args_dict["factor_score_embedder_args"] = [
                            ("num_features_per_node", int(new_args_dict["embed_lag"])), 
                            ("num_graph_conv_layers", int(new_args_dict["embed_num_graph_conv_layers"])), 
                            ("num_hidden_nodes", int(new_args_dict["embed_num_hidden_nodes"])), 
                            ("sigmoid_eccentricity_coeff", float(new_args_dict["sigmoid_eccentricity_coeff"]))
                        ]
                    elif args_dict["factor_score_embedder_type"] == "Vanilla_Embedder":
                        args_dict["factor_score_embedder_args"] = []
                    else:
                        raise ValueError("input_argument_utils.read_in_model_args: UNRECOGNIZED args_dict['factor_score_embedder_type'] == "+str(args_dict["factor_score_embedder_type"]))
                    args_dict["primary_gc_est_mode"] = new_args_dict["primary_gc_est_mode"]
                    args_dict["forward_pass_mode"] = new_args_dict["forward_pass_mode"]
                    
                args_dict["coeff_dict"]["FACTOR_SCORE_COEFF"] = float(new_args_dict["FACTOR_SCORE_COEFF"])
                args_dict["coeff_dict"]["DAGNESS_REG_COEFF"] = float(new_args_dict["DAGNESS_REG_COEFF"])
                args_dict["coeff_dict"]["DAGNESS_LAG_COEFF"] = 0#float(new_args_dict["DAGNESS_LAG_COEFF"])
                args_dict["coeff_dict"]["DAGNESS_NODE_COEFF"] = 0# float(new_args_dict["DAGNESS_NODE_COEFF"])
                if "_S_" in args_dict["model_type"]:
                    args_dict["coeff_dict"]["FACTOR_WEIGHT_L1_COEFF"] = float(new_args_dict["FACTOR_WEIGHT_L1_COEFF"])
                    args_dict["coeff_dict"]["FACTOR_COS_SIM_COEFF"] = float(new_args_dict["FACTOR_COS_SIM_COEFF"])
                    if "FACTOR_WEIGHT_SMOOTHING_PENALTY_COEFF" in new_args_dict.keys():
                        args_dict["coeff_dict"]["FACTOR_WEIGHT_SMOOTHING_PENALTY_COEFF"] = float(new_args_dict["FACTOR_WEIGHT_SMOOTHING_PENALTY_COEFF"])

                print("read_in_model_args: SETTING TRAINING MODE new_args_dict['training_mode'] == ", new_args_dict['training_mode'])
                args_dict["training_mode"] = new_args_dict["training_mode"]
                print("read_in_model_args: args_dict['training_mode'] == ", args_dict['training_mode'])

                args_dict["embed_lr"] = float(new_args_dict["embed_lr"])
                args_dict['embed_eps'] = float(new_args_dict['embed_eps'])
                args_dict['embed_weight_decay'] = float(new_args_dict['embed_weight_decay'])

                args_dict["num_pretrain_epochs"] = int(new_args_dict["num_pretrain_epochs"])
                if "_S_" in args_dict["model_type"]:
                    args_dict["num_acclimation_epochs"] = int(new_args_dict["num_acclimation_epochs"])
                args_dict["prior_factors_path"] = None if new_args_dict["prior_factors_path"] == "None" else new_args_dict["prior_factors_path"]#None, 
                args_dict["cost_criteria"] = new_args_dict["cost_criteria"]#="CosineSimilarity", 
                args_dict["unsupervised_start_index"] = int(new_args_dict["unsupervised_start_index"])#=0, 
                args_dict["max_factor_prior_batches"] = int(new_args_dict["max_factor_prior_batches"])#=10, 
                args_dict["stopping_criteria_forecast_coeff"] = float(new_args_dict["stopping_criteria_forecast_coeff"])#=1., 
                args_dict["stopping_criteria_factor_coeff"] = float(new_args_dict["stopping_criteria_factor_coeff"])#=1., 
                args_dict["stopping_criteria_cosSim_coeff"] = float(new_args_dict["stopping_criteria_cosSim_coeff"])#=1.
                
                args_dict["deltaConEps"] = float(new_args_dict["deltaConEps"])#, #0.1, 
                args_dict["in_degree_coeff"] = float(new_args_dict["in_degree_coeff"])#, #1., 
                args_dict["out_degree_coeff"] = float(new_args_dict["out_degree_coeff"])#, #1., 

        elif "DCSFA" in args_dict["model_type"]:
            args_dict['batch_size'] = int(new_args_dict["batch_size"])
            args_dict["best_model_name"] = new_args_dict["best_model_name"] # best_model_name="dCSFA-NMF-best-model.pt",
            args_dict["num_high_level_node_features"] = int(new_args_dict["num_high_level_node_features"]) # this value is reported in the logs if you are having trouble determining it - ctrl+f for "NormalizedSyntheticWVARDataset.__getitem__: len(self.freq_bins) =="
            args_dict["num_node_features"] = int(new_args_dict["num_node_features"])
            args_dict["n_components"] = int(new_args_dict["n_components"])
            args_dict["n_sup_networks"] = int(new_args_dict["n_sup_networks"])
            args_dict["signal_format"] = new_args_dict["signal_format"]
            args_dict["h"] = int(new_args_dict["h"])
            args_dict["momentum"] = float(new_args_dict["momentum"])
            args_dict["lr"] = float(new_args_dict["lr"])
            args_dict["recon_weight"] = float(new_args_dict["recon_weight"])
            args_dict["sup_weight"] = float(new_args_dict["sup_weight"])
            args_dict["sup_recon_weight"] = float(new_args_dict["sup_recon_weight"])
            args_dict["sup_smoothness_weight"] = float(new_args_dict["sup_smoothness_weight"])
            args_dict["n_epochs"] = int(new_args_dict["n_epochs"])
            args_dict["n_pre_epochs"] = int(new_args_dict["n_pre_epochs"])
            args_dict["nmf_max_iter"] = int(new_args_dict["nmf_max_iter"])
            args_dict["dirspec_params"] = {
                "fs": 1000,
                "min_freq": 0.0,
                "max_freq": 250.0,
                "directed_spectrum": True,
                "csd_params": {
                    "detrend": "constant",
                    "window": "hann",
                    "nperseg": args_dict["num_node_features"], 
                    "noverlap": int(args_dict["num_node_features"]*0.5), 
                    "nfft": None,
                }, 
            }
            
        elif "DGCNN" in args_dict["model_type"]:
            if "REDCLIFF" not in args_dict["model_type"]:
                args_dict["num_classes"] = int(new_args_dict["num_classes"])
            args_dict['batch_size'] = int(new_args_dict["batch_size"])
            args_dict['gen_eps'] = float(new_args_dict['gen_eps'])
            args_dict['gen_weight_decay'] = float(new_args_dict['gen_weight_decay'])
            args_dict["max_iter"] = int(new_args_dict["max_iter"]) 
            args_dict["lookback"] = int(new_args_dict["lookback"]) 
            args_dict["check_every"] = int(new_args_dict["check_every"]) 
            args_dict["verbose"] = int(new_args_dict["verbose"]) 
            args_dict["num_features_per_node"] = int(new_args_dict["num_features_per_node"])
            args_dict["num_graph_conv_layers"] = int(new_args_dict["num_graph_conv_layers"])
            args_dict["num_hidden_nodes"] = int(new_args_dict["num_hidden_nodes"])
            args_dict["wavelet_level"] = 0 if new_args_dict["wavelet_level"] == "None" else int(new_args_dict["wavelet_level"])
            args_dict["num_wavelets_per_chan"] = int(new_args_dict["num_wavelets_per_chan"])
            args_dict["gen_lr"] = float(new_args_dict["gen_lr"])
            if args_dict["wavelet_level"] is not None and args_dict["wavelet_level"] != 0:
                args_dict["signal_format"] = "wavelet_decomp"
            else:
                args_dict["signal_format"] = "original"
                
            if "REDCLIFF" in args_dict["model_type"]:
                args_dict["num_sims"] = int(new_args_dict["num_sims"])
                args_dict["embed_hidden_sizes"] = parse_input_list_of_ints(new_args_dict["embed_hidden_sizes"])
                args_dict["coeff_dict"] = {
                    "FORECAST_COEFF": float(new_args_dict["FORECAST_COEFF"]), # "RIDGE_REG_COEFF": float(new_args_dict["RIDGE_REG_COEFF"]), 
                    "ADJ_L1_REG_COEFF": float(new_args_dict["ADJ_L1_REG_COEFF"]),
                    "DAGNESS_REG_COEFF": float(new_args_dict["DAGNESS_REG_COEFF"])
                }
                
                args_dict["num_factors"] = int(new_args_dict["num_factors"])
                args_dict["num_supervised_factors"] = int(new_args_dict["num_supervised_factors"])
                if "_S_" in args_dict["model_type"]:
                    args_dict["embed_num_features_per_node"] = int(new_args_dict["embed_lag"])
                    args_dict["use_sigmoid_restriction"] = bool(int(new_args_dict["use_sigmoid_restriction"]))
                    args_dict["factor_score_embedder_type"] = new_args_dict["factor_score_embedder_type"]
                    if args_dict["factor_score_embedder_type"] == "cEmbedder":
                        args_dict["factor_score_embedder_args"] = [
                            ("sigmoid_eccentricity_coeff", float(new_args_dict["sigmoid_eccentricity_coeff"])), 
                            ("lag", int(new_args_dict["embed_lag"])), 
                            ("hidden", copy.deepcopy(args_dict["embed_hidden_sizes"]))
                        ]
                    elif args_dict["factor_score_embedder_type"] == "DGCNN":
                        args_dict["factor_score_embedder_args"] = [
                            ("num_features_per_node", int(new_args_dict["embed_lag"])), 
                            ("num_graph_conv_layers", int(new_args_dict["embed_num_graph_conv_layers"])), 
                            ("num_hidden_nodes", int(new_args_dict["embed_num_hidden_nodes"])), 
                            ("sigmoid_eccentricity_coeff", float(new_args_dict["sigmoid_eccentricity_coeff"]))
                        ]
                    elif args_dict["factor_score_embedder_type"] == "Vanilla_Embedder":
                        args_dict["factor_score_embedder_args"] = []
                    else:
                        raise ValueError("input_argument_utils.read_in_model_args: UNRECOGNIZED args_dict['factor_score_embedder_type'] == "+str(args_dict["factor_score_embedder_type"]))
                    args_dict["primary_gc_est_mode"] = new_args_dict["primary_gc_est_mode"]
                    args_dict["forward_pass_mode"] = new_args_dict["forward_pass_mode"]
                    
                args_dict["coeff_dict"]["FACTOR_SCORE_COEFF"] = float(new_args_dict["FACTOR_SCORE_COEFF"])
                args_dict["coeff_dict"]["DAGNESS_REG_COEFF"] = float(new_args_dict["DAGNESS_REG_COEFF"])
                args_dict["coeff_dict"]["DAGNESS_LAG_COEFF"] = 0#float(new_args_dict["DAGNESS_LAG_COEFF"])
                args_dict["coeff_dict"]["DAGNESS_NODE_COEFF"] = 0# float(new_args_dict["DAGNESS_NODE_COEFF"])
                if "_S_" in args_dict["model_type"]:
                    args_dict["coeff_dict"]["FACTOR_WEIGHT_L1_COEFF"] = float(new_args_dict["FACTOR_WEIGHT_L1_COEFF"])
                    args_dict["coeff_dict"]["FACTOR_COS_SIM_COEFF"] = float(new_args_dict["FACTOR_COS_SIM_COEFF"])
                    if "FACTOR_WEIGHT_SMOOTHING_PENALTY_COEFF" in new_args_dict.keys():
                        args_dict["coeff_dict"]["FACTOR_WEIGHT_SMOOTHING_PENALTY_COEFF"] = float(new_args_dict["FACTOR_WEIGHT_SMOOTHING_PENALTY_COEFF"])

                print("read_in_model_args: SETTING TRAINING MODE new_args_dict['training_mode'] == ", new_args_dict['training_mode'])
                args_dict["training_mode"] = new_args_dict["training_mode"]
                print("read_in_model_args: args_dict['training_mode'] == ", args_dict['training_mode'])

                args_dict["embed_lr"] = float(new_args_dict["embed_lr"])
                args_dict['embed_eps'] = float(new_args_dict['embed_eps'])
                args_dict['embed_weight_decay'] = float(new_args_dict['embed_weight_decay'])

                args_dict["num_pretrain_epochs"] = int(new_args_dict["num_pretrain_epochs"])
                if "_S_" in args_dict["model_type"]:
                    args_dict["num_acclimation_epochs"] = int(new_args_dict["num_acclimation_epochs"])
                args_dict["prior_factors_path"] = None if new_args_dict["prior_factors_path"] == "None" else new_args_dict["prior_factors_path"]#None, 
                args_dict["cost_criteria"] = new_args_dict["cost_criteria"]#="CosineSimilarity", 
                args_dict["unsupervised_start_index"] = int(new_args_dict["unsupervised_start_index"])#=0, 
                args_dict["max_factor_prior_batches"] = int(new_args_dict["max_factor_prior_batches"])#=10, 
                args_dict["stopping_criteria_forecast_coeff"] = float(new_args_dict["stopping_criteria_forecast_coeff"])#=1., 
                args_dict["stopping_criteria_factor_coeff"] = float(new_args_dict["stopping_criteria_factor_coeff"])#=1., 
                args_dict["stopping_criteria_cosSim_coeff"] = float(new_args_dict["stopping_criteria_cosSim_coeff"])#=1.
                
                args_dict["deltaConEps"] = float(new_args_dict["deltaConEps"])#, #0.1, 
                args_dict["in_degree_coeff"] = float(new_args_dict["in_degree_coeff"])#, #1., 
                args_dict["out_degree_coeff"] = float(new_args_dict["out_degree_coeff"])#, #1., 

        elif "DYNOTEARS" in args_dict["model_type"]:
            
            print("args_dict.keys() == ", args_dict.keys(), flush=True)

            args_dict["signal_format"] = new_args_dict["signal_format"]
            args_dict["lambda_w"] = float(new_args_dict["lambda_w"])
            args_dict["lambda_a"] = float(new_args_dict["lambda_a"])
            args_dict["max_iter"] = int(new_args_dict["max_iter"])
            args_dict["h_tol"] = float(new_args_dict["h_tol"])
            args_dict["w_threshold"] = float(new_args_dict["w_threshold"])
            args_dict["tabu_edges"] = None if new_args_dict["tabu_edges"] == "None" else new_args_dict["tabu_edges"]
            args_dict["tabu_parent_nodes"] = None if new_args_dict["tabu_parent_nodes"] == "None" else new_args_dict["tabu_parent_nodes"]
            args_dict["tabu_child_nodes"] = None if new_args_dict["tabu_child_nodes"] == "None" else new_args_dict["tabu_child_nodes"]
            
            args_dict["X_train"] = None
            args_dict["X_val"] = None
            args_dict["lag_size"] = int(new_args_dict["lag_size"])

            if "Vanilla" not in args_dict["model_type"]:
                args_dict['batch_size'] = int(new_args_dict["batch_size"])
                args_dict["grad_step"] = float(new_args_dict["grad_step"])
                args_dict["wa_est"] = None if new_args_dict["wa_est"] == "None" else new_args_dict["wa_est"]
                args_dict["rho"] = float(new_args_dict["rho"])
                args_dict["alpha"] = float(new_args_dict["alpha"])
                args_dict["h_value"] = np.inf if new_args_dict["h_value"] == "inf" else float(new_args_dict["h_value"])
                args_dict["h_new"] = np.inf if new_args_dict["h_new"] == "inf" else float(new_args_dict["h_new"])

                args_dict["max_data_iter"] = int(new_args_dict["max_data_iter"])
                args_dict["iter_start"] = int(new_args_dict["iter_start"])
                args_dict["num_iters_prior_to_stop"] = int(new_args_dict["num_iters_prior_to_stop"])
                args_dict["reuse_rho"] = bool(int(new_args_dict["reuse_rho"]))
                args_dict["reuse_alpha"] = bool(int(new_args_dict["reuse_alpha"]))
                args_dict["reuse_h_val"] = bool(int(new_args_dict["reuse_h_val"]))
                args_dict["reuse_h_new"] = bool(int(new_args_dict["reuse_h_new"]))
                args_dict["check_every"] = int(new_args_dict["check_every"])
        
        elif "NAVAR" in args_dict["model_type"]:
            args_dict["num_nodes"] = int(new_args_dict["num_nodes"])
            args_dict["num_hidden"] = int(new_args_dict["num_hidden"])
            args_dict["maxlags"] = int(new_args_dict["maxlags"])
            args_dict["hidden_layers"] = int(new_args_dict["hidden_layers"])
            args_dict["dropout"] = float(new_args_dict["dropout"])

            args_dict["X_train"] = None
            args_dict["y_train"] = None
            args_dict["X_val"] = None#None, 
            args_dict["y_val"] = None#,None, 
            args_dict["val_proportion"] = float(new_args_dict["val_proportion"])#0.0, 
            args_dict["epochs"] = int(new_args_dict["epochs"])#,200, 
            args_dict["batch_size"] = int(new_args_dict["batch_size"])#,300, 
            args_dict["check_every"] = int(new_args_dict["check_every"])#1000
            args_dict["lambda1"] = float(new_args_dict["lambda1"])#lambda1=0
            
            args_dict["learning_rate"] = float(new_args_dict["learning_rate"])
            args_dict["weight_decay"] = float(new_args_dict["weight_decay"])
            
            args_dict["signal_format"] = new_args_dict["signal_format"]

            if "MLP" in args_dict["model_type"]:
                args_dict["split_timeseries"] = bool(int(new_args_dict["split_timeseries"]))#,False, 
            
        else:
            raise ValueError("read_in_model_args: model_type == "+str(args_dict["model_type"])+" IS NOT SUPPORTED (CURRENTLY)")

    return args_dict


def read_in_data_args(args_dict, include_gc_views_for_eval=False, read_in_gc_factors_for_eval=False):
    with open(args_dict["data_cached_args_file"], 'r') as infile:
        new_args_dict = json.load(infile)
        print("input_argument_utils.read_in_data_args: args_dict['data_cached_args_file'] == ", args_dict['data_cached_args_file'])
        args_dict["data_root_path"] = new_args_dict["data_root_path"]
        print("input_argument_utils.read_in_data_args: args_dict['data_root_path'] == ", args_dict['data_root_path'])
        args_dict["num_channels"] = int(new_args_dict["num_channels"])
        print("input_argument_utils.read_in_data_args: args_dict['num_channels'] == ", args_dict['num_channels'])
        
        print("input_argument_utils.read_in_data_args: args_dict['model_type'] == ", args_dict['model_type'])
        if "cMLP" in args_dict["model_type"] or "REDCLIFF" in args_dict["model_type"]:
            args_dict["true_GC_tensor"] = None
            args_dict["true_GC_factors"] = []
            for key in new_args_dict.keys():
                if "adjacency_tensor" in key:
                    curr_tensor = parse_tensor_string_representation(new_args_dict[key])
                    curr_tensor = torch.from_numpy(curr_tensor).cpu().data.numpy()[:,:,::-1]
                    args_dict["true_GC_factors"].append(curr_tensor)
                    if args_dict["true_GC_tensor"] is None:
                        args_dict["true_GC_tensor"] = curr_tensor
                    else:
                        args_dict["true_GC_tensor"] = args_dict["true_GC_tensor"] + curr_tensor
                    currGC = [curr_tensor]
                    plot_gc_est_comparissons_by_factor(currGC, None, args_dict['save_root_path']+os.sep+key+"_"+args_dict["data_set_name"]+"_visualization.png", include_lags=True)
            args_dict["true_GC_tensor"] = [curr_tensor]

        elif "cLSTM" in args_dict["model_type"]:
            args_dict["true_GC_tensor"] = None
            for key in new_args_dict.keys():
                if "adjacency_tensor" in key:
                    curr_lagged_tensor = parse_tensor_string_representation(new_args_dict[key])
                    curr_lagged_tensor = curr_lagged_tensor[:,:,::-1]
                    curr_nonTemporal_matrix = np.sum(curr_lagged_tensor, axis=2)
                    if args_dict["true_GC_tensor"] is None:
                        args_dict["true_GC_tensor"] = curr_nonTemporal_matrix
                    else:
                        args_dict["true_GC_tensor"] = args_dict["true_GC_tensor"] + curr_nonTemporal_matrix
                    curr_lagged_GC = [torch.from_numpy(curr_lagged_tensor.copy()).cpu().data.numpy()]
                    plot_gc_est_comparissons_by_factor(
                        curr_lagged_GC, 
                        None, 
                        args_dict['save_root_path']+os.sep+key+"_"+args_dict["data_set_name"]+"_visualization_WITH_LAGS.png", 
                        include_lags=True
                    )
                    curr_nonTemporal_GC = [torch.from_numpy(curr_nonTemporal_matrix).cpu().data.numpy()]
                    plot_gc_est_comparissons_by_factor(
                        curr_nonTemporal_GC, 
                        None, 
                        args_dict['save_root_path']+os.sep+key+"_"+args_dict["data_set_name"]+"_visualization_WITHOUT_LAGS.png", 
                        include_lags=False
                    )
            args_dict["true_GC_tensor"] = [torch.from_numpy(args_dict["true_GC_tensor"]).cpu().data.numpy()]

        elif "DCSFA" in args_dict["model_type"]:
            args_dict["true_GC_tensor"] = None
            for key in new_args_dict.keys():
                if "adjacency_tensor" in key:
                    curr_lagged_tensor = parse_tensor_string_representation(new_args_dict[key])
                    curr_lagged_tensor = curr_lagged_tensor[:,:,::-1]
                    curr_nonTemporal_matrix = np.sum(curr_lagged_tensor, axis=2)
                    if args_dict["true_GC_tensor"] is None:
                        args_dict["true_GC_tensor"] = curr_nonTemporal_matrix
                    else:
                        args_dict["true_GC_tensor"] = args_dict["true_GC_tensor"] + curr_nonTemporal_matrix
                    curr_lagged_GC = [torch.from_numpy(curr_lagged_tensor.copy()).cpu().data.numpy()]
                    plot_gc_est_comparissons_by_factor(
                        curr_lagged_GC, 
                        None, 
                        args_dict['save_root_path']+os.sep+key+"_"+args_dict["data_set_name"]+"_visualization_WITH_LAGS.png", 
                        include_lags=True
                    )
                    curr_nonTemporal_GC = [torch.from_numpy(curr_nonTemporal_matrix).cpu().data.numpy()]
                    plot_gc_est_comparissons_by_factor(
                        curr_nonTemporal_GC, 
                        None, 
                        args_dict['save_root_path']+os.sep+key+"_"+args_dict["data_set_name"]+"_visualization_WITHOUT_LAGS.png", 
                        include_lags=False
                    )
            args_dict["true_GC_tensor"] = [torch.from_numpy(args_dict["true_GC_tensor"]).cpu().data.numpy()]

        elif "DGCNN" in args_dict["model_type"]:
            args_dict["true_GC_tensor"] = None
            for key in new_args_dict.keys():
                if "adjacency_tensor" in key:
                    curr_lagged_tensor = parse_tensor_string_representation(new_args_dict[key])
                    curr_lagged_tensor = curr_lagged_tensor[:,:,::-1]
                    curr_nonTemporal_matrix = np.sum(curr_lagged_tensor, axis=2)
                    if args_dict["true_GC_tensor"] is None:
                        args_dict["true_GC_tensor"] = curr_nonTemporal_matrix
                    else:
                        args_dict["true_GC_tensor"] = args_dict["true_GC_tensor"] + curr_nonTemporal_matrix
                    curr_lagged_GC = [torch.from_numpy(curr_lagged_tensor.copy()).cpu().data.numpy()]
                    plot_gc_est_comparissons_by_factor(
                        curr_lagged_GC, 
                        None, 
                        args_dict['save_root_path']+os.sep+key+"_"+args_dict["data_set_name"]+"_visualization_WITH_LAGS.png", 
                        include_lags=True
                    )
                    curr_nonTemporal_GC = [torch.from_numpy(curr_nonTemporal_matrix).cpu().data.numpy()]
                    plot_gc_est_comparissons_by_factor(
                        curr_nonTemporal_GC, 
                        None, 
                        args_dict['save_root_path']+os.sep+key+"_"+args_dict["data_set_name"]+"_visualization_WITHOUT_LAGS.png", 
                        include_lags=False
                    )
            args_dict["true_GC_tensor"] = [args_dict["true_GC_tensor"]]

        elif "DYNOTEARS" in args_dict["model_type"]:
            args_dict["true_GC_tensor"] = None
            for key in new_args_dict.keys():
                if "adjacency_tensor" in key:
                    curr_lagged_tensor = parse_tensor_string_representation(new_args_dict[key])
                    curr_lagged_tensor = curr_lagged_tensor[:,:,::-1]
                    curr_nonTemporal_matrix = np.sum(curr_lagged_tensor, axis=2)
                    if args_dict["true_GC_tensor"] is None:
                        args_dict["true_GC_tensor"] = curr_nonTemporal_matrix
                    else:
                        args_dict["true_GC_tensor"] = args_dict["true_GC_tensor"] + curr_nonTemporal_matrix
                    curr_lagged_GC = [torch.from_numpy(curr_lagged_tensor.copy()).cpu().data.numpy()]
                    plot_gc_est_comparissons_by_factor(
                        curr_lagged_GC, 
                        None, 
                        args_dict['save_root_path']+os.sep+key+"_"+args_dict["data_set_name"]+"_visualization_WITH_LAGS.png", 
                        include_lags=True
                    )
                    curr_nonTemporal_GC = [torch.from_numpy(curr_nonTemporal_matrix).cpu().data.numpy()]
                    plot_gc_est_comparissons_by_factor(
                        curr_nonTemporal_GC, 
                        None, 
                        args_dict['save_root_path']+os.sep+key+"_"+args_dict["data_set_name"]+"_visualization_WITHOUT_LAGS.png", 
                        include_lags=False
                    )
            args_dict["true_GC_tensor"] = [torch.from_numpy(args_dict["true_GC_tensor"]).cpu().data.numpy()]
        
        elif "NAVAR" in args_dict["model_type"]:
            args_dict["true_GC_tensor"] = None
            for key in new_args_dict.keys():
                if "adjacency_tensor" in key:
                    curr_lagged_tensor = parse_tensor_string_representation(new_args_dict[key])
                    curr_lagged_tensor = curr_lagged_tensor[:,:,::-1]
                    curr_nonTemporal_matrix = np.sum(curr_lagged_tensor, axis=2)
                    if args_dict["true_GC_tensor"] is None:
                        args_dict["true_GC_tensor"] = curr_nonTemporal_matrix
                    else:
                        args_dict["true_GC_tensor"] = args_dict["true_GC_tensor"] + curr_nonTemporal_matrix
                    curr_lagged_GC = [torch.from_numpy(curr_lagged_tensor.copy()).cpu().data.numpy()]
                    plot_gc_est_comparissons_by_factor(
                        curr_lagged_GC, 
                        None, 
                        args_dict['save_path']+os.sep+key+"_"+args_dict["data_set_name"]+"_visualization_WITH_LAGS.png", 
                        include_lags=True
                    )
                    curr_nonTemporal_GC = [torch.from_numpy(curr_nonTemporal_matrix).cpu().data.numpy()]
                    plot_gc_est_comparissons_by_factor(
                        curr_nonTemporal_GC, 
                        None, 
                        args_dict['save_path']+os.sep+key+"_"+args_dict["data_set_name"]+"_visualization_WITHOUT_LAGS.png", 
                        include_lags=False
                    )
            args_dict["true_GC_tensor"] = [torch.from_numpy(args_dict["true_GC_tensor"]).cpu().data.numpy()]

        else:
            raise ValueError("read_in_data_args: model_type == "+str(args_dict["model_type"])+" IS NOT SUPPORTED (CURRENTLY)")
    
        if read_in_gc_factors_for_eval:
            try:
                print('general_utils.input_argument_utils.read_in_data_args: len(args_dict["true_GC_factors"]) == ', len(args_dict["true_GC_factors"]))
            except:
                args_dict["true_GC_factors"] = []
                for key in new_args_dict.keys():
                    if "adjacency_tensor" in key:
                        curr_tensor = parse_tensor_string_representation(new_args_dict[key])
                        curr_tensor = torch.from_numpy(curr_tensor).cpu().data.numpy()[:,:,::-1]
                        args_dict["true_GC_factors"].append(curr_tensor)
                print('general_utils.input_argument_utils.read_in_data_args: len(args_dict["true_GC_factors"]) == ', len(args_dict["true_GC_factors"]))


    if include_gc_views_for_eval: # DECRIMENTED 01/01/2024
        args_dict["true_lagged_GC_tensor"] = None
        args_dict["true_nontemporal_GC_tensor"] = None
        args_dict["true_lagged_GC_tensor_factors"] = [None, None, None, None]
        args_dict["true_nontemporal_GC_tensor_factors"] = [None, None, None, None]
        for key in new_args_dict.keys():
            if "adjacency_tensor" in key:
                curr_lagged_tensor = parse_tensor_string_representation(new_args_dict[key])
                curr_lagged_tensor = curr_lagged_tensor[:,:,::-1].copy() # lagged tensors have been saved in reverse-lag order, which we correct here for later comparissons with gc estimates
                curr_nonTemporal_matrix = np.sum(curr_lagged_tensor, axis=2)
                
                curr_factor_ind = int(key[3])-1 # convention that all tensor keys begin with "net" followed by an integer and "_"
                args_dict["true_lagged_GC_tensor_factors"][curr_factor_ind] = torch.from_numpy(curr_lagged_tensor).cpu().data.numpy()
                args_dict["true_nontemporal_GC_tensor_factors"][curr_factor_ind] = torch.from_numpy(curr_nonTemporal_matrix).cpu().data.numpy()

                if args_dict["true_lagged_GC_tensor"] is None:
                    args_dict["true_lagged_GC_tensor"] = curr_lagged_tensor
                else:
                    args_dict["true_lagged_GC_tensor"] = args_dict["true_lagged_GC_tensor"] + curr_lagged_tensor

                if args_dict["true_nontemporal_GC_tensor"] is None:
                    args_dict["true_nontemporal_GC_tensor"] = curr_nonTemporal_matrix
                else:
                    args_dict["true_nontemporal_GC_tensor"] = args_dict["true_nontemporal_GC_tensor"] + curr_nonTemporal_matrix
                
                curr_lagged_GC = [torch.from_numpy(curr_lagged_tensor).cpu().data.numpy()]
                plot_gc_est_comparissons_by_factor(
                    curr_lagged_GC, 
                    None, 
                    args_dict['save_root_path']+os.sep+key+"_visualization_WITH_LAGS.png", 
                    include_lags=True
                )
                curr_nonTemporal_GC = [torch.from_numpy(curr_nonTemporal_matrix).cpu().data.numpy()]
                plot_gc_est_comparissons_by_factor(
                    curr_nonTemporal_GC, 
                    None, 
                    args_dict['save_root_path']+os.sep+key+"_visualization_WITHOUT_LAGS.png", 
                    include_lags=False
                )
    return args_dict